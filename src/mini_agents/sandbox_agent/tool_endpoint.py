"""MiDojo's host interceptor, when the runtime names one.

In host interception mode MiDojo's openshell runtime gives the sandbox the
interceptor's streamable HTTP MCP URL as ``MIDOJO_TOOL_URL``
(``http://host.openshell.internal:<port>/mcp``) and the bearer token that
interceptor alone accepts as ``MIDOJO_TOOL_TOKEN``. The interceptor answers
401 to a request without ``Authorization: Bearer <token>`` and forwards the
rest to the real tools.

Both set: every turn's MCP tool points at the interceptor and carries the
token as the Responses tool's ``authorization``, which OGX sends as that
header. Neither set: the tool points at the domain's server in the sandbox,
as before. One set without the other is refused, never a fallback to the
server in the sandbox. A variable set to an empty or blank value counts as
unset.

The token stays out of every message: the endpoint's ``repr`` leaves it out,
``redact_text`` takes it out of a gateway's answer before it is cut short, and
``redact`` out of a failure's detail.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from mini_agents.sandbox_agent import failures
from mini_agents.sandbox_agent.failures import AgentFailure

TOOL_URL = "MIDOJO_TOOL_URL"
TOOL_TOKEN = "MIDOJO_TOOL_TOKEN"
REDACTED = f"<{TOOL_TOKEN}>"


@dataclass(frozen=True)
class ToolEndpoint:
    """The interceptor's MCP URL and the bearer token it accepts."""

    url: str
    token: str = field(repr=False)


def read_tool_endpoint(environ: Mapping[str, str]) -> ToolEndpoint | None:
    """The interceptor when both variables are set, ``None`` when neither is."""

    url = environ.get(TOOL_URL, "").strip()
    token = environ.get(TOOL_TOKEN, "").strip()
    if not url and not token:
        return None
    if not url or not token:
        given, missing = (TOOL_TOKEN, TOOL_URL) if token else (TOOL_URL, TOOL_TOKEN)
        raise AgentFailure(
            failures.INPUT_MISSING,
            f"{given} is set but {missing} is not: set both {TOOL_URL} and "
            f"{TOOL_TOKEN} to call the tools through MiDojo's interceptor, or "
            "neither to call the domain's server in the sandbox",
        )
    # The value stays out of the detail: a swapped pair would print the token.
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in ("http", "https") and bool(parsed.hostname)
    except ValueError:
        valid = False
    if not valid:
        raise AgentFailure(failures.INPUT_INVALID, f"{TOOL_URL} is not an http(s) URL")
    return ToolEndpoint(url=url, token=token)


def redact_text(text: str, secret: str | None) -> str:
    """``text`` with ``secret`` replaced by ``REDACTED``. Redact before
    cutting a text short: a cut through the secret leaves a prefix no later
    replacement matches."""

    return text.replace(secret, REDACTED) if secret else text


def redact(failure: AgentFailure, endpoint: ToolEndpoint | None) -> AgentFailure:
    """``failure`` with the token replaced by ``REDACTED`` in its detail."""

    if endpoint is None or endpoint.token not in failure.detail:
        return failure
    return AgentFailure(
        failure.reason,
        redact_text(failure.detail, endpoint.token),
        gateway_requests=failure.gateway_requests,
    )
