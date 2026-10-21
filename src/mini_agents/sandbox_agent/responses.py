"""One Responses request per turn, in stage 2's shape.

The body copies MiDojo's ``OpenAIResponsesAgentClient._create_response``:
``model``, ``input``, ``instructions``, one ``mcp`` tool with
``require_approval: never``, ``stream: false``, and ``previous_response_id``
from the second turn on. One difference: the tool carries no ``headers``. The
server reads the session token from its own environment, so the token never
passes through OGX.
"""

from __future__ import annotations

from typing import Any

import httpx

from mini_agents.sandbox_agent.failures import (
    GATEWAY_REQUEST_FAILED,
    GATEWAY_RESPONSE_FAILED,
    GATEWAY_RESPONSE_INVALID,
    AgentFailure,
)

MCP_SERVER_LABEL = "midojo"
RESPONSES_PATH = "/v1/responses"
_BODY_EXCERPT = 200


def request_body(
    *,
    model: str,
    message: str,
    instructions: str,
    mcp_url: str,
    previous_response_id: str | None,
    allowed_tools: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    # The package's exposed tools, as the Garak path sends them; None exposes
    # every tool the server serves.
    restriction = {"allowed_tools": list(allowed_tools)} if allowed_tools is not None else {}
    continuation = (
        {"previous_response_id": previous_response_id}
        if previous_response_id is not None
        else {}
    )
    return {
        "model": model,
        "input": message,
        "instructions": instructions,
        "tools": [
            {
                "type": "mcp",
                "server_label": MCP_SERVER_LABEL,
                "server_url": mcp_url,
                "require_approval": "never",
                **restriction,
            }
        ],
        "stream": False,
        **continuation,
    }


def reply_text(response: dict[str, Any]) -> str:
    """The reply, as the openai SDK's ``Response.output_text`` joins it."""

    texts = [
        part.get("text", "")
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type") == "message"
        for part in item.get("content") or []
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    return "".join(texts)


def create_response(
    gateway_url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """POST the body to OGX; return the response or fail with a named reason.

    Every failure here comes after the request was sent, so it counts one
    gateway request.
    """

    url = gateway_url.rstrip("/") + RESPONSES_PATH
    owned = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        try:
            reply = http.post(url, json=body, timeout=timeout)
        except httpx.HTTPError as exc:
            raise AgentFailure(
                GATEWAY_REQUEST_FAILED,
                f"POST {url}: {type(exc).__name__}: {exc}",
                gateway_requests=1,
            ) from exc
    finally:
        if owned:
            http.close()
    if not reply.is_success:
        raise AgentFailure(
            GATEWAY_REQUEST_FAILED,
            f"POST {url}: HTTP {reply.status_code} {reply.text[:_BODY_EXCERPT]}".rstrip(),
            gateway_requests=1,
        )
    try:
        response = reply.json()
    except ValueError as exc:
        raise AgentFailure(
            GATEWAY_RESPONSE_INVALID,
            f"POST {url}: the body is not JSON",
            gateway_requests=1,
        ) from exc
    if not isinstance(response, dict) or not isinstance(response.get("id"), str):
        raise AgentFailure(
            GATEWAY_RESPONSE_INVALID,
            f"POST {url}: the response has no id",
            gateway_requests=1,
        )
    if response.get("status") == "failed" or response.get("error"):
        raise AgentFailure(
            GATEWAY_RESPONSE_FAILED,
            f"response {response['id']}: {response.get('error')!r}",
            gateway_requests=1,
        )
    return response
