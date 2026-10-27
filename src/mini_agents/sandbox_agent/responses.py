"""One Responses request per turn, in stage 2's shape.

The body copies MiDojo's ``OpenAIResponsesAgentClient._create_response``:
``model``, ``input``, ``instructions``, one ``mcp`` tool with
``require_approval: never``, ``stream: false``, and ``previous_response_id``
from the second turn on. One difference: the tool carries no ``headers``.
Behind MiDojo's host interceptor the tool's ``server_url`` is the
interceptor's and its ``authorization`` the interceptor's token; OGX sends it
as ``Authorization: Bearer <token>`` (it refuses that header in ``headers``)
and leaves the field out of the response it stores and returns.
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
from mini_agents.sandbox_agent.tool_endpoint import redact_text

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
    authorization: str | None = None,
) -> dict[str, Any]:
    # The package's exposed tools, as the Garak path sends them; None exposes
    # every tool the server serves.
    restriction = {"allowed_tools": list(allowed_tools)} if allowed_tools is not None else {}
    credential = {"authorization": authorization} if authorization is not None else {}
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
                **credential,
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
    secret: str | None = None,
) -> dict[str, Any]:
    """POST the body to OGX; return the response or fail with a named reason.

    Every failure here comes after the request was sent, so it counts one
    gateway request. ``secret`` (the interceptor's token) is taken out of
    every failure detail, out of OGX's answer before it is cut short.
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
                redact_text(f"POST {url}: {type(exc).__name__}: {exc}", secret),
                gateway_requests=1,
            ) from exc
    finally:
        if owned:
            http.close()
    if not reply.is_success:
        raise AgentFailure(
            GATEWAY_REQUEST_FAILED,
            f"POST {url}: HTTP {reply.status_code} "
            f"{redact_text(reply.text, secret)[:_BODY_EXCERPT]}".rstrip(),
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
            redact_text(f"response {response['id']}: {response.get('error')!r}", secret),
            gateway_requests=1,
        )
    return response
