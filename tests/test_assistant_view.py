"""The assistant view of each MCP server hides the state-summary observer.

A client that sends ``X-Mini-Agents-View: assistant`` lists every tool except
``get_<domain>_state_summary``. Every other client, and every call by name,
still reaches the observer. The tests drive the low-level ``tools/list``
handler inside a fake request context; no transport is involved.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from mini_agents.kernel import mcp_server

DOMAINS = mcp_server.DOMAINS


def _observer(domain):
    return f"get_{domain}_state_summary"


def _list(server, headers):
    async def run():
        token = request_ctx.set(
            RequestContext(
                request_id=1,
                meta=None,
                session=None,
                lifespan_context=None,
                request=SimpleNamespace(headers=headers),
            )
        )
        try:
            handler = server._mcp_server.request_handlers[types.ListToolsRequest]
            result = await handler(types.ListToolsRequest(method="tools/list"))
        finally:
            request_ctx.reset(token)
        return {tool.name for tool in result.root.tools}

    return asyncio.run(run())


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_assistant_view_lists_every_tool_but_the_observer(domain):
    server = mcp_server.BUILDERS[domain]()
    full = _list(server, {})

    assistant = _list(server, {mcp_server.VIEW_HEADER: "assistant"})

    assert _observer(domain) in full
    assert assistant == full - {_observer(domain)}


@pytest.mark.parametrize("domain", DOMAINS)
def test_another_view_value_keeps_the_observer(domain):
    server = mcp_server.BUILDERS[domain]()

    listed = _list(server, {mcp_server.VIEW_HEADER: "orchestrator"})

    assert _observer(domain) in listed


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_listing_outside_a_request_keeps_the_observer(domain):
    server = mcp_server.BUILDERS[domain]()

    listed = {tool.name for tool in asyncio.run(server.list_tools())}

    assert _observer(domain) in listed


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_observer_stays_callable_by_name(domain):
    server = mcp_server.BUILDERS[domain]()
    _list(server, {mcp_server.VIEW_HEADER: "assistant"})

    content = asyncio.run(server.call_tool(_observer(domain), {}))

    summary = json.loads(content[0][0].text)
    assert "session_id" in summary
