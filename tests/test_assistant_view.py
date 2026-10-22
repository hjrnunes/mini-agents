"""Each MCP server hides the state-summary observer from ``tools/list``.

By default no client lists ``get_<domain>_state_summary``. ``--state-summary``
(or ``MINI_AGENTS_STATE_SUMMARY=1``) lists it again, except to a client that
sends ``X-Mini-Agents-View: assistant``. Every call by name reaches the
observer in every view. The tests drive the low-level ``tools/list`` handler
inside a fake request context; no transport is involved.
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


def _server(domain, *, listed=False):
    server = mcp_server.BUILDERS[domain]()
    server.list_observer = listed
    return server


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
def test_the_observer_is_hidden_by_default(domain):
    server = _server(domain)

    listed = _list(server, {})

    assert listed
    assert _observer(domain) not in listed


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_listing_outside_a_request_hides_the_observer_by_default(domain):
    server = _server(domain)

    listed = {tool.name for tool in asyncio.run(server.list_tools())}

    assert _observer(domain) not in listed


@pytest.mark.parametrize("domain", DOMAINS)
def test_enabled_the_observer_is_listed_beside_every_other_tool(domain):
    hidden = _list(_server(domain), {})

    listed = _list(_server(domain, listed=True), {})

    assert listed == hidden | {_observer(domain)}


@pytest.mark.parametrize("domain", DOMAINS)
def test_enabled_the_assistant_view_still_leaves_out_the_observer(domain):
    server = _server(domain, listed=True)
    full = _list(server, {})

    assistant = _list(server, {mcp_server.VIEW_HEADER: "assistant"})

    assert _observer(domain) in full
    assert assistant == full - {_observer(domain)}


@pytest.mark.parametrize("domain", DOMAINS)
def test_enabled_another_view_value_keeps_the_observer(domain):
    server = _server(domain, listed=True)

    listed = _list(server, {mcp_server.VIEW_HEADER: "orchestrator"})

    assert _observer(domain) in listed


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_hidden_observer_stays_callable_by_name(domain):
    server = _server(domain)
    _list(server, {mcp_server.VIEW_HEADER: "assistant"})

    content = asyncio.run(server.call_tool(_observer(domain), {}))

    summary = json.loads(content[0][0].text)
    assert "session_id" in summary


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_instructions_do_not_point_at_the_observer(domain):
    server = _server(domain)

    assert "state_summary" not in server.instructions
    assert f"Inspect {_observer(domain)}" not in server.instructions


class _FakeServer:
    def __init__(self):
        self.settings = SimpleNamespace(host=None, port=None)

    def run(self, transport):
        pass


@pytest.fixture
def built(monkeypatch):
    servers = []

    def builder(state=None):
        servers.append(_FakeServer())
        return servers[-1]

    monkeypatch.setitem(mcp_server.BUILDERS, "klarna", builder)
    monkeypatch.delenv(mcp_server.STATE_SUMMARY_ENV, raising=False)
    return servers


def _main(*extra):
    mcp_server.main(["--domain", "klarna", "--port", "1", *extra])


def test_main_hides_the_observer_without_the_flag(built):
    _main()

    assert built[0].list_observer is False


def test_the_flag_lists_the_observer(built):
    _main("--state-summary")

    assert built[0].list_observer is True


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_the_environment_variable_lists_the_observer(built, monkeypatch, value):
    monkeypatch.setenv(mcp_server.STATE_SUMMARY_ENV, value)

    _main()

    assert built[0].list_observer is True


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_other_environment_values_keep_the_observer_hidden(built, monkeypatch, value):
    monkeypatch.setenv(mcp_server.STATE_SUMMARY_ENV, value)

    _main()

    assert built[0].list_observer is False
