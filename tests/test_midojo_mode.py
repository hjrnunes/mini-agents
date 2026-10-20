"""MiDojo mode: load the evaluation's ledger, push it after each call, report the call.

A fake control plane on ``httpx.MockTransport`` serves ``GET /agent/environment``
per session token and records every request. Tools are called through the
low-level ``tools/call`` handler inside a fake request context, as a client's
``POST /messages/`` would reach them; no port is bound.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from mini_agents.kernel import mcp_server
from mini_agents.kernel.midojo_mode import SESSION_HEADER, TOKEN_ENV, MidojoBinding

CONTROL_URL = "http://control.test"
BASE = json.loads(
    (Path(mcp_server.__file__).parent.parent / "midojo-base.json").read_text()
)


def _klarna_environment(marker):
    environment = copy.deepcopy(BASE["targets"]["klarna"]["environment"])
    environment["orders"]["ORD-101"]["item"] = marker
    return environment


class ControlPlane:
    """Records each request; serves one environment per session token."""

    def __init__(self, environments, *, get=200, put=200, post=201):
        self.environments = environments
        self.statuses = {"GET": get, "PUT": put, "POST": post}
        self.requests = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        authorization = request.headers.get("Authorization", "")
        self.requests.append(
            SimpleNamespace(
                method=request.method,
                path=request.url.path,
                authorization=authorization,
                body=json.loads(request.content) if request.content else None,
            )
        )
        status = self.statuses[request.method]
        if isinstance(status, Exception):
            raise status
        if status >= 400:
            return httpx.Response(status, json={"detail": "refused"})
        if request.method == "GET":
            token = authorization.removeprefix("Bearer ")
            return httpx.Response(status, json=self.environments[token])
        return httpx.Response(status, json={})

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def methods(self):
        return [request.method for request in self.requests]

    def by_method(self, method):
        return [request for request in self.requests if request.method == method]


@pytest.fixture(autouse=True)
def no_token_in_the_environment(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "midojo_capture" / "calls.jsonl"


def _server(plane, log_path, domain="klarna"):
    server = mcp_server.BUILDERS[domain]()
    server.midojo = MidojoBinding(CONTROL_URL, log_path, http=plane.client())
    return server


def _call(server, name, arguments, headers):
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
            handler = server._mcp_server.request_handlers[types.CallToolRequest]
            result = await handler(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(name=name, arguments=arguments),
                )
            )
        finally:
            request_ctx.reset(token)
        return result.root

    return asyncio.run(run())


def _session(token):
    return {SESSION_HEADER: token}


def _lookup(server, token="tok-1", order_id="ORD-101"):
    return _call(server, "lookup_order", {"order_id": order_id}, _session(token))


def _log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_first_call_loads_the_ledger_from_the_evaluation(log_path):
    plane = ControlPlane({"tok-1": _klarna_environment("evaluation lamp")})
    server = _server(plane, log_path)

    result = _lookup(server)

    assert not result.isError
    assert json.loads(result.content[0].text)["item"] == "evaluation lamp"
    (get,) = plane.by_method("GET")
    assert get.path == "/agent/environment"
    assert get.authorization == "Bearer tok-1"


@pytest.mark.parametrize("domain", mcp_server.DOMAINS)
def test_each_domain_loads_its_ledger_from_the_evaluation(domain, log_path):
    environment = copy.deepcopy(BASE["targets"][domain]["environment"])
    environment["session_id"] = "EVAL0001"
    plane = ControlPlane({"tok-1": environment})
    server = _server(plane, log_path, domain)

    result = _call(server, f"get_{domain}_state_summary", {}, _session("tok-1"))

    assert json.loads(result.content[0].text)["session_id"] == "EVAL0001"
    assert plane.methods() == ["GET", "PUT", "POST"]


def test_every_call_pushes_the_ledger_and_then_reports_the_call(log_path):
    plane = ControlPlane({"tok-1": _klarna_environment("evaluation lamp")})
    server = _server(plane, log_path)

    result = _lookup(server)

    assert plane.methods() == ["GET", "PUT", "POST"]
    assert [request.path for request in plane.requests] == [
        "/agent/environment",
        "/agent/environment",
        "/agent/function-calls",
    ]
    assert {request.authorization for request in plane.requests} == {"Bearer tok-1"}
    (put,) = plane.by_method("PUT")
    assert put.body == server.executor.state.to_environment()
    (post,) = plane.by_method("POST")
    assert post.body == {
        "function": "lookup_order",
        "args": {"order_id": "ORD-101"},
        "result": result.content[0].text,
        "error": None,
        "blocked": False,
    }
    assert _log(log_path) == [
        {
            "sequence": 1,
            "function": "lookup_order",
            "load": "ok",
            "push": "ok",
            "report": "ok",
            "failure": None,
        }
    ]


def test_a_lookup_is_pushed_because_it_adds_an_audit_entry(log_path):
    loaded = _klarna_environment("evaluation lamp")
    plane = ControlPlane({"tok-1": loaded})
    server = _server(plane, log_path)

    _lookup(server)

    (put,) = plane.by_method("PUT")
    assert len(put.body["audit_logs"]) == len(loaded["audit_logs"]) + 1


def test_later_calls_of_the_same_session_keep_the_loaded_ledger(log_path):
    loaded = _klarna_environment("evaluation lamp")
    plane = ControlPlane({"tok-1": loaded})
    server = _server(plane, log_path)

    _lookup(server)
    _lookup(server)

    assert plane.methods() == ["GET", "PUT", "POST", "PUT", "POST"]
    second_put = plane.by_method("PUT")[1]
    assert len(second_put.body["audit_logs"]) == len(loaded["audit_logs"]) + 2
    assert [line["load"] for line in _log(log_path)] == ["ok", None]
    assert [line["sequence"] for line in _log(log_path)] == [1, 2]


def test_a_new_session_token_reloads_the_ledger(log_path):
    first = _klarna_environment("first evaluation lamp")
    second = _klarna_environment("second evaluation lamp")
    plane = ControlPlane({"tok-1": first, "tok-2": second})
    server = _server(plane, log_path)
    _lookup(server, "tok-1")

    result = _lookup(server, "tok-2")

    assert json.loads(result.content[0].text)["item"] == "second evaluation lamp"
    assert [get.authorization for get in plane.by_method("GET")] == [
        "Bearer tok-1",
        "Bearer tok-2",
    ]
    second_put = plane.by_method("PUT")[1]
    assert second_put.authorization == "Bearer tok-2"
    assert len(second_put.body["audit_logs"]) == len(second["audit_logs"]) + 1
    assert [line["load"] for line in _log(log_path)] == ["ok", "ok"]


@pytest.mark.parametrize(
    "refusal",
    [401, httpx.ConnectError("connection refused")],
    ids=["refused-token", "unreachable"],
)
def test_an_unreachable_control_plane_fails_the_call_and_is_logged(refusal, log_path):
    plane = ControlPlane({}, get=refusal)
    server = _server(plane, log_path)

    result = _lookup(server)

    assert result.isError
    assert "MiDojo environment unavailable" in result.content[0].text
    assert plane.methods() == ["GET"]
    (line,) = _log(log_path)
    assert line["load"] == "failed"
    assert line["push"] is None
    assert line["report"] is None
    assert line["failure"]


def test_a_call_without_a_token_fails_and_is_logged(log_path):
    plane = ControlPlane({})
    server = _server(plane, log_path)

    result = _call(server, "lookup_order", {"order_id": "ORD-101"}, {})

    assert result.isError
    assert "MiDojo session token missing" in result.content[0].text
    assert plane.requests == []
    assert _log(log_path) == [
        {
            "sequence": 1,
            "function": "lookup_order",
            "load": "no_token",
            "push": None,
            "report": None,
            "failure": "MiDojo session token missing",
        }
    ]


def test_a_call_fastmcp_rejects_is_still_reported(log_path):
    plane = ControlPlane({"tok-1": _klarna_environment("evaluation lamp")})
    server = _server(plane, log_path)

    result = _call(server, "lookup_order", {}, _session("tok-1"))

    assert result.isError
    assert plane.methods() == ["GET", "PUT", "POST"]
    (post,) = plane.by_method("POST")
    assert post.body["function"] == "lookup_order"
    assert post.body["args"] == {}
    assert post.body["result"] == ""
    assert "order_id" in post.body["error"]
    assert post.body["error"] in result.content[0].text
    (line,) = _log(log_path)
    assert line["push"] == "ok"
    assert line["report"] == "ok"
    assert line["failure"] is None


def test_a_failed_report_is_logged_and_the_result_still_returns(log_path):
    plane = ControlPlane({"tok-1": _klarna_environment("evaluation lamp")}, post=500)
    server = _server(plane, log_path)

    result = _lookup(server)

    assert not result.isError
    assert json.loads(result.content[0].text)["item"] == "evaluation lamp"
    (line,) = _log(log_path)
    assert line["push"] == "ok"
    assert line["report"] == "failed"
    assert "500" in line["failure"]


def test_a_rejected_push_is_logged_and_the_call_is_still_reported(log_path):
    plane = ControlPlane({"tok-1": _klarna_environment("evaluation lamp")}, put=422)
    server = _server(plane, log_path)

    result = _lookup(server)

    assert not result.isError
    assert plane.methods() == ["GET", "PUT", "POST"]
    (line,) = _log(log_path)
    assert line["push"] == "failed"
    assert line["report"] == "ok"
    assert "422" in line["failure"]


def test_the_header_wins_over_the_environment_variable(monkeypatch, log_path):
    monkeypatch.setenv(TOKEN_ENV, "tok-env")
    plane = ControlPlane(
        {
            "tok-1": _klarna_environment("header lamp"),
            "tok-env": _klarna_environment("variable lamp"),
        }
    )
    server = _server(plane, log_path)

    result = _lookup(server, "tok-1")

    assert json.loads(result.content[0].text)["item"] == "header lamp"
    assert {request.authorization for request in plane.requests} == {"Bearer tok-1"}


def test_the_environment_variable_binds_a_call_without_the_header(
    monkeypatch, log_path
):
    monkeypatch.setenv(TOKEN_ENV, "tok-env")
    plane = ControlPlane({"tok-env": _klarna_environment("variable lamp")})
    server = _server(plane, log_path)

    result = _call(server, "lookup_order", {"order_id": "ORD-101"}, {})

    assert json.loads(result.content[0].text)["item"] == "variable lamp"
    assert {request.authorization for request in plane.requests} == {"Bearer tok-env"}


def test_without_a_binding_no_request_is_made(monkeypatch, tmp_path):
    sent = []

    async def refuse(self, request, **kwargs):
        sent.append(request)
        raise AssertionError("no request expected without a binding")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    server = mcp_server.build_klarna_mcp()

    result = _lookup(server)

    assert server.midojo is None
    assert not result.isError
    assert json.loads(result.content[0].text)["item"] == "Desk lamp"
    assert sent == []
    assert list(tmp_path.iterdir()) == []


class FakeServer:
    def __init__(self):
        self.settings = SimpleNamespace(host=None, port=None)
        self.ran = False

    def run(self, transport):
        self.ran = True


@pytest.fixture
def built(monkeypatch):
    servers = []

    def builder(state=None):
        server = FakeServer()
        servers.append((state, server))
        return server

    monkeypatch.setitem(mcp_server.BUILDERS, "klarna", builder)
    return servers


def test_main_installs_a_binding_from_the_flags(tmp_path, built):
    log = tmp_path / "midojo_capture" / "calls.jsonl"

    mcp_server.main(
        [
            "--domain",
            "klarna",
            "--port",
            "1",
            "--midojo-url",
            "http://127.0.0.1:9/",
            "--midojo-log",
            str(log),
        ]
    )

    ((state, server),) = built
    assert state is None
    assert isinstance(server.midojo, MidojoBinding)
    assert server.midojo.control_url == "http://127.0.0.1:9"
    assert server.midojo.log_path == log
    assert server.ran


def test_main_without_the_flags_installs_no_binding(built):
    mcp_server.main(["--domain", "klarna", "--port", "1"])

    ((_, server),) = built
    assert getattr(server, "midojo", None) is None


def test_midojo_url_and_seed_file_are_exclusive(tmp_path, built, capsys):
    with pytest.raises(SystemExit) as exited:
        mcp_server.main(
            [
                "--domain",
                "klarna",
                "--midojo-url",
                "http://127.0.0.1:9",
                "--seed-file",
                str(tmp_path / "seed.json"),
            ]
        )

    assert exited.value.code == 2
    assert "MiDojo provisions the seed" in capsys.readouterr().err
    assert built == []


def test_midojo_log_needs_midojo_url(tmp_path, built, capsys):
    with pytest.raises(SystemExit) as exited:
        mcp_server.main(
            ["--domain", "klarna", "--midojo-log", str(tmp_path / "calls.jsonl")]
        )

    assert exited.value.code == 2
    assert "--midojo-url" in capsys.readouterr().err
    assert built == []
