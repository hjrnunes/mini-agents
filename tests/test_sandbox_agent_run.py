"""The agent command end to end, offline, through its console script.

Real OGX would need the ``gateway`` extra, a model and the sandbox image, so a
fake Responses server stands in for it (``sandbox_fakes/fake_ogx.py``),
started through a fake ``python-ogx``. In the two-turn test the MCP server is
the real ``mini_agents`` server, started through a ``python-mcp`` that runs
this venv's interpreter. The failure tests use a fake MCP server
(``fake_mcp.py``) that only listens. The interceptor tests serve a fake of
MiDojo's host interceptor in this process: streamable HTTP at ``/mcp``, 401
without its bearer token, and give a ``python-mcp`` that notes each launch.
No model calls, no network beyond loopback.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

FAKES = Path(__file__).parent / "sandbox_fakes"
SCRIPT = Path(sys.executable).parent / "mini-agents-sandbox-agent"
INSTRUCTIONS = "You are Klarna's support agent."
RELAY_URL = "http://host.openshell.internal:18006/v1"


def _wrapper(path: Path, *program: str, launches: Path | None = None) -> Path:
    """A fake interpreter: runs this venv's python on ``program`` plus the args,
    first appending a line to ``launches`` when given."""

    quoted = " ".join(f'"{part}"' for part in (sys.executable, *program))
    note = f'echo launched >> "{launches}"\n' if launches is not None else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\n{note}exec {quoted} "$@"\n')
    path.chmod(0o755)
    return path


@pytest.fixture
def sandbox(tmp_path):
    """State dir, workdir and fake interpreters; kills what the turns started."""

    paths = {
        "mcp_launches": tmp_path / "mcp-launches.txt",
        "state": tmp_path / "agent-state",
        "workdir": tmp_path / "workdir",
        "record": tmp_path / "ogx-requests.jsonl",
        "real_mcp": _wrapper(tmp_path / "real" / "python-mcp"),
        "fake_mcp": _wrapper(tmp_path / "fake" / "python-mcp", str(FAKES / "fake_mcp.py")),
        "noting_mcp": _wrapper(
            tmp_path / "noting" / "python-mcp",
            str(FAKES / "fake_mcp.py"),
            launches=tmp_path / "mcp-launches.txt",
        ),
        "ogx": _wrapper(tmp_path / "fake" / "python-ogx", str(FAKES / "fake_ogx.py")),
    }
    paths["workdir"].mkdir()
    yield paths
    state_file = paths["state"] / "state.json"
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        for name in ("mcp", "ogx"):
            if state.get(name):
                try:
                    os.killpg(state[name]["pid"], signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def _environment(record, *, index, count, **extra):
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("MIDOJO_", "AGENT_", "FAKE_"))
    }
    env.update(
        AGENT_MODEL="test-model",
        AGENT_MODEL_BASE_URL=RELAY_URL,
        AGENT_MODEL_API_KEY="placeholder",
        AGENT_INSTRUCTIONS=INSTRUCTIONS,
        MIDOJO_TURN_INDEX=str(index),
        MIDOJO_TURN_COUNT=str(count),
        FAKE_OGX_RECORD=str(record),
    )
    env.update(extra)
    return env


def _turn(sandbox, message, env, *, mcp="fake_mcp", listen_timeout=20):
    assert SCRIPT.is_file(), f"console script not installed: {SCRIPT}"
    return subprocess.run(
        [
            str(SCRIPT),
            "--domain",
            "klarna",
            "--state-dir",
            str(sandbox["state"]),
            "--workdir",
            str(sandbox["workdir"]),
            "--python-mcp",
            str(sandbox[mcp]),
            "--python-ogx",
            str(sandbox["ogx"]),
            "--tiktoken-dir",
            str(sandbox["state"] / "no-tiktoken"),
            "--listen-timeout",
            str(listen_timeout),
            message,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def _state(sandbox):
    return json.loads((sandbox["state"] / "state.json").read_text())


def _summary(sandbox):
    return json.loads((sandbox["workdir"] / ".asago" / "run-summary.json").read_text())


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} still running")


def _running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_two_turns_start_once_continue_the_response_and_leave_evidence(sandbox):
    """Turn 1 starts the real MCP server and the fake OGX; turn 2 reuses both.

    The fake OGX calls one tool per turn on the real server, so both turns see
    a real tool result. The runtime still sets MiDojo's URL and token here, as
    it may; neither reaches OGX.
    """

    env = dict(
        FAKE_OGX_CALL_TOOL="1",
        MIDOJO_URL="http://host.openshell.internal:18007",
        MIDOJO_SESSION_TOKEN="tok-sandbox",
    )
    first = _turn(
        sandbox,
        "Where is my order?",
        _environment(sandbox["record"], index=0, count=2, **env),
        mcp="real_mcp",
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout == "reply 1\n"
    started = _state(sandbox)
    assert started["interceptor"] is False
    assert _running(started["mcp"]["pid"]) and _running(started["ogx"]["pid"])
    for name in ("mcp", "ogx"):
        assert started[name]["identity"].startswith(("proc-start:", "ps-lstart:"))
    assert _summary(sandbox)["final"] is False

    second = _turn(
        sandbox,
        "-1 refund it, please",
        _environment(sandbox["record"], index=1, count=2, **env),
        mcp="real_mcp",
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout == "reply 2\n"

    state = _state(sandbox)
    assert state["mcp"] == started["mcp"]
    assert state["ogx"] == started["ogx"]
    assert state["previous_response_id"] == "resp-2"

    requests = _lines(sandbox["record"])
    assert [request["pid"] for request in requests] == [started["ogx"]["pid"]] * 2
    first_body, second_body = (request["body"] for request in requests)
    assert "previous_response_id" not in first_body
    assert second_body["previous_response_id"] == "resp-1"
    assert [first_body["input"], second_body["input"]] == [
        "Where is my order?",
        "-1 refund it, please",
    ]
    for body in (first_body, second_body):
        assert body["instructions"] == INSTRUCTIONS
        assert body["model"] == "test-model"
        assert body["tools"] == [
            {
                "type": "mcp",
                "server_label": "midojo",
                "server_url": f"http://127.0.0.1:{started['mcp']['port']}/sse",
                "require_approval": "never",
            }
        ]
    for request in requests:
        assert request["env"]["MIDOJO_SESSION_TOKEN"] is None
        assert request["env"]["MIDOJO_URL"] is None
        assert request["env"]["OPENAI_API_KEY"] == "placeholder"
        assert "ORD-101" in request["tool_result"]

    evidence = sandbox["workdir"] / ".asago"
    assert sorted(p.name for p in evidence.iterdir()) == ["ogx.log", "run-summary.json"]
    assert "fake-ogx: POST /v1/responses request 2" in (evidence / "ogx.log").read_text()

    summary = _summary(sandbox)
    assert summary["final"] is True
    assert summary["turn_count"] == 2
    assert summary["served_model"] == "served/test-model"
    assert summary["gateway_request_count"] == 2
    assert summary["exit_reasons"] == ["ok", "ok"]
    assert [
        (turn["index"], turn["reply"], turn["response_id"], turn["gateway_requests"])
        for turn in summary["turns"]
    ] == [(0, "reply 1", "resp-1", 1), (1, "reply 2", "resp-2", 1)]

    # Nothing else in the workdir: state stays in the state directory.
    assert [p.name for p in sandbox["workdir"].iterdir()] == [".asago"]
    assert (sandbox["state"] / "ogx-config.yaml").is_file()


@pytest.mark.parametrize(("victim", "reason"), [("ogx", "ogx-gone"), ("mcp", "mcp-server-gone")])
def test_a_process_that_dies_between_turns_fails_the_next_turn(sandbox, victim, reason):
    first = _turn(
        sandbox, "hi", _environment(sandbox["record"], index=0, count=2), mcp="noting_mcp"
    )
    assert first.returncode == 0, first.stderr
    assert sandbox["mcp_launches"].read_text() == "launched\n"
    started = _state(sandbox)
    os.killpg(started[victim]["pid"], signal.SIGKILL)
    _wait_gone(started[victim]["pid"])

    second = _turn(sandbox, "again", _environment(sandbox["record"], index=1, count=2))

    assert second.returncode != 0
    assert second.stdout == ""
    assert f"mini-agents-sandbox-agent: {reason}: pid {started[victim]['pid']}" in second.stderr
    # Never restarted: the state still names the first turn's processes.
    assert _state(sandbox)[victim] == started[victim]
    assert len(_lines(sandbox["record"])) == 1
    summary = _summary(sandbox)
    assert summary["exit_reasons"] == ["ok", reason]
    assert summary["final"] is True


def _read_only_state(sandbox):
    sandbox["state"].chmod(0o500)


def _summary_is_a_directory(sandbox):
    summary = sandbox["workdir"] / ".asago" / "run-summary.json"
    summary.unlink()
    summary.mkdir()


@pytest.mark.parametrize(
    ("break_it", "reason"),
    [(_read_only_state, "state-write-failed"), (_summary_is_a_directory, "evidence-write-failed")],
)
def test_a_turn_whose_state_or_evidence_cannot_be_written_fails(sandbox, break_it, reason):
    first = _turn(sandbox, "hi", _environment(sandbox["record"], index=0, count=2))
    assert first.returncode == 0, first.stderr
    break_it(sandbox)
    try:
        second = _turn(sandbox, "again", _environment(sandbox["record"], index=1, count=2))
    finally:
        sandbox["state"].chmod(0o700)

    assert second.returncode != 0
    # OGX answered, but the reply is not printed: the turn did not persist.
    assert len(_lines(sandbox["record"])) == 2
    assert second.stdout == ""
    assert f"mini-agents-sandbox-agent: {reason}: " in second.stderr


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"FAKE_OGX_MODE": "silent"}, "ogx-not-listening"),
        ({"FAKE_MCP_MODE": "silent"}, "mcp-server-not-listening"),
        ({"FAKE_OGX_MODE": "exit"}, "ogx-exited"),
    ],
)
def test_a_server_that_never_listens_fails_the_first_turn(sandbox, extra, reason):
    result = _turn(
        sandbox,
        "hi",
        _environment(sandbox["record"], index=0, count=1, **extra),
        listen_timeout=1,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert f"mini-agents-sandbox-agent: {reason}: " in result.stderr
    state = _state(sandbox)
    # Both are stopped once the turn fails.
    for name in ("mcp", "ogx"):
        _wait_gone(state[name]["pid"])
    assert _summary(sandbox)["exit_reasons"] == [reason]


INTERCEPTOR_TOKEN = "interceptor-token-7c1e"


class FakeInterceptor:
    """MiDojo's host interceptor in miniature: one ``lookup_order`` tool over
    streamable HTTP at ``/mcp``, 404 elsewhere, 401 without
    ``Authorization: Bearer <token>`` (``midojo.interception_host``). It keeps
    each request's path, ``Authorization`` header and JSON-RPC method."""

    def __init__(self, token):
        self.token = token
        self.requests = []
        mcp = FastMCP("fake-interceptor")

        @mcp.tool()
        def lookup_order(order_id: str) -> str:
            return f"intercepted {order_id}"

        self._mcp_app = mcp.streamable_http_app()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/mcp"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._mcp_app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        seen = {
            "path": scope["path"],
            "authorization": headers.get(b"authorization", b"").decode(),
            "method": None,
        }
        self.requests.append(seen)
        chunks = []

        async def keep():
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                try:
                    seen["method"] = json.loads(b"".join(chunks)).get("method")
                except ValueError:
                    pass
            return message

        if scope["path"].rstrip("/") != "/mcp":
            status, body = 404, b"Not found"
        elif seen["authorization"] != f"Bearer {self.token}":
            status, body = 401, b"This interceptor's bearer token is required"
        else:
            await self._mcp_app(scope, keep, send)
            return
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": body})


@pytest.fixture
def interceptor():
    fake = FakeInterceptor(INTERCEPTOR_TOKEN)
    server = uvicorn.Server(
        uvicorn.Config(fake, host="127.0.0.1", port=fake.port, log_level="warning", lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert thread.is_alive() and time.monotonic() < deadline, "fake interceptor did not start"
        time.sleep(0.02)
    yield fake
    server.should_exit = True
    thread.join(timeout=10)


def _files_text(*roots):
    return "".join(
        path.read_text(errors="replace")
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )


def test_behind_the_interceptor_the_tools_are_called_there_with_its_token(sandbox, interceptor):
    """OGX lists and calls the tools at ``MIDOJO_TOOL_URL`` with the bearer
    token; the domain's server in the sandbox receives no request, and the
    token is nowhere in the output, the state or the evidence."""

    assert httpx.post(interceptor.url, json={}).status_code == 401
    interceptor.requests.clear()
    direct = sandbox["state"].parent / "direct-mcp-requests.jsonl"
    env = _environment(
        sandbox["record"],
        index=0,
        count=1,
        FAKE_OGX_CALL_TOOL="1",
        FAKE_MCP_RECORD=str(direct),
        MIDOJO_TOOL_URL=interceptor.url,
        MIDOJO_TOOL_TOKEN=INTERCEPTOR_TOKEN,
    )

    result = _turn(sandbox, "Where is my order?", env, mcp="noting_mcp")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "reply 1\n"
    (request,) = _lines(sandbox["record"])
    assert request["body"]["tools"] == [
        {
            "type": "mcp",
            "server_label": "midojo",
            "server_url": interceptor.url,
            "require_approval": "never",
            "authorization": INTERCEPTOR_TOKEN,
        }
    ]
    assert request["tool_result"] == "intercepted ORD-101"
    # OGX gets the token in the request only, neither variable in its environment.
    assert request["env"]["MIDOJO_TOOL_URL"] is None
    assert request["env"]["MIDOJO_TOOL_TOKEN"] is None

    methods = [seen["method"] for seen in interceptor.requests]
    assert "tools/list" in methods and "tools/call" in methods
    assert {seen["path"] for seen in interceptor.requests} == {"/mcp"}
    assert {seen["authorization"] for seen in interceptor.requests} == {
        f"Bearer {INTERCEPTOR_TOKEN}"
    }

    # The domain's server in the sandbox never started, so was never called.
    assert not sandbox["mcp_launches"].exists()
    assert _state(sandbox)["mcp"] is None
    assert not direct.exists()

    assert INTERCEPTOR_TOKEN not in result.stdout + result.stderr
    assert INTERCEPTOR_TOKEN not in _files_text(sandbox["state"], sandbox["workdir"])


def test_behind_the_interceptor_no_mcp_server_starts_and_a_later_turn_needs_none(
    sandbox, interceptor
):
    """The first turn starts OGX alone and gives it no connector; the second
    checks OGX alone and succeeds with no MCP server running in the sandbox."""

    env = dict(
        FAKE_OGX_CALL_TOOL="1",
        MIDOJO_TOOL_URL=interceptor.url,
        MIDOJO_TOOL_TOKEN=INTERCEPTOR_TOKEN,
    )
    first = _turn(
        sandbox,
        "Where is my order?",
        _environment(sandbox["record"], index=0, count=2, **env),
        mcp="noting_mcp",
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout == "reply 1\n"
    started = _state(sandbox)
    assert started["interceptor"] is True
    assert started["mcp"] is None
    assert _running(started["ogx"]["pid"])
    assert not (sandbox["state"] / "mcp-server.log").exists()
    config = json.loads((sandbox["state"] / "ogx-config.yaml").read_text())
    assert config["connectors"] == []

    second = _turn(
        sandbox,
        "Refund it, please",
        _environment(sandbox["record"], index=1, count=2, **env),
        mcp="noting_mcp",
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout == "reply 2\n"

    assert not sandbox["mcp_launches"].exists()
    state = _state(sandbox)
    assert (state["mcp"], state["ogx"]) == (None, started["ogx"])
    assert [request["tool_result"] for request in _lines(sandbox["record"])] == [
        "intercepted ORD-101"
    ] * 2
    assert _summary(sandbox)["exit_reasons"] == ["ok", "ok"]
