"""The agent command end to end, offline, through its console script.

Real OGX would need the ``gateway`` extra, a model and the sandbox image, so a
fake Responses server stands in for it (``sandbox_fakes/fake_ogx.py``),
started through a fake ``python-ogx``. In the two-turn test the MCP server is
the real ``mini_agents`` server in MiDojo mode, started through a
``python-mcp`` that runs this venv's interpreter, against a fake control plane
on loopback. The failure tests use a fake MCP server (``fake_mcp.py``) that
only listens. No model calls, no network beyond loopback.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mini_agents.kernel import mcp_server

FAKES = Path(__file__).parent / "sandbox_fakes"
SCRIPT = Path(sys.executable).parent / "mini-agents-sandbox-agent"
TOKEN = "tok-sandbox"
INSTRUCTIONS = "You are Klarna's support agent."
RELAY_URL = "http://host.openshell.internal:18006/v1"
BASE = json.loads(
    (Path(mcp_server.__file__).parent.parent / "midojo-base.json").read_text()
)


class ControlPlane:
    """MiDojo's agent API on loopback: serves the klarna ledger, records calls."""

    def __init__(self):
        self.requests = []
        environment = copy.deepcopy(BASE["targets"]["klarna"]["environment"])
        plane = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, status, body):
                length = int(self.headers.get("Content-Length") or 0)
                content = self.rfile.read(length) if length else b""
                plane.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(content) if content else None,
                    }
                )
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802
                self._answer(200, environment)

            def do_PUT(self):  # noqa: N802
                self._answer(200, {})

            def do_POST(self):  # noqa: N802
                self._answer(201, {})

            def log_message(self, format, *args):  # noqa: A002
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _wrapper(path: Path, *program: str) -> Path:
    """A fake interpreter: runs this venv's python on ``program`` plus the args."""

    quoted = " ".join(f'"{part}"' for part in (sys.executable, *program))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nexec {quoted} "$@"\n')
    path.chmod(0o755)
    return path


@pytest.fixture
def sandbox(tmp_path):
    """State dir, workdir and fake interpreters; kills what the turns started."""

    paths = {
        "state": tmp_path / "agent-state",
        "workdir": tmp_path / "workdir",
        "record": tmp_path / "ogx-requests.jsonl",
        "real_mcp": _wrapper(tmp_path / "real" / "python-mcp"),
        "fake_mcp": _wrapper(tmp_path / "fake" / "python-mcp", str(FAKES / "fake_mcp.py")),
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


def _environment(control_url, record, *, index, count, **extra):
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("MIDOJO_", "AGENT_", "FAKE_"))
    }
    env.update(
        MIDOJO_URL=control_url,
        MIDOJO_SESSION_TOKEN=TOKEN,
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

    The fake OGX calls one tool per turn on the real server, so the server's
    MiDojo log and the control plane see real calls.
    """

    with ControlPlane() as plane:
        env = dict(FAKE_OGX_CALL_TOOL="1")
        first = _turn(
            sandbox,
            "Where is my order?",
            _environment(plane.url, sandbox["record"], index=0, count=2, **env),
            mcp="real_mcp",
        )
        assert first.returncode == 0, first.stderr
        assert first.stdout == "reply 1\n"
        started = _state(sandbox)
        assert _running(started["mcp"]["pid"]) and _running(started["ogx"]["pid"])
        for name in ("mcp", "ogx"):
            assert started[name]["identity"].startswith(("proc-start:", "ps-lstart:"))
        assert _summary(sandbox)["final"] is False

        second = _turn(
            sandbox,
            "-1 refund it, please",
            _environment(plane.url, sandbox["record"], index=1, count=2, **env),
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

    # The server took the token from its environment: no header reached it.
    calls = [r for r in plane.requests if r["path"] == "/agent/function-calls"]
    assert len(calls) == 2
    assert {r["authorization"] for r in plane.requests} == {f"Bearer {TOKEN}"}

    evidence = sandbox["workdir"] / ".asago"
    assert sorted(p.name for p in evidence.iterdir()) == [
        "mcp-calls.jsonl",
        "ogx.log",
        "run-summary.json",
    ]
    log = _lines(evidence / "mcp-calls.jsonl")
    assert [line["function"] for line in log] == ["lookup_order"] * 2
    assert [line["report"] for line in log] == ["ok", "ok"]
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
    with ControlPlane() as plane:
        first = _turn(
            sandbox, "hi", _environment(plane.url, sandbox["record"], index=0, count=2)
        )
        assert first.returncode == 0, first.stderr
        started = _state(sandbox)
        os.killpg(started[victim]["pid"], signal.SIGKILL)
        _wait_gone(started[victim]["pid"])

        second = _turn(
            sandbox, "again", _environment(plane.url, sandbox["record"], index=1, count=2)
        )

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
    with ControlPlane() as plane:
        first = _turn(
            sandbox, "hi", _environment(plane.url, sandbox["record"], index=0, count=2)
        )
        assert first.returncode == 0, first.stderr
        break_it(sandbox)
        try:
            second = _turn(
                sandbox, "again", _environment(plane.url, sandbox["record"], index=1, count=2)
            )
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
    with ControlPlane() as plane:
        result = _turn(
            sandbox,
            "hi",
            _environment(plane.url, sandbox["record"], index=0, count=1, **extra),
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
