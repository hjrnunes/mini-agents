"""Each part of the agent command on its own: config, state, request,
processes, environment, summary, and the turn's inputs."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import httpx
import pytest

from mini_agents.sandbox_agent import cli, failures
from mini_agents.sandbox_agent.environment import (
    SCRUBBED,
    ogx_environment,
    server_environment,
)
from mini_agents.sandbox_agent.failures import AgentFailure
from mini_agents.sandbox_agent.ogx_config import render_ogx_config, write_ogx_config
from mini_agents.sandbox_agent.processes import (
    free_ports,
    mcp_command,
    ogx_command,
    pid_alive,
    process_identity,
    start_detached,
    stop,
    wait_listening,
)
from mini_agents.sandbox_agent.responses import create_response, reply_text, request_body
from mini_agents.sandbox_agent.state import (
    AgentState,
    Process,
    Turn,
    load_state,
    save_state,
)
from mini_agents.sandbox_agent.summary import (
    copy_ogx_log,
    record_rejection,
    run_summary,
    write_run_summary,
)

ROOT = Path(__file__).resolve().parent.parent
RELAY = "http://host.openshell.internal:18006/v1"


def _config(tmp_path, **overrides):
    values = dict(
        domain="klarna",
        mcp_port=20001,
        ogx_port=20002,
        model="test-model",
        model_base_url=RELAY,
        state_dir=tmp_path / "agent-state",
    )
    values.update(overrides)
    return render_ogx_config(**values)


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _keyed(value, key):
    if isinstance(value, dict):
        for item_key, item in value.items():
            if item_key == key:
                yield item
            yield from _keyed(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _keyed(item, key)


# OGX config


def test_the_config_holds_one_loopback_connector_for_the_domain(tmp_path):
    config = _config(tmp_path, domain="airbnb")
    assert config["connectors"] == [
        {"connector_id": "airbnb-safe", "url": "http://127.0.0.1:20001/sse"}
    ]


def test_the_server_binds_loopback_on_its_port(tmp_path):
    assert _config(tmp_path)["server"] == {"port": 20002, "host": "127.0.0.1"}


def test_inference_points_at_the_relay_and_keeps_the_key_out_of_the_file(tmp_path):
    (inference,) = _config(tmp_path)["providers"]["inference"]
    assert inference["config"] == {
        "api_token": "${env.OPENAI_API_KEY}",
        "base_url": RELAY,
    }
    (model,) = _config(tmp_path)["registered_resources"]["models"]
    assert model["model_id"] == "test-model"


def test_the_config_has_no_file_processors(tmp_path):
    config = _config(tmp_path)
    assert "file_processors" not in config["apis"]
    assert "file_processors" not in config["providers"]
    assert "file_processors" not in json.dumps(config)


def test_every_store_is_under_the_state_dir(tmp_path):
    config = _config(tmp_path)
    state_dir = tmp_path / "agent-state"
    paths = [*_keyed(config, "db_path"), *_keyed(config, "storage_dir")]
    assert len(paths) == 4
    assert all(Path(path).is_relative_to(state_dir) for path in paths)
    assert not [s for s in _strings(config) if "~" in s or "SQLITE_STORE_DIR" in s]


def test_the_config_keeps_the_ogx_config_sections_except_file_processors(tmp_path):
    source = (ROOT / "ogx-config.yaml").read_text()
    config = _config(tmp_path)
    for api in config["apis"]:
        assert f"  - {api}\n" in source
    for provider in config["providers"]:
        assert f"\n  {provider}:\n" in source


def test_a_written_config_is_json_that_reads_back(tmp_path):
    config = _config(tmp_path)
    path = write_ogx_config(config, tmp_path / "agent-state" / "ogx-config.yaml")
    assert json.loads(path.read_text()) == config
    assert [p.name for p in path.parent.iterdir()] == ["ogx-config.yaml"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"domain": "nope"},
        {"mcp_port": 0},
        {"ogx_port": 20001},
        {"model_base_url": "host.openshell.internal:18006"},
        {"model": ""},
    ],
)
def test_the_config_refuses_bad_values(tmp_path, overrides):
    with pytest.raises(ValueError):
        _config(tmp_path, **overrides)


# State


def _state():
    return AgentState(
        domain="klarna",
        turn_count=2,
        mcp=Process(
            pid=11, port=20001, command=["python-mcp"], log="/s/mcp-server.log", identity="id-11"
        ),
        ogx=Process(pid=12, port=20002, command=["python-ogx"], log="/s/ogx.log", identity="id-12"),
        previous_response_id="resp-1",
        turns=[Turn(0, "reply 1", "resp-1", "served/m", 1, "ok")],
    )


def test_state_round_trips(tmp_path):
    path = save_state(_state(), tmp_path)
    assert path == tmp_path / "state.json"
    assert load_state(tmp_path) == _state()
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_no_state_file_loads_as_none(tmp_path):
    assert load_state(tmp_path) is None


def test_a_state_of_another_version_is_refused(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError):
        load_state(tmp_path)


# Request body


def test_the_first_turn_sends_no_previous_response_id():
    body = request_body(
        model="m",
        message="hello",
        instructions="be careful",
        mcp_url="http://127.0.0.1:20001/sse",
        previous_response_id=None,
    )
    assert body == {
        "model": "m",
        "input": "hello",
        "instructions": "be careful",
        "tools": [
            {
                "type": "mcp",
                "server_label": "midojo",
                "server_url": "http://127.0.0.1:20001/sse",
                "require_approval": "never",
            }
        ],
        "stream": False,
    }


def test_allowed_tools_restrict_the_mcp_tool_as_the_garak_path_does():
    body = request_body(
        model="m",
        message="hello",
        instructions="",
        mcp_url="http://127.0.0.1:20001/sse",
        previous_response_id=None,
        allowed_tools=("lookup_order", "get_account_details"),
    )
    assert body["tools"][0]["allowed_tools"] == ["lookup_order", "get_account_details"]


def test_a_later_turn_continues_the_previous_response():
    body = request_body(
        model="m",
        message="and now?",
        instructions="",
        mcp_url="http://127.0.0.1:20001/sse",
        previous_response_id="resp-1",
    )
    assert body["previous_response_id"] == "resp-1"
    assert "headers" not in body["tools"][0]


def test_the_reply_joins_the_message_texts_only():
    response = {
        "output": [
            {"type": "mcp_call", "output": "tool text"},
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Hello, "},
                    {"type": "refusal", "refusal": "no"},
                    {"type": "output_text", "text": "there."},
                ],
            },
        ]
    }
    assert reply_text(response) == "Hello, there."
    assert reply_text({}) == ""


def _gateway(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("handler", "reason"),
    [
        (lambda request: httpx.Response(500, text="boom"), failures.GATEWAY_REQUEST_FAILED),
        (lambda request: httpx.Response(200, text="not json"), failures.GATEWAY_RESPONSE_INVALID),
        (lambda request: httpx.Response(200, json={"output": []}), failures.GATEWAY_RESPONSE_INVALID),
        (
            lambda request: httpx.Response(
                200, json={"id": "resp-1", "status": "failed", "error": {"message": "x"}}
            ),
            failures.GATEWAY_RESPONSE_FAILED,
        ),
    ],
)
def test_a_bad_gateway_answer_fails_with_a_named_reason(handler, reason):
    with pytest.raises(AgentFailure) as raised:
        create_response("http://127.0.0.1:1", {}, timeout=1, client=_gateway(handler))
    assert raised.value.reason == reason
    assert raised.value.gateway_requests == 1


def test_the_request_goes_to_the_responses_path():
    seen = []

    def handler(request):
        seen.append((request.url, json.loads(request.content)))
        return httpx.Response(200, json={"id": "resp-1", "model": "m", "output": []})

    response = create_response(
        "http://127.0.0.1:9/", {"model": "m"}, timeout=1, client=_gateway(handler)
    )
    assert response["id"] == "resp-1"
    assert [(str(url), body) for url, body in seen] == [
        ("http://127.0.0.1:9/v1/responses", {"model": "m"})
    ]


# Processes


def test_the_commands_use_the_configured_interpreters():
    assert mcp_command(Path("/venv/bin/python-mcp"), domain="occiai", port=20001) == [
        "/venv/bin/python-mcp",
        "-m",
        "mini_agents",
        "--domain",
        "occiai",
        "--host",
        "127.0.0.1",
        "--port",
        "20001",
    ]
    assert ogx_command(Path("/venv/bin/python-ogx"), Path("/s/ogx-config.yaml")) == [
        "/venv/bin/python-ogx",
        "-m",
        "ogx.cli.ogx",
        "run",
        "/s/ogx-config.yaml",
        "--insecure",
    ]


def test_the_defaults_are_the_contract_paths():
    assert cli.PYTHON_MCP == Path("/opt/app-root/.venv/bin/python-mcp")
    assert cli.PYTHON_OGX == Path("/opt/app-root/.venv/bin/python-ogx")
    assert cli.STATE_DIR == Path("/tmp/agent-state")
    assert cli.WORKDIR == Path("/sandbox/workdir")
    assert cli.TIKTOKEN_DIR == Path("/opt/app-root/tiktoken")


def test_free_ports_are_distinct():
    first, second = free_ports(2)
    assert first != second


def _sleeper(tmp_path, code="import time; print('up', flush=True); time.sleep(60)"):
    return start_detached(
        [sys.executable, "-c", code],
        env=os.environ,
        log=tmp_path / "logs" / "sleeper.log",
        cwd=tmp_path,
        name="sleeper",
    )


def test_a_detached_process_runs_in_its_own_session_with_stdio_to_its_log(tmp_path):
    process = _sleeper(tmp_path)
    try:
        assert os.getpgid(process.pid) == process.pid != os.getpgid(0)
        assert pid_alive(process.pid)
        deadline = time.monotonic() + 10
        log = tmp_path / "logs" / "sleeper.log"
        while "up" not in log.read_text() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert log.read_text() == "up\n"
    finally:
        stop(process)
    assert process.returncode is not None
    assert not pid_alive(process.pid)


def test_a_command_that_cannot_start_fails_with_a_named_reason(tmp_path):
    with pytest.raises(AgentFailure) as raised:
        start_detached(
            [str(tmp_path / "missing-python")],
            env={},
            log=tmp_path / "x.log",
            cwd=tmp_path,
            name="ogx",
        )
    assert raised.value.reason == failures.START_FAILED


def test_waiting_returns_once_the_port_listens(tmp_path):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        process = _sleeper(tmp_path)
        try:
            wait_listening(
                process,
                port=server.getsockname()[1],
                name="ogx",
                deadline=time.monotonic() + 5,
                log=tmp_path / "x.log",
            )
        finally:
            stop(process)


def test_a_process_that_never_listens_fails_at_the_deadline(tmp_path):
    (port,) = free_ports(1)
    process = _sleeper(tmp_path)
    try:
        started = time.monotonic()
        with pytest.raises(AgentFailure) as raised:
            wait_listening(
                process, port=port, name="ogx", deadline=started + 0.5, log=tmp_path / "x.log"
            )
        assert time.monotonic() - started < 3
    finally:
        stop(process)
    assert raised.value.reason == "ogx-not-listening"


def test_a_process_that_exits_before_listening_fails_at_once(tmp_path):
    (port,) = free_ports(1)
    process = _sleeper(tmp_path, code="raise SystemExit(7)")
    with pytest.raises(AgentFailure) as raised:
        wait_listening(
            process,
            port=port,
            name="mcp-server",
            deadline=time.monotonic() + 30,
            log=tmp_path / "x.log",
        )
    assert raised.value.reason == "mcp-server-exited"
    assert "exited with 7" in raised.value.detail


def _proc(tmp_path, pid, state, start=987654):
    """A fake /proc entry: fields 4-21 are their own numbers, field 22 ``start``."""
    entry = tmp_path / "proc" / str(pid)
    entry.mkdir(parents=True)
    middle = " ".join(str(field) for field in range(4, 22))
    (entry / "stat").write_text(f"{pid} (python-ogx (x)) {state} {middle} {start} 23 24\n")
    return tmp_path / "proc"


def test_liveness_reads_proc_when_there_is_one(tmp_path):
    root = _proc(tmp_path, 41, "S")
    _proc(tmp_path, 42, "Z")
    assert pid_alive(41, proc_root=root)
    assert not pid_alive(42, proc_root=root)
    assert not pid_alive(43, proc_root=root)


def test_liveness_falls_back_to_kill_without_proc(tmp_path):
    root = tmp_path / "no-proc"
    assert pid_alive(os.getpid(), proc_root=root)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    assert not pid_alive(process.pid, proc_root=root)


# Environment


ENVIRON = {
    "MIDOJO_URL": "http://host.openshell.internal:18007",
    "MIDOJO_SESSION_TOKEN": "tok",
    "MIDOJO_TOOL_URL": "http://host.openshell.internal:18008/mcp",
    "MIDOJO_TOOL_TOKEN": "tool-tok",
    "MIDOJO_TURN_INDEX": "0",
    "AGENT_MODEL": "m",
    "HOME": "/sandbox",
    "PATH": "/usr/bin",
}


def test_ogx_gets_neither_midojo_url_nor_token():
    env = ogx_environment(
        ENVIRON,
        venv_bin=Path("/opt/app-root/.venv/bin"),
        tiktoken_dir=Path("/opt/app-root/tiktoken"),
        api_key="placeholder",
    )
    assert SCRUBBED == (
        "MIDOJO_SESSION_TOKEN",
        "MIDOJO_URL",
        "MIDOJO_TOOL_TOKEN",
        "MIDOJO_TOOL_URL",
    )
    assert not set(SCRUBBED) & set(env)
    assert env["OPENAI_API_KEY"] == "placeholder"
    assert env["PATH"] == "/opt/app-root/.venv/bin:/usr/local/bin:/usr/bin:/bin"
    assert env["TIKTOKEN_CACHE_DIR"] == "/opt/app-root/tiktoken"
    assert env["HOME"] == "/sandbox"
    assert "MIDOJO_SESSION_TOKEN" in ENVIRON


def test_the_server_gets_neither_midojo_url_nor_token():
    env = server_environment(
        ENVIRON,
        venv_bin=Path("/opt/app-root/.venv/bin"),
        tiktoken_dir=Path("/opt/app-root/tiktoken"),
    )
    assert not set(SCRUBBED) & set(env)
    assert env["HOME"] == "/sandbox"
    assert env["PATH"] == "/opt/app-root/.venv/bin:/usr/local/bin:/usr/bin:/bin"
    assert env["TIKTOKEN_CACHE_DIR"] == "/opt/app-root/tiktoken"
    assert "OPENAI_API_KEY" not in env


# Run summary


def test_the_summary_lists_each_turn_and_the_totals():
    state = _state()
    state.turns.append(Turn(1, None, None, None, 0, "ogx-gone", "pid 12 is gone"))
    summary = run_summary(state)
    assert summary == {
        "schema": "mini-agents-run-summary-v1",
        "domain": "klarna",
        "turn_count": 2,
        "turns_run": 2,
        "final": True,
        "served_model": "served/m",
        "gateway_request_count": 1,
        "exit_reasons": ["ok", "ogx-gone"],
        "turns": [
            {
                "index": 0,
                "reply": "reply 1",
                "response_id": "resp-1",
                "served_model": "served/m",
                "gateway_requests": 1,
                "exit_reason": "ok",
                "failure": None,
            },
            {
                "index": 1,
                "reply": None,
                "response_id": None,
                "served_model": None,
                "gateway_requests": 0,
                "exit_reason": "ogx-gone",
                "failure": "pid 12 is gone",
            },
        ],
        "rejected": [],
    }


def test_the_summary_is_not_final_before_the_last_turn():
    assert run_summary(_state())["final"] is False


def test_the_summary_and_ogx_log_land_in_the_evidence_dir(tmp_path):
    workdir = tmp_path / "workdir"
    path = write_run_summary(_state(), workdir)
    assert path == workdir / ".asago" / "run-summary.json"
    assert json.loads(path.read_text()) == run_summary(_state())
    assert copy_ogx_log(tmp_path / "missing.log", workdir) is None
    source = tmp_path / "ogx.log"
    source.write_text("started\n")
    assert copy_ogx_log(source, workdir).read_text() == "started\n"


# Inputs and the turn


FULL = {
    "AGENT_MODEL": "m",
    "AGENT_MODEL_BASE_URL": RELAY,
    "AGENT_MODEL_API_KEY": "placeholder",
    "AGENT_INSTRUCTIONS": "",
}


@pytest.mark.parametrize(
    ("environ", "turn"),
    [({}, (0, 1)), ({"MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "3"}, (1, 3))],
)
def test_the_turn_defaults_to_the_first_of_one(environ, turn):
    assert cli.read_turn(environ) == turn


@pytest.mark.parametrize(
    "environ",
    [
        {"MIDOJO_TURN_INDEX": "x"},
        {"MIDOJO_TURN_INDEX": "2", "MIDOJO_TURN_COUNT": "2"},
        {"MIDOJO_TURN_INDEX": "-1"},
        {"MIDOJO_TURN_COUNT": "0"},
    ],
)
def test_a_bad_turn_is_invalid_input(environ):
    with pytest.raises(AgentFailure) as raised:
        cli.read_turn(environ)
    assert raised.value.reason == failures.INPUT_INVALID


@pytest.mark.parametrize("name", [*cli.REQUIRED, "AGENT_INSTRUCTIONS"])
def test_each_input_variable_is_required(name):
    environ = {key: value for key, value in FULL.items() if key != name}
    with pytest.raises(AgentFailure) as raised:
        cli.read_inputs(environ)
    assert raised.value.reason == failures.INPUT_MISSING
    assert name in raised.value.detail


def test_allowed_tools_are_optional_and_read_as_a_comma_separated_list():
    assert cli.read_inputs(FULL).allowed_tools is None
    inputs = cli.read_inputs({**FULL, "AGENT_ALLOWED_TOOLS": "lookup_order, get_account_details"})
    assert inputs.allowed_tools == ("lookup_order", "get_account_details")


@pytest.mark.parametrize("value", ["", "lookup_order,", ",lookup_order"])
def test_a_malformed_tool_list_is_invalid_input(value):
    with pytest.raises(AgentFailure) as raised:
        cli.read_inputs({**FULL, "AGENT_ALLOWED_TOOLS": value})
    assert raised.value.reason == failures.INPUT_INVALID


def _main(
    tmp_path,
    environ,
    *args,
    message="hello",
    alive=lambda pid: True,
    identity=lambda pid: f"id-{pid}",
):
    """Run one turn in-process; by default the state's PIDs 11 and 12 pass."""
    return cli.main(
        [
            "--domain",
            "klarna",
            "--state-dir",
            str(tmp_path / "agent-state"),
            "--workdir",
            str(tmp_path / "workdir"),
            "--python-mcp",
            str(tmp_path / "no-python-mcp"),
            "--python-ogx",
            str(tmp_path / "no-python-ogx"),
            *args,
            message,
        ],
        environ,
        alive=alive,
        identity=identity,
    )


def test_a_missing_input_variable_fails_the_turn_before_anything_starts(tmp_path, capsys):
    environ = {key: value for key, value in FULL.items() if key != "AGENT_MODEL"}
    assert _main(tmp_path, environ) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "mini-agents-sandbox-agent: input-missing: not set: AGENT_MODEL\n"
    state = load_state(tmp_path / "agent-state")
    assert state.mcp is None and state.ogx is None
    summary = json.loads((tmp_path / "workdir" / ".asago" / "run-summary.json").read_text())
    assert summary["exit_reasons"] == ["input-missing"]
    assert summary["gateway_request_count"] == 0


def _summary(tmp_path):
    return json.loads((tmp_path / "workdir" / ".asago" / "run-summary.json").read_text())


def test_a_later_turn_without_state_fails_and_is_recorded_as_rejected(tmp_path, capsys):
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}) == 1
    assert "mini-agents-sandbox-agent: state-missing: " in capsys.readouterr().err
    assert not (tmp_path / "agent-state").exists()
    summary = _summary(tmp_path)
    assert summary["turns"] == [] and summary["exit_reasons"] == []
    assert summary["turn_count"] == 2 and summary["final"] is False
    (rejected,) = summary["rejected"]
    assert (rejected["index"], rejected["exit_reason"]) == (1, "state-missing")


def test_a_turn_out_of_order_fails_and_is_recorded_as_rejected(tmp_path, capsys):
    save_state(_state(), tmp_path / "agent-state")
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "0", "MIDOJO_TURN_COUNT": "2"}) == 1
    assert "mini-agents-sandbox-agent: turn-out-of-order: " in capsys.readouterr().err
    state = load_state(tmp_path / "agent-state")
    assert state.turns == _state().turns
    assert [(t.index, t.exit_reason) for t in state.rejected] == [(0, "turn-out-of-order")]
    summary = _summary(tmp_path)
    assert summary["exit_reasons"] == ["ok"]
    assert [r["exit_reason"] for r in summary["rejected"]] == ["turn-out-of-order"]


def test_a_rejected_run_does_not_shift_the_next_turn(tmp_path, monkeypatch, capsys):
    save_state(_state(), tmp_path / "agent-state")
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "5", "MIDOJO_TURN_COUNT": "2"}) == 1
    _answer(monkeypatch)
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}) == 0
    summary = _summary(tmp_path)
    assert summary["exit_reasons"] == ["ok", "ok"]
    assert [(r["index"], r["exit_reason"]) for r in summary["rejected"]] == [(5, "input-invalid")]


def test_an_unparseable_turn_index_is_recorded_with_no_index(tmp_path, capsys):
    save_state(_state(), tmp_path / "agent-state")
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "two", "MIDOJO_TURN_COUNT": "2"}) == 1
    assert "mini-agents-sandbox-agent: input-invalid: MIDOJO_TURN_INDEX" in capsys.readouterr().err
    (rejected,) = _summary(tmp_path)["rejected"]
    assert rejected["index"] is None
    assert rejected["exit_reason"] == "input-invalid"
    assert load_state(tmp_path / "agent-state").turns == _state().turns


def test_an_unreadable_state_is_left_alone_and_the_summary_extended(tmp_path, capsys):
    write_run_summary(_state(), tmp_path / "workdir")
    state_file = tmp_path / "agent-state" / "state.json"
    state_file.parent.mkdir()
    state_file.write_text("{not json")
    assert _main(tmp_path, {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "x"}) == 1
    assert "mini-agents-sandbox-agent: state-missing: unreadable state file" in (
        capsys.readouterr().err
    )
    assert state_file.read_text() == "{not json"
    summary = _summary(tmp_path)
    assert summary["exit_reasons"] == ["ok"]
    assert [(r["index"], r["exit_reason"]) for r in summary["rejected"]] == [(1, "state-missing")]


def test_a_rejection_without_any_summary_starts_one(tmp_path):
    rejected = Turn(None, None, None, None, 0, "input-invalid", "bad")
    path = record_rejection(tmp_path, domain="klarna", turn_count=None, rejected=rejected)
    summary = json.loads(path.read_text())
    assert summary["turn_count"] is None and summary["final"] is False
    assert summary["rejected"] == [
        {
            "index": None,
            "reply": None,
            "response_id": None,
            "served_model": None,
            "gateway_requests": 0,
            "exit_reason": "input-invalid",
            "failure": "bad",
        }
    ]


def test_a_later_turn_checks_both_pids_with_the_liveness_check(tmp_path, capsys):
    save_state(_state(), tmp_path / "agent-state")
    checked = []

    def alive(pid):
        checked.append(pid)
        return pid != 12

    environ = {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}
    assert _main(tmp_path, environ, alive=alive) == 1
    assert checked == [11, 12]
    assert capsys.readouterr().err.startswith("mini-agents-sandbox-agent: ogx-gone: pid 12 ")
    assert load_state(tmp_path / "agent-state").turns[-1].exit_reason == "ogx-gone"


def test_the_console_script_is_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert (
        project["project"]["scripts"]["mini-agents-sandbox-agent"]
        == "mini_agents.sandbox_agent.cli:main"
    )


def _answer(monkeypatch, sent=None):
    """Stand in for OGX in-process: answer every request as turn 2."""

    def create(url, body, *, timeout, secret=None):
        if sent is not None:
            sent.append(body)
        return {
            "id": "resp-2",
            "model": "served/m",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "reply 2"}]}
            ],
        }

    monkeypatch.setattr(cli, "create_response", create)


SECOND = {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}


# Process identity


def test_identity_is_the_start_time_in_proc(tmp_path):
    root = _proc(tmp_path, 41, "S", start=555)
    assert process_identity(41, proc_root=root) == "proc-start:555"
    assert process_identity(43, proc_root=root) is None
    (root / "44").mkdir()
    (root / "44" / "stat").write_text("44 (short) S 1\n")
    assert process_identity(44, proc_root=root) is None


def test_identity_falls_back_to_ps_without_proc(tmp_path):
    root = tmp_path / "no-proc"
    mine = process_identity(os.getpid(), proc_root=root)
    assert mine is not None and mine.startswith("ps-lstart:")
    assert process_identity(os.getpid(), proc_root=root) == mine
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    assert process_identity(process.pid, proc_root=root) is None


@pytest.mark.parametrize(("pid", "reason"), [(11, "mcp-server-gone"), (12, "ogx-gone")])
def test_a_reused_pid_fails_the_turn_with_the_named_reason(tmp_path, capsys, pid, reason):
    save_state(_state(), tmp_path / "agent-state")

    def identity(checked):
        return f"id-{checked}" if checked != pid else "id-someone-else"

    assert _main(tmp_path, SECOND, alive=lambda checked: True, identity=identity) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(
        f"mini-agents-sandbox-agent: {reason}: pid {pid} is no longer the process "
        "the first turn started"
    )
    assert _summary(tmp_path)["exit_reasons"] == ["ok", reason]


def test_a_live_pid_with_another_start_time_is_gone(tmp_path, capsys):
    """The real checks: this test's own PID runs, but did not start as recorded."""

    state = _state()
    state.mcp.pid = os.getpid()
    state.mcp.identity = "ps-lstart:Thu Jan  1 00:00:00 1970"
    save_state(state, tmp_path / "agent-state")
    assert (
        _main(tmp_path, SECOND, alive=pid_alive, identity=process_identity) == 1
    )
    assert f"mcp-server-gone: pid {os.getpid()} is no longer the process" in (
        capsys.readouterr().err
    )


def test_a_pid_recorded_without_identity_is_gone(tmp_path, capsys):
    state = _state()
    state.ogx.identity = None
    save_state(state, tmp_path / "agent-state")
    assert _main(tmp_path, SECOND, alive=lambda pid: True) == 1
    assert "mini-agents-sandbox-agent: ogx-gone: pid 12 " in capsys.readouterr().err


# Writes that fail


def _unwritable(monkeypatch, name):
    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, name, fail)


def test_a_successful_turn_prints_the_reply_and_records_it(tmp_path, monkeypatch, capsys):
    save_state(_state(), tmp_path / "agent-state")
    _answer(monkeypatch)
    assert _main(tmp_path, SECOND) == 0
    assert capsys.readouterr() == ("reply 2\n", "")
    state = load_state(tmp_path / "agent-state")
    assert state.previous_response_id == "resp-2"
    assert _summary(tmp_path)["final"] is True


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("save_state", "state-write-failed"),
        ("write_run_summary", "evidence-write-failed"),
        ("copy_ogx_log", "evidence-write-failed"),
    ],
)
def test_a_turn_whose_writes_fail_fails_without_a_reply(
    tmp_path, monkeypatch, capsys, name, reason
):
    save_state(_state(), tmp_path / "agent-state")
    (tmp_path / "agent-state" / "ogx.log").write_text("ogx\n")
    _answer(monkeypatch)
    _unwritable(monkeypatch, name)
    assert _main(tmp_path, SECOND) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"mini-agents-sandbox-agent: {reason}: ")
    assert "No space left on device" in err
    # What could still be written says the turn failed, and keeps its reply.
    if name == "save_state":
        turn = _summary(tmp_path)["turns"][-1]
        assert (turn["exit_reason"], turn["reply"]) == (reason, "reply 2")
    else:
        turn = load_state(tmp_path / "agent-state").turns[-1]
        assert (turn.exit_reason, turn.reply, turn.response_id) == (reason, "reply 2", "resp-2")


def test_a_failed_turn_whose_writes_fail_names_both(tmp_path, monkeypatch, capsys):
    save_state(_state(), tmp_path / "agent-state")
    _unwritable(monkeypatch, "write_run_summary")
    assert _main(tmp_path, SECOND, alive=lambda pid: pid != 12) == 1
    err = capsys.readouterr().err.splitlines()
    assert err[0].startswith("mini-agents-sandbox-agent: evidence-write-failed: ")
    assert err[1].startswith("mini-agents-sandbox-agent: ogx-gone: pid 12 ")
    turn = load_state(tmp_path / "agent-state").turns[-1]
    assert turn.exit_reason == "ogx-gone"
    assert "evidence-write-failed" in turn.failure


# The message argument


@pytest.mark.parametrize("message", ["--help", "-h", "--", "-x", "--domain"])
def test_the_last_argument_is_the_message_whatever_it_looks_like(
    tmp_path, monkeypatch, capsys, message
):
    save_state(_state(), tmp_path / "agent-state")
    sent = []
    _answer(monkeypatch, sent)
    assert _main(tmp_path, SECOND, message=message) == 0
    assert capsys.readouterr() == ("reply 2\n", "")
    assert [body["input"] for body in sent] == [message]


def test_a_double_dash_before_the_message_is_dropped():
    args = cli.parse_args(["--domain", "airbnb", "--", "-x"])
    assert (args.domain, args.message) == ("airbnb", "-x")
    args = cli.parse_args(["--domain", "airbnb", "--", "--"])
    assert args.message == "--"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_a_lone_help_flag_prints_help(flag, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.parse_args([flag])
    assert raised.value.code == 0
    assert "usage: mini-agents-sandbox-agent --domain DOMAIN" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["hello"], ["--domain", "klarna"]])
def test_a_missing_domain_or_message_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(argv)
    assert raised.value.code == 2
