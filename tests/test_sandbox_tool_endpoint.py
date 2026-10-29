"""MiDojo's host interceptor in the agent command: ``MIDOJO_TOOL_URL`` and
``MIDOJO_TOOL_TOKEN``, both, neither, or one without the other.

In-process, with OGX stood in for by a function; the end-to-end path through
a fake interceptor is in ``test_sandbox_agent_run.py``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from mini_agents.sandbox_agent import cli, failures
from mini_agents.sandbox_agent.failures import AgentFailure
from mini_agents.sandbox_agent.responses import create_response, request_body
from mini_agents.sandbox_agent.state import AgentState, Process, Turn, load_state, save_state
from mini_agents.sandbox_agent.summary import summary_path
from mini_agents.sandbox_agent.tool_endpoint import (
    REDACTED,
    TOOL_TOKEN,
    TOOL_URL,
    ToolEndpoint,
    read_tool_endpoint,
    redact,
)

URL = "http://host.openshell.internal:18008/mcp"
TOKEN = "tool-token-3f9a"
FULL = {
    "AGENT_MODEL": "m",
    "AGENT_MODEL_BASE_URL": "http://host.openshell.internal:18006/v1",
    "AGENT_MODEL_API_KEY": "placeholder",
    "AGENT_INSTRUCTIONS": "",
}
SECOND = {**FULL, "MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}


# Reading the variables


def test_both_set_is_the_interceptor():
    endpoint = read_tool_endpoint({TOOL_URL: URL, TOOL_TOKEN: TOKEN})
    assert endpoint == ToolEndpoint(url=URL, token=TOKEN)
    assert TOKEN not in repr(endpoint)


def test_surrounding_blanks_are_dropped():
    endpoint = read_tool_endpoint({TOOL_URL: f" {URL}\n", TOOL_TOKEN: f"{TOKEN}\n"})
    assert endpoint == ToolEndpoint(url=URL, token=TOKEN)


@pytest.mark.parametrize(
    "environ",
    [{}, {TOOL_URL: "", TOOL_TOKEN: ""}, {TOOL_URL: " ", TOOL_TOKEN: "\n"}, {TOOL_URL: ""}],
)
def test_neither_set_or_both_blank_is_no_interceptor(environ):
    assert read_tool_endpoint(environ) is None


@pytest.mark.parametrize(
    ("environ", "given", "missing"),
    [
        ({TOOL_URL: URL}, TOOL_URL, TOOL_TOKEN),
        ({TOOL_URL: URL, TOOL_TOKEN: ""}, TOOL_URL, TOOL_TOKEN),
        ({TOOL_URL: URL, TOOL_TOKEN: "  "}, TOOL_URL, TOOL_TOKEN),
        ({TOOL_TOKEN: TOKEN}, TOOL_TOKEN, TOOL_URL),
        ({TOOL_TOKEN: TOKEN, TOOL_URL: ""}, TOOL_TOKEN, TOOL_URL),
    ],
)
def test_one_without_the_other_is_refused_naming_both(environ, given, missing):
    with pytest.raises(AgentFailure) as raised:
        read_tool_endpoint(environ)
    assert raised.value.reason == failures.INPUT_MISSING
    detail = raised.value.detail
    assert detail.startswith(f"{given} is set but {missing} is not: ")
    assert TOOL_URL in detail and TOOL_TOKEN in detail
    assert TOKEN not in detail


@pytest.mark.parametrize(
    "url", ["host.openshell.internal:18008/mcp", "ftp://h/mcp", "http:///mcp", "http://[x/mcp"]
)
def test_a_url_that_is_not_http_is_invalid_and_not_echoed(url):
    with pytest.raises(AgentFailure) as raised:
        read_tool_endpoint({TOOL_URL: url, TOOL_TOKEN: TOKEN})
    assert raised.value.reason == failures.INPUT_INVALID
    assert raised.value.detail == f"{TOOL_URL} is not an http(s) URL"


def test_swapped_values_do_not_print_the_token():
    with pytest.raises(AgentFailure) as raised:
        read_tool_endpoint({TOOL_URL: TOKEN, TOOL_TOKEN: URL})
    assert TOKEN not in raised.value.detail


def test_the_inputs_carry_the_endpoint():
    assert cli.read_inputs(FULL).tool_endpoint is None
    inputs = cli.read_inputs({**FULL, TOOL_URL: URL, TOOL_TOKEN: TOKEN})
    assert inputs.tool_endpoint == ToolEndpoint(url=URL, token=TOKEN)


def test_redact_replaces_the_token_and_keeps_the_rest():
    endpoint = ToolEndpoint(url=URL, token=TOKEN)
    failure = AgentFailure("r", f"echo {TOKEN} and {TOKEN}", gateway_requests=1)
    redacted = redact(failure, endpoint)
    assert (redacted.reason, redacted.gateway_requests) == ("r", 1)
    assert redacted.detail == f"echo {REDACTED} and {REDACTED}"
    assert TOKEN not in str(redacted)
    assert redact(failure, None) is failure


# The request


def test_the_token_goes_in_the_tools_authorization_not_its_headers():
    body = request_body(
        model="m",
        message="hello",
        instructions="",
        mcp_url=URL,
        previous_response_id=None,
        authorization=TOKEN,
    )
    assert body["tools"] == [
        {
            "type": "mcp",
            "server_label": "midojo",
            "server_url": URL,
            "require_approval": "never",
            "authorization": TOKEN,
        }
    ]


# A turn


def _state():
    return AgentState(
        domain="klarna",
        turn_count=2,
        mcp=Process(pid=11, port=20001, command=["python-mcp"], log="/s/mcp.log", identity="id-11"),
        ogx=Process(pid=12, port=20002, command=["python-ogx"], log="/s/ogx.log", identity="id-12"),
        previous_response_id="resp-1",
        turns=[Turn(0, "reply 1", "resp-1", "served/m", 1, "ok")],
    )


def _main(tmp_path, environ, *, alive=lambda pid: True):
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
            "hello",
        ],
        environ,
        alive=alive,
        identity=lambda pid: f"id-{pid}",
    )


def _answer(monkeypatch, sent):
    def create(url, body, *, timeout, secret=None):
        sent.append(body)
        return {
            "id": "resp-2",
            "model": "served/m",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "reply 2"}]}
            ],
        }

    monkeypatch.setattr(cli, "create_response", create)


def _files_text(*roots):
    return "".join(
        path.read_text(errors="replace")
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )


def test_a_turn_behind_the_interceptor_sends_its_tools_there(tmp_path, monkeypatch, capsys):
    save_state(_state(), tmp_path / "agent-state")
    sent = []
    _answer(monkeypatch, sent)
    assert _main(tmp_path, {**SECOND, TOOL_URL: URL, TOOL_TOKEN: TOKEN}) == 0
    out, err = capsys.readouterr()
    assert (out, err) == ("reply 2\n", "")
    (tool,) = sent[0]["tools"]
    assert (tool["server_url"], tool["authorization"]) == (URL, TOKEN)
    assert "headers" not in tool
    assert TOKEN not in _files_text(tmp_path / "agent-state", tmp_path / "workdir")


def _first_turn_behind_the_interceptor():
    """The state a first turn behind the interceptor leaves: OGX alone."""

    state = _state()
    state.interceptor, state.mcp = True, None
    return state


def test_a_later_turn_behind_the_interceptor_checks_ogx_alone(tmp_path, monkeypatch, capsys):
    save_state(_first_turn_behind_the_interceptor(), tmp_path / "agent-state")
    sent, checked = [], []
    _answer(monkeypatch, sent)

    def alive(pid):
        checked.append(pid)
        return True

    assert _main(tmp_path, {**SECOND, TOOL_URL: URL, TOOL_TOKEN: TOKEN}, alive=alive) == 0
    assert capsys.readouterr() == ("reply 2\n", "")
    assert checked == [12]
    (tool,) = sent[0]["tools"]
    assert tool["server_url"] == URL
    state = load_state(tmp_path / "agent-state")
    assert state.interceptor and state.mcp is None


def test_a_later_turn_without_the_interceptor_needs_the_server_the_first_did_not_start(
    tmp_path, monkeypatch, capsys
):
    save_state(_first_turn_behind_the_interceptor(), tmp_path / "agent-state")
    sent = []
    _answer(monkeypatch, sent)
    assert _main(tmp_path, SECOND) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(
        "mini-agents-sandbox-agent: mcp-server-gone: the first turn did not start it"
    )
    assert TOOL_URL in err and TOOL_TOKEN in err
    assert sent == []
    summary = json.loads(summary_path(tmp_path / "workdir").read_text())
    assert summary["exit_reasons"] == ["ok", "mcp-server-gone"]


def test_a_later_turn_behind_the_interceptor_still_checks_a_server_the_first_started(
    tmp_path, monkeypatch, capsys
):
    """A first turn without the interceptor started both; a later turn checks both."""

    save_state(_state(), tmp_path / "agent-state")
    _answer(monkeypatch, [])
    checked = []

    def alive(pid):
        checked.append(pid)
        return pid != 11

    assert _main(tmp_path, {**SECOND, TOOL_URL: URL, TOOL_TOKEN: TOKEN}, alive=alive) == 1
    assert checked == [11]
    assert capsys.readouterr().err.startswith("mini-agents-sandbox-agent: mcp-server-gone: pid 11 ")


def test_a_turn_without_the_interceptor_is_unchanged(tmp_path, monkeypatch):
    save_state(_state(), tmp_path / "agent-state")
    sent = []
    _answer(monkeypatch, sent)
    assert _main(tmp_path, SECOND) == 0
    assert sent[0]["tools"] == [
        {
            "type": "mcp",
            "server_label": "midojo",
            "server_url": "http://127.0.0.1:20001/sse",
            "require_approval": "never",
        }
    ]


@pytest.mark.parametrize(
    "half", [{TOOL_URL: URL}, {TOOL_TOKEN: TOKEN}, {TOOL_URL: URL, TOOL_TOKEN: ""}]
)
def test_a_half_configured_interceptor_fails_the_first_turn_before_anything_starts(
    tmp_path, monkeypatch, capsys, half
):
    sent = []
    _answer(monkeypatch, sent)
    assert _main(tmp_path, {**FULL, **half}) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("mini-agents-sandbox-agent: input-missing: ")
    assert TOOL_URL in err and TOOL_TOKEN in err
    assert TOKEN not in err
    assert sent == []
    state = load_state(tmp_path / "agent-state")
    assert state.mcp is None and state.ogx is None
    summary = json.loads(summary_path(tmp_path / "workdir").read_text())
    assert summary["exit_reasons"] == ["input-missing"]
    assert TOKEN not in _files_text(tmp_path / "agent-state", tmp_path / "workdir")


def test_a_gateway_failure_that_echoes_the_token_is_redacted(tmp_path, monkeypatch, capsys):
    save_state(_state(), tmp_path / "agent-state")

    def create(url, body, *, timeout, secret=None):
        raise AgentFailure(
            failures.GATEWAY_REQUEST_FAILED,
            f"POST {url}: HTTP 400 {json.dumps(body)}",
            gateway_requests=1,
        )

    monkeypatch.setattr(cli, "create_response", create)
    assert _main(tmp_path, {**SECOND, TOOL_URL: URL, TOOL_TOKEN: TOKEN}) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("mini-agents-sandbox-agent: gateway-request-failed: ")
    assert REDACTED in err and TOKEN not in err
    turn = load_state(tmp_path / "agent-state").turns[-1]
    assert turn.exit_reason == "gateway-request-failed" and turn.gateway_requests == 1
    assert TOKEN not in _files_text(tmp_path / "agent-state", tmp_path / "workdir")


# A token echoed across the cut OGX's answer gets in a failure detail


STRADDLER = "Zq8vK2mP4rT7wX1y"


def _fragments(text, secret=STRADDLER, size=4):
    """Every piece of ``secret`` of ``size`` characters or more in ``text``."""

    return [
        secret[start:end]
        for start in range(len(secret))
        for end in range(start + size, len(secret) + 1)
        if secret[start:end] in text
    ]


def _straddling_gateway():
    """OGX answering 400 with the token across the 200-character cut."""

    prefix = "x" * 192
    body = prefix + STRADDLER + " rest of the echoed request"
    assert len(prefix) < 200 < len(prefix) + len(STRADDLER)
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(400, text=body)))


def test_an_unredacted_cut_would_leak_a_prefix_of_the_token():
    # Without the secret, the cut leaves a piece a later replacement misses.
    with pytest.raises(AgentFailure) as raised:
        create_response("http://127.0.0.1:9", {}, timeout=1, client=_straddling_gateway())
    assert STRADDLER not in raised.value.detail
    assert _fragments(raised.value.detail)


def test_a_token_across_the_cut_leaves_no_fragment():
    with pytest.raises(AgentFailure) as raised:
        create_response(
            "http://127.0.0.1:9", {}, timeout=1, client=_straddling_gateway(), secret=STRADDLER
        )
    assert raised.value.reason == failures.GATEWAY_REQUEST_FAILED
    assert raised.value.detail.startswith("POST http://127.0.0.1:9/v1/responses: HTTP 400 xxx")
    assert _fragments(raised.value.detail) == []


def test_a_turn_whose_gateway_echoes_the_token_across_the_cut_leaks_no_fragment(
    tmp_path, monkeypatch, capsys
):
    save_state(_state(), tmp_path / "agent-state")
    gateway = _straddling_gateway()

    def create(url, body, *, timeout, secret=None):
        return create_response(url, body, timeout=timeout, client=gateway, secret=secret)

    monkeypatch.setattr(cli, "create_response", create)
    assert _main(tmp_path, {**SECOND, TOOL_URL: URL, TOOL_TOKEN: STRADDLER}) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("mini-agents-sandbox-agent: gateway-request-failed: ")
    assert _fragments(out + err) == []
    assert _fragments(_files_text(tmp_path / "agent-state", tmp_path / "workdir")) == []
