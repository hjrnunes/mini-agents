"""The model shim rewrites a model's chat completions and passes the rest on.

``rewritten`` is the pure rule set; ``ModelShim`` is the loopback proxy that
applies it to one upstream. The upstream here is a local stub, so no test makes
a model call. The real key appears only in the stub's record of the
``Authorization`` header, never in the shim's log.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
from model_stub import ANSWER, StubModel

from mini_agents.kernel import model_shim
from mini_agents.kernel.model_shim import (
    MODEL_REWRITES,
    ModelShim,
    Rewrites,
    rewrites_for,
    rewritten,
)

GPT_6_SOL = MODEL_REWRITES["gpt-6-sol"]
KEY = "sk-real-key-0123456789"
PLACEHOLDER = "sk-placeholder"


def _body(**fields):
    return json.dumps({"model": "gpt-6-sol", "messages": [], **fields}).encode()


def test_gpt_6_sol_asks_for_no_reasoning_and_max_completion_tokens():
    assert GPT_6_SOL == Rewrites(
        reasoning_effort="none", token_param="max_completion_tokens"
    )


def test_a_model_that_is_not_listed_gets_no_rewrite():
    assert not rewrites_for("gemma-4-26b-a4b-it").active
    assert not rewrites_for(None).active
    assert not rewrites_for("").active
    assert rewrites_for("gpt-6-sol") == GPT_6_SOL


def test_max_tokens_as_the_token_param_changes_nothing():
    assert not Rewrites(token_param="max_tokens").active


def test_a_body_with_both_token_fields_gets_the_effort_and_loses_max_tokens():
    body = _body(max_tokens=4096, max_completion_tokens=2048, tools=[{"x": 1}])

    out, changes = rewritten(body, GPT_6_SOL)

    assert changes == [
        {"op": "inject", "field": "reasoning_effort", "value": "none"},
        {"op": "drop", "field": "max_tokens"},
    ]
    assert json.loads(out) == {
        "model": "gpt-6-sol",
        "messages": [],
        "max_completion_tokens": 2048,
        "tools": [{"x": 1}],
        "reasoning_effort": "none",
    }
    assert out == json.dumps(json.loads(out), separators=(",", ":")).encode()


def test_a_body_with_max_tokens_alone_has_it_renamed():
    out, changes = rewritten(_body(max_tokens=512), GPT_6_SOL)

    assert changes[1] == {
        "op": "rename",
        "from": "max_tokens",
        "to": "max_completion_tokens",
    }
    assert json.loads(out)["max_completion_tokens"] == 512
    assert "max_tokens" not in json.loads(out)


def test_an_existing_reasoning_effort_is_overwritten_and_the_old_value_kept():
    out, changes = rewritten(_body(reasoning_effort="high"), GPT_6_SOL)

    assert changes == [
        {
            "op": "overwrite",
            "field": "reasoning_effort",
            "was": "high",
            "value": "none",
        }
    ]
    assert json.loads(out)["reasoning_effort"] == "none"


def test_a_body_with_no_token_field_only_gets_the_effort():
    out, changes = rewritten(_body(), GPT_6_SOL)

    assert [change["op"] for change in changes] == ["inject"]
    assert set(json.loads(out)) == {"model", "messages", "reasoning_effort"}


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'"text"', b""])
def test_a_body_that_is_not_a_json_object_comes_back_byte_for_byte(body):
    assert rewritten(body, GPT_6_SOL) == (body, [])


def test_a_model_with_no_rules_gets_its_body_byte_for_byte():
    body = b'{"model":  "other",  "max_tokens": 5}'

    assert rewritten(body, Rewrites()) == (body, [])
    assert rewritten(body, Rewrites(token_param="max_tokens")) == (body, [])


def test_the_rules_come_from_a_table_by_model_id():
    table = {"a": Rewrites(reasoning_effort="low")}

    assert rewrites_for("a", table) == Rewrites(reasoning_effort="low")
    assert rewrites_for("gpt-6-sol", table) == Rewrites()


@pytest.fixture
def upstream():
    with StubModel("gpt-6-sol") as stub:
        yield stub


def _shim(upstream, tmp_path, **options):
    return ModelShim(
        upstream.url,
        options.pop("key", KEY),
        log_path=tmp_path / "log",
        **options,
    )


def _post(shim, path, content, headers=None):
    return httpx.post(
        f"{shim.url}{path}",
        content=content,
        headers={"Authorization": f"Bearer {PLACEHOLDER}", **(headers or {})},
        timeout=10,
    )


def test_a_chat_completion_reaches_the_upstream_rewritten_with_the_real_key(
    upstream, tmp_path
):
    body = _body(max_tokens=4096, max_completion_tokens=2048)
    with _shim(upstream, tmp_path) as shim:
        response = _post(shim, "/chat/completions", body)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ANSWER
    (received,) = upstream.completions()
    assert received.path == "/v1/chat/completions"
    assert received.headers["authorization"] == f"Bearer {KEY}"
    assert received.json() == {
        "model": "gpt-6-sol",
        "messages": [],
        "max_completion_tokens": 2048,
        "reasoning_effort": "none",
    }


def test_a_trailing_slash_does_not_hide_the_route(upstream, tmp_path):
    with _shim(upstream, tmp_path) as shim:
        _post(shim, "/chat/completions/", _body(max_tokens=1))

    assert "max_tokens" not in upstream.completions()[0].json()


def test_a_streamed_answer_arrives_before_the_upstream_finishes(tmp_path):
    release = threading.Event()
    with StubModel("gpt-6-sol", pause=release) as upstream:
        with _shim(upstream, tmp_path) as shim:
            with httpx.stream(
                "POST",
                f"{shim.url}/chat/completions",
                content=_body(stream=True),
                timeout=10,
            ) as response:
                lines = response.iter_lines()
                first = next(line for line in lines if line.startswith("data:"))
                assert ANSWER in first
                # The stub holds the rest back until the first chunk is read.
                release.set()
                rest = [line for line in lines if line.startswith("data:")]

    assert rest[-1] == "data: [DONE]"


def test_other_routes_and_methods_pass_through_unchanged(upstream, tmp_path):
    odd = b'{"model":  "gpt-6-sol",   "max_tokens": 9}'
    with _shim(upstream, tmp_path) as shim:
        models = httpx.get(f"{shim.url}/models", timeout=10)
        other = _post(shim, "/embeddings", odd)
        text = _post(shim, "/chat/completions", b"not json")

    assert models.json()["data"][0]["id"] == "gpt-6-sol"
    assert other.status_code == 404
    sent = {r.path: r for r in upstream.requests}
    assert sent["/v1/embeddings"].body == odd
    assert [r.body for r in upstream.completions()] == [b"not json"]
    assert text.status_code == 400
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in upstream.requests)


def test_the_clients_own_authorization_passes_when_the_shim_has_no_key(
    upstream, tmp_path
):
    with _shim(upstream, tmp_path, key=None) as shim:
        _post(shim, "/chat/completions", _body())

    assert upstream.completions()[0].headers["authorization"] == f"Bearer {PLACEHOLDER}"


def test_a_path_outside_the_base_path_is_refused_without_a_call(upstream, tmp_path):
    with _shim(upstream, tmp_path) as shim:
        response = httpx.post(
            f"http://127.0.0.1:{shim.port}/other/chat/completions",
            content=_body(),
            timeout=10,
        )

    assert response.status_code == 404
    assert upstream.requests == []


def test_a_chunked_request_body_is_refused(upstream, tmp_path):
    with _shim(upstream, tmp_path) as shim:
        with socket.create_connection(("127.0.0.1", shim.port), timeout=10) as sock:
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            )
            assert sock.recv(64).startswith(b"HTTP/1.0 411")

    assert upstream.requests == []


def test_an_unreachable_provider_is_a_502_and_a_logged_error(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = sock.getsockname()[1]
    shim = ModelShim(f"http://127.0.0.1:{dead}/v1", KEY, log_path=tmp_path / "log")
    with shim:
        response = httpx.post(
            f"{shim.url}/chat/completions", content=_body(), timeout=10
        )

    assert response.status_code == 502
    (row,) = [json.loads(line) for line in (tmp_path / "log").read_text().splitlines()]
    assert row["status"] == 0
    assert row["error"] == "ConnectError"


def test_the_rewrite_log_names_each_completion_and_holds_no_secret(upstream, tmp_path):
    with _shim(upstream, tmp_path) as shim:
        _post(shim, "/chat/completions", _body(max_tokens=3, max_completion_tokens=3))
        _post(shim, "/chat/completions", _body())
        httpx.get(f"{shim.url}/models", timeout=10)

    text = (tmp_path / "log").read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert rows[0] == {
        "sequence": 1,
        "route": "/chat/completions",
        "status": 200,
        "rewrites": [
            {"op": "inject", "field": "reasoning_effort", "value": "none"},
            {"op": "drop", "field": "max_tokens"},
        ],
        "error": None,
    }
    assert [change["op"] for change in rows[1]["rewrites"]] == ["inject"]
    assert KEY not in text and PLACEHOLDER not in text
    assert "messages" not in text


@pytest.mark.parametrize(
    "named",
    [{"model": "another-model"}, {"model": ""}, {"model": 7}, {}],
    ids=["other model", "empty name", "not a name", "no model"],
)
def test_a_request_that_does_not_name_a_listed_model_is_forwarded_untouched(
    upstream, tmp_path, named
):
    body = json.dumps({**named, "max_tokens": 7, "messages": []}, indent=1).encode()
    with _shim(upstream, tmp_path) as shim:
        _post(shim, "/chat/completions", body)

    (received,) = upstream.completions()
    assert received.body == body
    (row,) = [json.loads(line) for line in (tmp_path / "log").read_text().splitlines()]
    assert row["rewrites"] == []


def test_a_shim_applies_the_rules_of_the_model_each_request_names(upstream, tmp_path):
    table = {
        "a": Rewrites(reasoning_effort="low"),
        "b": Rewrites(reasoning_effort="high"),
    }
    with ModelShim(upstream.url, KEY, table) as shim:
        for name in ("a", "b", "c"):
            _post(shim, "/chat/completions", _body(model=name))

    efforts = [r.json().get("reasoning_effort") for r in upstream.completions()]
    assert efforts == ["low", "high", None]


def test_a_value_the_client_chose_never_reaches_the_log(upstream, tmp_path):
    secret = "sk-client-chose-this-0123456789"
    with _shim(upstream, tmp_path) as shim:
        _post(shim, "/chat/completions", _body(reasoning_effort=secret))

    (received,) = upstream.completions()
    assert received.json()["reasoning_effort"] == "none"
    text = (tmp_path / "log").read_text()
    assert secret not in text
    (row,) = [json.loads(line) for line in text.splitlines()]
    assert row["rewrites"] == [
        {"op": "overwrite", "field": "reasoning_effort", "value": "none"}
    ]


def test_every_value_in_a_log_row_is_the_shims_own(upstream, tmp_path):
    marker = "client-value-0123456789"
    body = _body(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": marker}],
        reasoning_effort=marker,
        max_tokens=1,
        max_completion_tokens=1,
        metadata={"k": marker},
    )
    with _shim(upstream, tmp_path) as shim:
        _post(shim, "/chat/completions", body)

    assert marker not in (tmp_path / "log").read_text()


def test_a_shim_without_a_log_path_writes_no_log(upstream, tmp_path):
    shim = ModelShim(upstream.url, KEY)
    with shim:
        _post(shim, "/chat/completions", _body())

    assert not (tmp_path / "log").exists()


def _rows(tmp_path):
    log = tmp_path / "log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _stream_on_a_thread(shim, first_chunk):
    """Read a streamed completion on a thread; set ``first_chunk`` on its first line."""

    def read():
        try:
            with httpx.stream(
                "POST",
                f"{shim.url}/chat/completions",
                content=_body(stream=True),
                timeout=30,
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        first_chunk.set()
        except httpx.HTTPError:
            pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    return reader


def test_stopping_the_shim_waits_for_a_request_in_flight_and_its_log_row(
    tmp_path, monkeypatch
):
    # The drain must not end first: the provider is released well inside it.
    monkeypatch.setattr(model_shim, "DRAIN_SECONDS", 30.0, raising=False)
    release = threading.Event()
    first_chunk = threading.Event()
    with StubModel("gpt-6-sol", pause=release) as upstream:
        shim = _shim(upstream, tmp_path).start()
        reader = _stream_on_a_thread(shim, first_chunk)
        assert first_chunk.wait(timeout=10)

        stopper = threading.Thread(target=shim.stop, daemon=True)
        stopper.start()
        # The server notices a shutdown within its 0.5 s poll. A stop that does
        # not wait for the request is done long before this window ends.
        stopper.join(timeout=3.0)
        waiting = stopper.is_alive()
        release.set()
        stopper.join(timeout=10)
        reader.join(timeout=5)
        rows = _rows(tmp_path)

    assert waiting
    assert [(row["status"], row["error"]) for row in rows] == [(200, None)]


def test_stopping_the_shim_waits_for_the_drain_and_no_longer_on_a_stuck_request(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(model_shim, "DRAIN_SECONDS", 1.0, raising=False)
    first_chunk = threading.Event()
    with StubModel("gpt-6-sol", pause=threading.Event()) as upstream:
        shim = _shim(upstream, tmp_path).start()
        # A daemon thread: stop does not close the abandoned request's client.
        _stream_on_a_thread(shim, first_chunk)
        assert first_chunk.wait(timeout=10)

        started = time.monotonic()
        shim.stop()
        stopped = time.monotonic() - started
        shim.stop()
        again = time.monotonic() - started - stopped

    # The stub holds the request for 30 s. A stop waits for the 1 s drain, not
    # for the request, and a second stop does not wait for the drain again.
    assert 1.0 <= stopped < 5
    assert again < 0.3
