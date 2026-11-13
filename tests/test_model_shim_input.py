"""What a client sends the model shim never comes back out of it.

The shim runs in a process of its own, as in the gateway, so its stderr is the
file the tests read. Each case puts a marker through one field a client
controls. The shim answers a malformed field with a fixed error and does not
forward the request. No marker reaches stderr, the rewrite log or an error
reply. ``tests/test_model_shim.py`` covers the rewrites themselves.
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from model_stub import StubModel

from mini_agents.kernel import model_shim
from mini_agents.kernel.model_shim import MODEL_REWRITES, ModelShim

ROOT = Path(__file__).resolve().parent.parent
KEY = "sk-shim-input-test-key-0123456789abcdef"
# The body cap of the shim under test; far below the shim's own default.
CAP = 1024
# More leading zeros than ``int`` converts: Python refuses a string of over
# 4300 digits, whatever its value.
PADDING = 5000
EXCHANGE_TIMEOUT = 5.0

SHIM_MAIN = """
import sys
import time
from pathlib import Path

from mini_agents.kernel import model_shim

upstream, key, log, cap = sys.argv[1:5]
model_shim.MAX_BODY_BYTES = int(cap)
shim = model_shim.ModelShim(upstream, key, log_path=Path(log))
shim.start()
print(shim.port, flush=True)
time.sleep(600)
"""

BODY = '{"model": "gpt-6-sol", "messages": []}'
LINE = "POST /v1/chat/completions HTTP/1.1"


@dataclass
class Running:
    port: int
    upstream: StubModel
    log: Path
    stderr: Path
    _read: int = 0

    def new_stderr(self) -> bytes:
        """Return what the shim wrote to stderr since the last call."""
        data = self.stderr.read_bytes()
        fresh, self._read = data[self._read :], len(data)
        return fresh

    def log_bytes(self) -> bytes:
        return self.log.read_bytes() if self.log.exists() else b""


@contextlib.contextmanager
def _running(tmp_path: Path):
    log, stderr = tmp_path / "rewrites.jsonl", tmp_path / "shim.stderr"
    with StubModel("gpt-6-sol") as upstream, stderr.open("wb") as errors:
        process = subprocess.Popen(
            [sys.executable, "-c", SHIM_MAIN, upstream.url, KEY, str(log), str(CAP)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            assert process.stdout is not None
            line = process.stdout.readline()
            assert line, "the shim process did not start"
            yield Running(int(line), upstream, log, stderr)
        finally:
            process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()


@pytest.fixture(scope="module")
def running(tmp_path_factory):
    with _running(tmp_path_factory.mktemp("shim")) as shim:
        yield shim


def _exchange(port: int, raw: str, *, half_close: bool = False) -> bytes:
    """Send ``raw`` and return everything the shim answers until it closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=EXCHANGE_TIMEOUT) as s:
        s.sendall(raw.encode("latin-1"))
        if half_close:
            s.shutdown(socket.SHUT_WR)
        chunks = []
        try:
            while chunk := s.recv(65536):
                chunks.append(chunk)
        except TimeoutError:
            pytest.fail(f"the shim sent no end of response in {EXCHANGE_TIMEOUT:.0f}s")
    return b"".join(chunks)


def _answer(response: bytes) -> tuple[int | None, bytes]:
    """Return the status and the body; the status is None when the reply has none.

    A request line of two words is an HTTP/0.9 request, which a reply of body
    alone answers.
    """
    assert response, "the shim closed the connection without an answer"
    if not response.startswith(b"HTTP/"):
        return None, response
    head, _, body = response.partition(b"\r\n\r\n")
    return int(head.split(b" ", 2)[1]), body


def _request(*headers: str, line: str = LINE, body: str = BODY) -> str:
    """Build a well-formed request around ``headers``; add its Content-Length."""
    length = f"Content-Length: {len(body.encode('latin-1'))}"
    return "\r\n".join([line, "Host: shim", *headers, length, "", body])


def _raw(line: str, *headers: str, body: str = "") -> Callable[[str], str]:
    """Return a builder of a request whose line, headers and body are as given.

    ``{m}`` in the line or a header stands for the case's marker.
    """
    return lambda marker: "\r\n".join(
        [line.format(m=marker), *[h.format(m=marker) for h in headers], "", body]
    )


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[str], str]
    status: int | None
    half_close: bool = False

    @property
    def marker(self) -> str:
        return f"MARKER-{self.name}"


REFUSED = [
    # Content-Length
    Case("length-not-a-number", _raw(LINE, "Content-Length: {m}"), 400),
    Case("length-with-a-sign", _raw(LINE, "Content-Length: +5", body="abcde"), 400),
    Case("length-with-underscore", _raw(LINE, "Content-Length: 1_0"), 400),
    Case("length-negative", _raw(LINE, "Content-Length: -1", body=BODY), 400),
    Case("length-empty", _raw(LINE, "Content-Length:", body=BODY), 400),
    Case(
        "length-twice",
        _raw(LINE, "Content-Length: 5", "Content-Length: 38", body=BODY),
        400,
    ),
    Case("length-over-the-cap", _raw(LINE, f"Content-Length: {CAP + 1}"), 413),
    Case("length-too-big-to-allocate", _raw(LINE, "Content-Length: " + "9" * 20), 413),
    Case(
        "length-padded-with-zeros-over-the-cap",
        _raw(LINE, "Content-Length: " + "0" * PADDING + str(CAP + 1)),
        413,
    ),
    Case(
        "length-past-the-body",
        _raw(LINE, "Content-Length: 500", body=BODY),
        400,
        half_close=True,
    ),
    # The request line
    Case(
        "target-with-a-control-character",
        lambda m: _request(line=f"POST /v1/chat/completions\x7f{m} HTTP/1.1"),
        400,
    ),
    Case(
        "target-with-a-low-control-character",
        lambda m: _request(line=f"POST /v1/chat/completions/\x01{m} HTTP/1.1"),
        400,
    ),
    Case(
        "target-with-a-non-ascii-character",
        lambda m: _request(line=f"POST /v1/chat/completions/\xe9{m} HTTP/1.1"),
        400,
    ),
    Case("method-the-shim-does-not-serve", _raw("MARK{m} /v1/x HTTP/1.1"), 501),
    # Transfer-Encoding: a chunked body needs a Content-Length the client lacks,
    # whichever header or list entry names it.
    Case(
        "chunked-in-one-header",
        _raw(LINE, "Transfer-Encoding: chunked", "Content-Length: 0"),
        411,
    ),
    Case(
        "chunked-after-another-coding",
        _raw(LINE, "Transfer-Encoding: gzip, chunked", "Content-Length: 0"),
        411,
    ),
    Case(
        "chunked-in-a-second-header",
        _raw(
            LINE,
            "Transfer-Encoding: identity",
            "Transfer-Encoding: chunked",
            "Content-Length: 0",
        ),
        411,
    ),
    # A request line the server cannot read is answered as HTTP/0.9: no status.
    Case("request-line-garbled", _raw("GARBAGE {m} x y"), None),
    Case("request-line-of-two-words", _raw("GARBAGE {m}"), None),
    Case("http-version-garbled", _raw("GET /v1/models HTTP/{m}"), None),
    # Headers
    Case("header-value-non-ascii", lambda m: _request(f"X-Probe: {m}\xe9"), 400),
    Case("header-value-control", lambda m: _request(f"X-Probe: {m}\x01"), 400),
    Case("header-name-not-a-token", lambda m: _request(f"X-({m}): v"), 400),
]

# Values the shim passes on without ever recording them.
FORWARDED = [
    Case(
        "query-string",
        lambda m: _request(line=f"GET /v1/models?q={m} HTTP/1.1", body=""),
        200,
    ),
    Case(
        "ordinary-headers",
        lambda m: _request(
            f"Authorization: Bearer {m}",
            f"Connection: {m}",
            f"Content-Type: {m}",
            f"User-Agent: {m}",
            f"X-Probe: {m}",
        ),
        200,
    ),
    Case("transfer-encoding-text", lambda m: _request(f"Transfer-Encoding: {m}"), 200),
    # A length of digits is a length however many zeros lead it.
    Case(
        "length-padded-with-zeros",
        _raw(
            LINE,
            "Content-Length: " + "0" * PADDING + str(len(BODY)),
            body=BODY,
        ),
        200,
    ),
    Case(
        "length-of-only-zeros",
        _raw("GET /v1/models HTTP/1.1", "Content-Length: " + "0" * PADDING),
        200,
    ),
    Case(
        "model-name",
        lambda m: _request(body=json.dumps({"model": m, "messages": []})),
        200,
    ),
    Case(
        "values-a-rewrite-replaces",
        lambda m: _request(
            body=json.dumps(
                {
                    "model": "gpt-6-sol",
                    "messages": [],
                    "reasoning_effort": m,
                    "max_tokens": m,
                    "max_completion_tokens": m,
                }
            )
        ),
        200,
    ),
    # The stand-in provider refuses a body that is not JSON; the shim relays it.
    Case("body-that-is-not-json", lambda m: _request(body=m), 400),
]


def _fixed_error(body: bytes) -> None:
    payload = json.loads(body)
    assert set(payload) == {"error"}
    assert isinstance(payload["error"], str) and payload["error"]


@pytest.mark.parametrize("case", REFUSED, ids=lambda case: case.name)
def test_a_malformed_request_gets_a_fixed_error_and_is_not_forwarded(running, case):
    running.new_stderr()
    seen = len(running.upstream.requests)

    response = _exchange(
        running.port, case.build(case.marker), half_close=case.half_close
    )

    errors = running.new_stderr()
    assert case.marker.encode() not in errors, errors.decode(errors="replace")
    assert case.marker.encode() not in running.log_bytes()
    assert errors == b"", errors.decode(errors="replace")
    status, body = _answer(response)
    assert status == case.status
    _fixed_error(body)
    assert case.marker.encode() not in response
    assert len(running.upstream.requests) == seen, "the shim forwarded the request"


@pytest.mark.parametrize("case", FORWARDED, ids=lambda case: case.name)
def test_a_value_the_shim_passes_on_is_never_logged_or_printed(running, case):
    running.new_stderr()
    seen = len(running.upstream.requests)

    response = _exchange(running.port, case.build(case.marker))

    errors = running.new_stderr()
    assert case.marker.encode() not in errors, errors.decode(errors="replace")
    assert case.marker.encode() not in running.log_bytes()
    assert errors == b"", errors.decode(errors="replace")
    assert _answer(response)[0] == case.status
    assert len(running.upstream.requests) == seen + 1


def test_a_rewrite_row_holds_the_rule_and_not_the_clients_values(running):
    running.new_stderr()
    marker = "MARKER-the-last-row"
    body = json.dumps(
        {"model": "gpt-6-sol", "reasoning_effort": marker, "max_tokens": 3}
    )

    _exchange(running.port, _request(body=body))

    rows = [json.loads(line) for line in running.log_bytes().splitlines()]
    assert rows[-1]["rewrites"] == [
        {"op": "overwrite", "field": "reasoning_effort", "value": "none"},
        {"op": "rename", "from": "max_tokens", "to": "max_completion_tokens"},
    ]
    assert marker.encode() not in running.log_bytes()


def test_a_client_that_resets_in_the_middle_of_a_body_leaves_no_traceback(tmp_path):
    marker = "MARKER-reset"
    unfinished = f"{LINE}\r\nHost: shim\r\nContent-Length: 5000\r\n\r\nabc{marker}"
    with _running(tmp_path) as shim:
        with socket.create_connection(("127.0.0.1", shim.port), timeout=5) as s:
            s.sendall(unfinished.encode())
            # Close with a reset, not a FIN.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        # The shim still serves the next client.
        answer = _exchange(shim.port, _request(line="GET /v1/models HTTP/1.1", body=""))
        assert _answer(answer)[0] == 200

    text = shim.stderr.read_text()
    assert marker not in text
    assert "Traceback" not in text
    for line in text.splitlines():
        assert re.fullmatch(r"model shim: a request failed \(\w+\)", line), line


def _recording_transport(seen: list[bytes]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        # A stream, not content: the shim reads the reply as a stream.
        return httpx.Response(200, stream=httpx.ByteStream(b'{"ok": true}'))

    return httpx.MockTransport(handler)


def _post(shim: ModelShim, content: bytes) -> httpx.Response:
    return httpx.post(f"{shim.url}/chat/completions", content=content, timeout=20)


def test_a_body_nested_past_the_json_limit_is_forwarded_unchanged(capfd):
    seen: list[bytes] = []
    # Deeper than the JSON parser reads on any thread stack.
    body = b"[" * 1_000_000
    transport = _recording_transport(seen)
    with ModelShim("http://provider.invalid/v1", KEY, transport=transport) as shim:
        response = _post(shim, body)

    assert response.status_code == 200
    assert seen == [body]
    assert capfd.readouterr().err == ""


def test_a_body_the_shim_cannot_serialize_again_comes_back_byte_for_byte(monkeypatch):
    def too_deep(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(model_shim.json, "dumps", too_deep)
    body = b'{"model": "gpt-6-sol", "messages": []}'

    assert model_shim.rewritten(body, MODEL_REWRITES["gpt-6-sol"]) == (body, [])


def test_a_body_of_several_megabytes_is_forwarded_whole(capfd):
    seen: list[bytes] = []
    body = json.dumps(
        {"model": "other", "messages": [{"role": "user", "content": "x" * 5_000_000}]}
    ).encode()
    transport = _recording_transport(seen)
    with ModelShim("http://provider.invalid/v1", KEY, transport=transport) as shim:
        response = _post(shim, body)

    assert response.status_code == 200
    assert seen == [body]
    assert capfd.readouterr().err == ""


def test_a_handler_that_fails_writes_one_fixed_line_and_none_of_the_request(
    monkeypatch, capfd
):
    marker = "MARKER-handler-failure"

    def fail(self, request, body):
        raise ValueError(f"{marker} {body!r}")

    monkeypatch.setattr(ModelShim, "_forward", fail)
    with ModelShim("http://provider.invalid/v1", KEY) as shim:
        with pytest.raises(httpx.HTTPError):
            _post(shim, marker.encode())

    err = capfd.readouterr().err
    assert marker not in err
    assert "Traceback" not in err
    assert err.splitlines() == ["model shim: a request failed (ValueError)"]
