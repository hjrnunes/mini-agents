"""A loopback proxy between the gateway and the model provider.

The gateway sends its model requests to the shim instead of the provider. The
shim holds the real key, rewrites the bodies some models need, and passes
everything else through byte for byte. The gateway never sees the key.

A model that refuses what OGX sends is listed in ``MODEL_REWRITES`` by the
provider's model id. ``gpt-6-sol`` refuses function tools unless a
``reasoning_effort`` is set, and refuses ``max_tokens``. The shim rewrites only
POST ``<base>/chat/completions`` bodies, by the ``model`` each body names:

- ``reasoning_effort``, when the model has one, is set; an existing value is
  overwritten.
- ``token_param`` ``max_completion_tokens`` renames ``max_tokens`` or, when the
  body already has ``max_completion_tokens``, drops ``max_tokens``.

A body that names another model or none, a body that is not a JSON object (or
is nested too deeply to read), another route, and a body the model's rules
leave alone are forwarded unchanged. A rewritten body is serialized compactly.

``MINI_AGENTS_REWRITE_LOG`` names a file that gets one JSON line per chat
completion: the route, the status, the rewrites made, in order, and the error
class when the provider was not reached. Everything in a row comes from the
shim, never from a request: no header, key or body, no model name the request
chose, and no value a rewrite replaced. A row is written after its response
ends, so a reader that needs every row reads after ``ModelShim.stop``, which
waits for the requests in flight, for at most ``DRAIN_SECONDS``.

A request the shim cannot serve gets a fixed error and is not forwarded: a
request target or header that is not plain ASCII, a ``Content-Length`` that is
not one run of digits (or is repeated), a body above ``MAX_BODY_BYTES`` or
shorter than its ``Content-Length``, and a method or request line the server
does not read. The error text never holds a value the request chose, and
neither does stderr: the shim logs no request, and a handler that fails writes
one line that names the exception class and nothing else.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

LOOPBACK_HOST = "127.0.0.1"
REWRITE_LOG_ENV = "MINI_AGENTS_REWRITE_LOG"
REASONING_EFFORT = "reasoning_effort"
MAX_TOKENS = "max_tokens"
MAX_COMPLETION_TOKENS = "max_completion_tokens"
CHAT_COMPLETIONS = "/chat/completions"
# A reasoning model can think for minutes before the first byte; the caller's
# own timeout fires first.
UPSTREAM_TIMEOUT = httpx.Timeout(700.0, connect=30.0)
# How long ``stop`` waits for the requests in flight. A request that is still
# open then waits on the provider, not on a client that is about to stop.
DRAIN_SECONDS = 5.0
# The largest request body the shim reads. A chat completion with a long
# history is a few megabytes; a ``Content-Length`` far above that is not a
# request to serve.
MAX_BODY_BYTES = 64 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
# A request target is visible ASCII; a header name is an HTTP token and a
# value is printable ASCII or a tab. What else a client sends is refused before
# anything parses it (the HTTP client turns it into an exception that names it).
_TARGET = re.compile(r"[\x21-\x7e]+")
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_HEADER_VALUE = re.compile(r"[\x20-\x7e\t]*")

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_REQUEST_ONLY = _HOP_BY_HOP | {"host", "authorization", "content-length"}
_RESPONSE_ONLY = _HOP_BY_HOP | {"server", "date"}


@dataclass(frozen=True)
class Rewrites:
    """What one model needs changed in a chat completion request."""

    reasoning_effort: str | None = None
    token_param: str | None = None

    @property
    def active(self) -> bool:
        return self.reasoning_effort is not None or (
            self.token_param == MAX_COMPLETION_TOKENS
        )


# By the model id the gateway sends upstream. A model that is not listed gets
# no rewrite.
MODEL_REWRITES: Mapping[str, Rewrites] = {
    "gpt-6-sol": Rewrites(reasoning_effort="none", token_param=MAX_COMPLETION_TOKENS),
}


def rewrites_for(
    model: str | None, table: Mapping[str, Rewrites] = MODEL_REWRITES
) -> Rewrites:
    """Return the rewrites of ``model``; none for an unlisted or absent one."""

    return table.get(model, Rewrites()) if model else Rewrites()


def rewritten(body: bytes, rewrites: Rewrites) -> tuple[bytes, list[dict[str, Any]]]:
    """Return the body with ``rewrites`` applied, and what changed, in order.

    A body that is not a JSON object, one nested too deeply to read or write
    again, and one the rules leave alone, come back byte for byte.
    """

    if not rewrites.active:
        return body, []
    payload = _object(body)
    if payload is None:
        return body, []
    changes: list[dict[str, Any]] = []
    if rewrites.reasoning_effort is not None:
        if REASONING_EFFORT in payload:
            changes.append(
                {
                    "op": "overwrite",
                    "field": REASONING_EFFORT,
                    "was": payload[REASONING_EFFORT],
                    "value": rewrites.reasoning_effort,
                }
            )
        else:
            changes.append(
                {
                    "op": "inject",
                    "field": REASONING_EFFORT,
                    "value": rewrites.reasoning_effort,
                }
            )
        payload[REASONING_EFFORT] = rewrites.reasoning_effort
    if rewrites.token_param == MAX_COMPLETION_TOKENS:
        if MAX_TOKENS in payload and MAX_COMPLETION_TOKENS in payload:
            del payload[MAX_TOKENS]
            changes.append({"op": "drop", "field": MAX_TOKENS})
        elif MAX_TOKENS in payload:
            payload[MAX_COMPLETION_TOKENS] = payload.pop(MAX_TOKENS)
            changes.append(
                {"op": "rename", "from": MAX_TOKENS, "to": MAX_COMPLETION_TOKENS}
            )
    try:
        return json.dumps(payload, separators=(",", ":")).encode(), changes
    except RecursionError:
        return body, []


def _object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _requested_model(body: bytes) -> str | None:
    model = (_object(body) or {}).get("model")
    return model if isinstance(model, str) and model else None


def _loggable(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``changes`` without the values a request supplied.

    An overwrite names the value it replaced (``was``), which the client chose
    and may be any string. The rest comes from the rule table.
    """

    return [{k: v for k, v in change.items() if k != "was"} for change in changes]


def _reply(request: BaseHTTPRequestHandler, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode()
    try:
        request.send_response(status)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(body)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(body)
    except OSError:
        pass
    request.close_connection = True


class _Refused(Exception):
    """A request the shim does not serve, and the fixed reply it gets."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _read_body(request: BaseHTTPRequestHandler) -> bytes:
    """Read the body ``Content-Length`` declares; raise ``_Refused`` if it cannot.

    A missing header means no body. Anything but one run of ASCII digits is
    refused: ``int`` also takes signs, underscores and surrounding blanks.
    """

    declared = request.headers.get_all("Content-Length", [])
    if not declared:
        return b""
    text = declared[0]
    if len(declared) > 1 or not (text.isascii() and text.isdigit()):
        raise _Refused(400, "invalid Content-Length")
    # Beyond 18 digits the value is over the cap, and ``int`` need not read it.
    if len(text.lstrip("0")) > 18 or int(text) > MAX_BODY_BYTES:
        raise _Refused(413, "the body is larger than the shim accepts")
    remaining = int(text)
    chunks: list[bytes] = []
    while remaining:
        try:
            chunk = request.rfile.read(min(remaining, _READ_CHUNK))
        except OSError:
            chunk = b""
        if not chunk:
            raise _Refused(400, "the body is shorter than its Content-Length")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        """Say nothing: a request line holds the path and a query."""

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        """Reply with the status's own phrase.

        The server's reply to a request line it cannot read repeats that line.
        """
        try:
            phrase = HTTPStatus(code).phrase
        except ValueError:
            phrase = "error"
        _reply(self, code, phrase)

    def _relay(self) -> None:
        self.server.shim.handle(self)  # type: ignore[attr-defined]

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _relay


class _Server(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        """Name the exception class of a failed handler, and nothing else.

        The server's own report prints the traceback, whose last line holds the
        exception's message, and that message can hold what the client sent.
        """
        error = sys.exc_info()[1]
        name = type(error).__name__
        print(f"model shim: a request failed ({name})", file=sys.stderr, flush=True)


class ModelShim:
    """Forward loopback requests to the provider, rewriting chat completions.

    ``upstream_base_url`` is the provider's base URL. Its path is kept, so a
    client that uses :attr:`url` as its base URL reaches the same routes. The
    shim replaces the client's ``Authorization`` header with ``api_key``, and
    passes the client's own through when ``api_key`` is None. ``rewrites``
    maps a model id to the rewrites of the chat completions that name it.
    :meth:`stop` waits up to ``DRAIN_SECONDS`` for the requests in flight.
    """

    def __init__(
        self,
        upstream_base_url: str,
        api_key: str | None,
        rewrites: Mapping[str, Rewrites] = MODEL_REWRITES,
        *,
        log_path: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parts = urlsplit(upstream_base_url)
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._base_path = parts.path.rstrip("/")
        self._api_key = api_key
        self._rewrites = rewrites
        self._log_path = log_path
        self._log_lock = threading.Lock()
        self._sequence = 0
        self._idle = threading.Condition()
        self._in_flight = 0
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._client = httpx.Client(
            timeout=UPSTREAM_TIMEOUT, transport=transport, follow_redirects=False
        )
        self._server = _Server((LOOPBACK_HOST, 0), _Handler)
        self._server.daemon_threads = True
        self._server.shim = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK_HOST}:{self.port}{self._base_path}"

    def start(self) -> ModelShim:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="model-shim", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving, once the requests in flight end or the drain runs out.

        A client has the end of a response before the shim writes that
        request's log row. Waiting for the requests the shim has begun to
        serve means a reader that reads after ``stop`` sees every row of a
        response a client received. A request still open when the drain ends
        is abandoned, and its row may be missing. A second call does nothing.
        """

        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self._server.shutdown()
            with self._idle:
                self._idle.wait_for(lambda: self._in_flight == 0, timeout=DRAIN_SECONDS)
            self._server.server_close()
            self._client.close()
            if self._thread is not None:
                self._thread.join(timeout=5)

    def __enter__(self) -> ModelShim:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        with self._idle:
            self._in_flight += 1
        try:
            self._serve(request)
        finally:
            with self._idle:
                self._in_flight -= 1
                self._idle.notify_all()

    def _accept(self, request: BaseHTTPRequestHandler) -> tuple[str, bytes]:
        """Return the path and the body of a request the shim serves.

        Raise ``_Refused`` for any other. Nothing here reads a value into an
        error: each refusal has its own fixed text.
        """
        if not _TARGET.fullmatch(request.path):
            raise _Refused(400, "malformed request target")
        for name, value in request.headers.items():
            if not (_HEADER_NAME.fullmatch(name) and _HEADER_VALUE.fullmatch(value)):
                raise _Refused(400, "malformed header")
        path = request.path.split("?", 1)[0]
        inside = path == self._base_path or path.startswith(self._base_path + "/")
        if not inside:
            raise _Refused(404, "outside the shim's base path")
        if "chunked" in request.headers.get("Transfer-Encoding", "").lower():
            raise _Refused(411, "a request body needs a Content-Length")
        return path, _read_body(request)

    def _serve(self, request: BaseHTTPRequestHandler) -> None:
        try:
            path, body = self._accept(request)
        except _Refused as refusal:
            _reply(request, refusal.status, refusal.message)
            return
        route = (path[len(self._base_path) :] or "/").rstrip("/") or "/"
        completion = request.command == "POST" and route == CHAT_COMPLETIONS
        changes: list[dict[str, Any]] = []
        if completion:
            rules = rewrites_for(_requested_model(body), self._rewrites)
            body, changes = rewritten(body, rules)
        status, error = self._forward(request, body)
        if completion:
            self._record(route, status, _loggable(changes), error)
        if status == 0:
            _reply(request, 502, "model provider unreachable")
        request.close_connection = True

    def _headers(self, request: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _REQUEST_ONLY | {"accept-encoding"}
        }
        if self._api_key is None:
            if "Authorization" in request.headers:
                headers["Authorization"] = request.headers["Authorization"]
        else:
            headers["Authorization"] = f"Bearer {self._api_key}"
        headers["Accept-Encoding"] = "identity"
        return headers

    def _forward(
        self, request: BaseHTTPRequestHandler, body: bytes
    ) -> tuple[int, str | None]:
        """Send the request upstream and stream the answer back.

        Return the upstream status (0 when none arrived, and nothing was sent
        back) and the error class name, if any.
        """

        status = 0
        try:
            with self._client.stream(
                request.command,
                self._origin + request.path,
                headers=self._headers(request),
                content=body or None,
            ) as upstream:
                status = upstream.status_code
                request.send_response(status)
                for name, value in upstream.headers.items():
                    if name.lower() not in _RESPONSE_ONLY:
                        request.send_header(name, value)
                request.send_header("Connection", "close")
                request.end_headers()
                for chunk in upstream.iter_raw():
                    request.wfile.write(chunk)
                    request.wfile.flush()
        except (httpx.HTTPError, OSError, RuntimeError) as error:
            return status, type(error).__name__
        return status, None

    def _record(
        self,
        route: str,
        status: int,
        changes: list[dict[str, Any]],
        error: str | None,
    ) -> None:
        if self._log_path is None:
            return
        with self._log_lock:
            self._sequence += 1
            row = {
                "sequence": self._sequence,
                "route": route,
                "status": status,
                "rewrites": changes,
                "error": error,
            }
            with self._log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, sort_keys=True) + "\n")
