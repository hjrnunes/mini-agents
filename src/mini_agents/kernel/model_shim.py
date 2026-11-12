"""A loopback proxy between the gateway and the model provider.

The gateway sends its model requests to the shim instead of the provider. The
shim holds the real key, rewrites the bodies some models need, and passes
everything else through byte for byte. The gateway never sees the key.

A model that refuses what OGX sends is listed in ``MODEL_REWRITES`` by the
provider's model id. ``gpt-6-sol`` refuses function tools unless a
``reasoning_effort`` is set, and refuses ``max_tokens``. The shim rewrites only
POST ``<base>/chat/completions`` bodies:

- ``reasoning_effort``, when the model has one, is set; an existing value is
  overwritten.
- ``token_param`` ``max_completion_tokens`` renames ``max_tokens`` or, when the
  body already has ``max_completion_tokens``, drops ``max_tokens``.

A body that is not a JSON object, another route, and a body the model's rules
leave alone are forwarded unchanged. A rewritten body is serialized compactly.

``MINI_AGENTS_REWRITE_LOG`` names a file that gets one JSON line per chat
completion: the route, the status, the model the request named and the
rewrites made, in order. It holds no header, key or body.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
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


def rewrites_for(model: str | None) -> Rewrites:
    """Return the rewrites of ``model``; none for an unlisted or absent one."""

    return MODEL_REWRITES.get(model, Rewrites()) if model else Rewrites()


def rewritten(body: bytes, rewrites: Rewrites) -> tuple[bytes, list[dict[str, Any]]]:
    """Return the body with ``rewrites`` applied, and what changed, in order.

    A body that is not a JSON object, and one the rules leave alone, come back
    byte for byte.
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
    return json.dumps(payload, separators=(",", ":")).encode(), changes


def _object(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _requested_model(body: bytes) -> str | None:
    model = (_object(body) or {}).get("model")
    return model if isinstance(model, str) and model else None


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


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        """Say nothing: a request line holds the path and a query."""

    def _relay(self) -> None:
        self.server.shim.handle(self)  # type: ignore[attr-defined]

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _relay


class ModelShim:
    """Forward loopback requests to the provider, rewriting chat completions.

    ``upstream_base_url`` is the provider's base URL. Its path is kept, so a
    client that uses :attr:`url` as its base URL reaches the same routes. The
    shim replaces the client's ``Authorization`` header with ``api_key``, and
    passes the client's own through when ``api_key`` is None.
    """

    def __init__(
        self,
        upstream_base_url: str,
        api_key: str | None,
        rewrites: Rewrites = Rewrites(),
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
        self._client = httpx.Client(
            timeout=UPSTREAM_TIMEOUT, transport=transport, follow_redirects=False
        )
        self._server = ThreadingHTTPServer((LOOPBACK_HOST, 0), _Handler)
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
        self._server.shutdown()
        self._server.server_close()
        self._client.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> ModelShim:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def handle(self, request: BaseHTTPRequestHandler) -> None:
        path = request.path.split("?", 1)[0]
        inside = path == self._base_path or path.startswith(self._base_path + "/")
        if not inside:
            _reply(request, 404, "outside the shim's base path")
            return
        if "chunked" in request.headers.get("Transfer-Encoding", "").lower():
            _reply(request, 411, "a request body needs a Content-Length")
            return
        body = request.rfile.read(int(request.headers.get("Content-Length") or 0))
        route = (path[len(self._base_path) :] or "/").rstrip("/") or "/"
        completion = request.command == "POST" and route == CHAT_COMPLETIONS
        changes: list[dict[str, Any]] = []
        if completion:
            requested_model = _requested_model(body)
            body, changes = rewritten(body, self._rewrites)
        status, error = self._forward(request, body)
        if completion:
            self._record(route, status, requested_model, changes, error)
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
        request_model: str | None,
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
                "request_model": request_model,
                "rewrites": changes,
                "error": error,
            }
            with self._log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, sort_keys=True) + "\n")
