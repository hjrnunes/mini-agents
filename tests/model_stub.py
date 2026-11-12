"""A stand-in model provider for the gateway tests: no network, no real model.

``StubModel`` serves an OpenAI-compatible ``/v1`` on loopback and records every
request it receives (method, path, headers, raw body). ``GET /v1/models`` lists
its model. ``POST /v1/chat/completions`` answers a canned completion, streamed
as server-sent events when the body asks for a stream. With a ``tool_call``, a
request that offers tools and holds no tool result yet gets that call, so one
agent turn takes two completions: the call, then the text.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ANSWER = "Done."


@dataclass
class Received:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class StubModel:
    model: str = "stub-model"
    # (tool name, arguments) the first completion of a turn calls.
    tool_call: tuple[str, dict[str, Any]] | None = None
    # A streamed answer waits for this event after its first chunk.
    pause: threading.Event | None = None
    # Models `GET /v1/models` lists before its own, as a provider with many does.
    also_listed: tuple[str, ...] = ()
    requests: list[Received] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.stub = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> StubModel:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.pause is not None:
            self.pause.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def completions(self) -> list[Received]:
        return [r for r in self.requests if _is_completion(r.path)]


def _is_completion(path: str) -> bool:
    return path.split("?")[0].rstrip("/").endswith("/chat/completions")


def _chunk(
    model: str, delta: dict[str, Any], finish: str | None, **extra: Any
) -> bytes:
    body = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> Received:
        length = int(self.headers.get("Content-Length") or 0)
        received = Received(
            self.command,
            self.path,
            {name.lower(): value for name, value in self.headers.items()},
            self.rfile.read(length),
        )
        self.server.stub.requests.append(received)  # type: ignore[attr-defined]
        return received

    def do_GET(self) -> None:
        self._read()
        stub: StubModel = self.server.stub  # type: ignore[attr-defined]
        if self.path.split("?")[0].endswith("/models"):
            listing = {
                "object": "list",
                "data": [
                    {"id": name, "object": "model"}
                    for name in (*stub.also_listed, stub.model)
                ],
            }
            self._send(200, json.dumps(listing).encode(), "application/json")
        else:
            self._send(404, b'{"error": "not found"}', "application/json")

    def do_POST(self) -> None:
        received = self._read()
        stub: StubModel = self.server.stub  # type: ignore[attr-defined]
        if not _is_completion(received.path):
            self._send(404, b'{"error": "not found"}', "application/json")
            return
        try:
            request = received.json()
        except ValueError:
            self._send(400, b'{"error": "not JSON"}', "application/json")
            return
        calling = (
            stub.tool_call is not None
            and bool(request.get("tools"))
            and not any(m.get("role") == "tool" for m in request.get("messages", []))
        )
        delta = (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_stub",
                        "type": "function",
                        "function": {
                            "name": stub.tool_call[0],  # type: ignore[index]
                            "arguments": json.dumps(stub.tool_call[1]),  # type: ignore[index]
                        },
                    }
                ],
            }
            if calling
            else {"role": "assistant", "content": ANSWER}
        )
        finish = "tool_calls" if calling else "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_chunk(stub.model, delta, None))
            self.wfile.flush()
            if stub.pause is not None:
                stub.pause.wait(timeout=30)
            self.wfile.write(_chunk(stub.model, {}, finish, usage=usage))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        message = {k: v for k, v in delta.items() if k != "index"}
        if "tool_calls" in message:
            message["tool_calls"] = [
                {k: v for k, v in call.items() if k != "index"}
                for call in message["tool_calls"]
            ]
        completion = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": stub.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        }
        self._send(200, json.dumps(completion).encode(), "application/json")
