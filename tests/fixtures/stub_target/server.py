"""A tiny target that is not a mini-agents domain, for the lifecycle tests.

    python server.py --port PORT       serve until SIGTERM
    python server.py --probe PORT      exit 0 once the server on PORT is healthy
    python server.py --reset PORT      return the server on PORT to its start state

The server keeps one counter in memory. ``GET /health`` answers ``ok``,
``GET /state`` the counter, ``POST /bump`` adds one and ``POST /reset`` sets it
to zero. Its config (``target.yaml``) names a ``health`` and a ``reset``
command, which a mini-agents target does not, and no gateway.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(b"ok")
        elif self.path == "/state":
            self._send(json.dumps({"counter": self.server.counter}).encode())  # type: ignore[attr-defined]
        else:
            self._send(b"not found", 404)

    def do_POST(self) -> None:
        with self.server.lock:  # type: ignore[attr-defined]
            if self.path == "/bump":
                self.server.counter += 1  # type: ignore[attr-defined]
            elif self.path == "/reset":
                self.server.counter = 0  # type: ignore[attr-defined]
            else:
                self._send(b"not found", 404)
                return
        self._send(json.dumps({"counter": self.server.counter}).encode())  # type: ignore[attr-defined]


def _call(port: int, method: str, path: str) -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return int(response.status)
    except (urllib.error.URLError, OSError):
        return 0


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    server.counter = 0  # type: ignore[attr-defined]
    server.lock = threading.Lock()  # type: ignore[attr-defined]
    signal.signal(
        signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start()
    )
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", type=int)
    group.add_argument("--probe", type=int)
    group.add_argument("--reset", type=int)
    args = parser.parse_args(argv)
    if args.port is not None:
        serve(args.port)
        return 0
    if args.probe is not None:
        return 0 if _call(args.probe, "GET", "/health") == 200 else 1
    return 0 if _call(args.reset, "POST", "/reset") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
