"""A stand-in for OGX, started as ``python-ogx -m ogx.cli.ogx run <config> --insecure``.

It reads the port from the rendered config and serves ``POST /v1/responses``
on loopback. For each request it appends a line to ``$FAKE_OGX_RECORD``: the
body, its own PID, and which variables its environment holds. With
``FAKE_OGX_CALL_TOOL=1`` it calls ``lookup_order`` on the request's MCP server
as OGX would: over SSE, or, when the tool carries ``authorization``, over
streamable HTTP with ``Authorization: Bearer <authorization>``, listing the
tools first. It answers with response ID ``resp-<n>`` and reply ``reply <n>``.

``FAKE_OGX_MODE=silent`` runs without ever listening; ``exit`` exits at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

WATCHED = (
    "MIDOJO_SESSION_TOKEN",
    "MIDOJO_URL",
    "MIDOJO_TOOL_TOKEN",
    "MIDOJO_TOOL_URL",
    "OPENAI_API_KEY",
    "PATH",
    "TIKTOKEN_CACHE_DIR",
)


async def _call_tool(url: str) -> str:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("lookup_order", {"order_id": "ORD-101"})
            return result.content[0].text


async def _call_tool_with_token(url: str, token: str) -> str:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert "lookup_order" in [tool.name for tool in listed.tools]
            result = await session.call_tool("lookup_order", {"order_id": "ORD-101"})
            return result.content[0].text


def main() -> None:
    argv = sys.argv[1:]
    config_path = argv[argv.index("run") + 1]
    config = json.loads(open(config_path, encoding="utf-8").read())
    mode = os.environ.get("FAKE_OGX_MODE", "serve")
    print(f"fake-ogx: config {config_path} mode {mode}", flush=True)
    if mode == "exit":
        sys.exit(3)
    if mode == "silent":
        time.sleep(600)
        return

    record = os.environ.get("FAKE_OGX_RECORD")
    lock = Lock()
    count = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                count[0] += 1
                n = count[0]
            tool_result = None
            if os.environ.get("FAKE_OGX_CALL_TOOL") == "1":
                tool = body["tools"][0]
                if "authorization" in tool:
                    call = _call_tool_with_token(tool["server_url"], tool["authorization"])
                else:
                    call = _call_tool(tool["server_url"])
                tool_result = asyncio.run(call)
            if record:
                line = {
                    "path": self.path,
                    "body": body,
                    "pid": os.getpid(),
                    "env": {name: os.environ.get(name) for name in WATCHED},
                    "tool_result": tool_result,
                }
                with lock, open(record, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line) + "\n")
            print(f"fake-ogx: POST {self.path} request {n}", flush=True)
            response = {
                "id": f"resp-{n}",
                "object": "response",
                "status": "completed",
                "model": f"served/{body['model']}",
                "output": [
                    {"type": "mcp_list_tools", "server_label": "midojo", "tools": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"reply {n}"}],
                    },
                ],
            }
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer((config["server"]["host"], config["server"]["port"]), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
