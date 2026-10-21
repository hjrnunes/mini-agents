"""A stand-in for the MCP server, started as ``python-mcp -m mini_agents ... --port N``.

It accepts and closes TCP connections on the port, enough for the agent
command's listen check. ``FAKE_MCP_MODE=exit`` exits at once; ``silent`` runs
without listening.
"""

from __future__ import annotations

import os
import socket
import sys
import time


def main() -> None:
    argv = sys.argv[1:]
    port = int(argv[argv.index("--port") + 1])
    host = argv[argv.index("--host") + 1]
    mode = os.environ.get("FAKE_MCP_MODE", "serve")
    print(f"fake-mcp: {host}:{port} mode {mode}", flush=True)
    if mode == "exit":
        sys.exit(4)
    if mode == "silent":
        time.sleep(600)
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        while True:
            connection, _ = server.accept()
            connection.close()


if __name__ == "__main__":
    main()
