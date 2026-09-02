"""Start every mini-agent MCP server, then OGX.

Same shape as MiniBank's Docker entrypoint: tools on SSE ports, Responses
API on 8321. One process tree, one command.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

MCP_SERVERS = (
    ("klarna", 8888, False),
    ("klarna", 8889, True),
    ("airbnb", 8890, False),
    ("airbnb", 8891, True),
    ("occiai", 8892, False),
    ("occiai", 8893, True),
)


def ogx_config_path() -> Path:
    override = os.environ.get("MINI_AGENTS_OGX_CONFIG")
    if override:
        return Path(override)
    start = Path(__file__).resolve().parent
    walked = [directory / "ogx-config.yaml" for directory in (start, *start.parents)]
    candidates = [
        *walked,
        Path("/opt/app-root/src/ogx-config.yaml"),
        Path.cwd() / "ogx-config.yaml",
        Path.cwd() / "mini-agents" / "ogx-config.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "ogx-config.yaml not found. Set MINI_AGENTS_OGX_CONFIG."
    )


def _wait_for_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"MCP server on port {port} did not become ready")


def start_mcp_servers() -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for domain, port, unsafe in MCP_SERVERS:
        cmd = [
            sys.executable,
            "-m",
            "mini_agents",
            "--domain",
            domain,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
        if unsafe:
            cmd.append("--unsafe")
        print(f"Starting {domain} {'unsafe' if unsafe else 'safe'} on :{port}")
        procs.append(subprocess.Popen(cmd))
    for _, port, _ in MCP_SERVERS:
        _wait_for_port(port)
    return procs


def _ogx_command(config: Path) -> list[str]:
    ogx = shutil.which("ogx")
    if ogx:
        return [ogx, "run", str(config)]
    return [sys.executable, "-m", "ogx", "run", str(config)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start MiniKlarna/Airbnb/OcciAI MCP servers and OGX."
    )
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Start MCP servers and wait; do not start OGX.",
    )
    args = parser.parse_args()

    procs = start_mcp_servers()

    def _stop(_signum=None, _frame=None) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if _signum is not None:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if args.mcp_only:
        print("MCP servers ready. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
                for proc in procs:
                    if proc.poll() is not None:
                        raise SystemExit(f"MCP process exited with {proc.returncode}")
        finally:
            _stop()

    config = ogx_config_path()
    cmd = _ogx_command(config)
    print(f"Starting OGX: {' '.join(cmd)}")
    print("Responses API → http://0.0.0.0:8321/v1")
    try:
        ogx_proc = subprocess.Popen(cmd)
    except FileNotFoundError as exc:
        _stop()
        raise SystemExit(
            "ogx not found. Install with `uv pip install 'ogx[starter]'` "
            "or use docker compose in mini-agents/."
        ) from exc
    procs.append(ogx_proc)
    try:
        raise SystemExit(ogx_proc.wait())
    finally:
        _stop()


if __name__ == "__main__":
    main()
