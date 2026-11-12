"""Run a target from its config alone: start, reset, stop.

A stand-in for the testing tool that does this in a run. It reads a
``target.yaml`` (JSON under ``#`` comment lines, which needs no YAML library),
fills the placeholders its commands hold, runs those commands from the
repository checkout, and knows nothing else about the target. The tests use it
for the six mini-agents targets and for the fixture target in
``tests/fixtures/stub_target``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
AUTHORIZED = ("mock", "test_instance")


def read_config(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    return json.loads("\n".join(line for line in lines if not line.startswith("#")))


def authorize(config: Mapping[str, Any]) -> None:
    """Refuse a target that does not say it is a mock or a test instance."""
    if config.get("authorized_as") not in AUTHORIZED:
        raise PermissionError(f"target {config.get('target')!r} is not authorized")


def fill(parts: Sequence[str], values: Mapping[str, object]) -> list[str]:
    """Fill every ``{name}``; a name with no value raises ``KeyError``."""
    return [PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), part) for part in parts]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class Process:
    """One started command, in a session of its own like a run's service."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        log: Path | None = None,
    ) -> None:
        output = log.open("wb") if log is not None else subprocess.DEVNULL
        try:
            self.popen = subprocess.Popen(
                list(command),
                cwd=ROOT,
                env={**os.environ, **(env or {})},
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            if log is not None:
                output.close()  # type: ignore[union-attr]

    @property
    def pid(self) -> int:
        return self.popen.pid

    def wait_serving(self, port: int, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.popen.poll() is None, (
                f"exited with {self.popen.returncode} before serving :{port}"
            )
            if listening(port):
                return
            time.sleep(0.2)
        self.kill()
        raise AssertionError(f"nothing listened on :{port} within {timeout:.0f}s")

    def group(self) -> list[int]:
        """Return the pids of the process group the command started."""
        result = subprocess.run(
            ["pgrep", "-g", str(self.pid)], capture_output=True, text=True, check=False
        )
        return [int(pid) for pid in result.stdout.split()]

    def stop(self, timeout: float = 30.0) -> int:
        """Send SIGTERM and wait; the process must exit on it."""
        if self.popen.poll() is None:
            self.popen.send_signal(signal.SIGTERM)
        try:
            code = self.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise AssertionError(f"pid {self.pid} ignored SIGTERM for {timeout:.0f}s")
        return code

    def kill(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.popen.wait()

    def group_is_gone(self, timeout: float = 15.0) -> bool:
        """Whether no process of the command's group is left."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(self.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False


def python_values(**values: object) -> dict[str, object]:
    return {"python": sys.executable, **values}


def start_target(
    config: Mapping[str, Any], *, stage: str | None = None, log: Path | None = None
) -> tuple[Process, int]:
    """Start the target the way a run does; return its process and port.

    ``stage`` names a ``start_stage`` whose extra arguments the run adds.
    """
    authorize(config)
    endpoint = config["endpoint"]
    port = free_port()
    parts = [*endpoint["start"], *endpoint.get("start_stage", {}).get(stage, [])]
    process = Process(fill(parts, python_values(port=port)), log=log)
    process.wait_serving(port, timeout=60)
    return process, port


def start_gateway(
    config: Mapping[str, Any],
    *,
    mcp_url: str,
    model: str,
    model_url: str,
    model_key: str,
    log: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[Process, int]:
    """Start the gateway the config names, with the model profile filled in."""
    authorize(config)
    gateway = config["gateway"]
    port = free_port()
    values = python_values(
        mcp_url=mcp_url,
        gateway_port=port,
        model=model,
        model_url=model_url,
        model_key=model_key,
    )
    command = [*fill(gateway["start"], values), *extra_args]
    environment = dict(zip(gateway["env"], fill(list(gateway["env"].values()), values)))
    process = Process(command, env={**environment, **(extra_env or {})}, log=log)
    process.wait_serving(port, timeout=180)
    return process, port
