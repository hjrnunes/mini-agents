"""Start the servers detached, wait until they listen, check they still run.

Each server starts in its own session (``setsid``) with stdin from
``/dev/null`` and stdout and stderr to a log file: with open stdio, the
sandbox's exec waits for the child and hangs. A detached child outlives the
command and is re-parented, so a later turn has only its PID; the image has
no ``ps``, so liveness reads ``/proc/<pid>/stat``, and falls back to
``kill(pid, 0)`` where there is no ``/proc`` (macOS, in tests). A PID alone
can be reused, so the first turn also records the process's start time and a
later turn requires the same one.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mini_agents.sandbox_agent.failures import (
    START_FAILED,
    AgentFailure,
    exited,
    not_listening,
)
from mini_agents.sandbox_agent.ogx_config import LOOPBACK_HOST

PROC_ROOT = Path("/proc")
POLL_SECONDS = 0.2
STOP_SECONDS = 5.0

Liveness = Callable[[int], bool]
Identity = Callable[[int], str | None]


def mcp_command(python: Path, *, domain: str, port: int) -> list[str]:
    """The domain's safe server, on loopback."""

    return [
        str(python),
        "-m",
        "mini_agents",
        "--domain",
        domain,
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(port),
    ]


def ogx_command(python: Path, config: Path) -> list[str]:
    # `ogx` ships no `__main__.py`; the module that does is `ogx.cli.ogx`.
    return [str(python), "-m", "ogx.cli.ogx", "run", str(config), "--insecure"]


def free_ports(count: int) -> list[int]:
    """``count`` distinct loopback ports that are free right now."""

    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((LOOPBACK_HOST, 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def start_detached(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log: Path,
    cwd: Path,
    name: str,
) -> subprocess.Popen:
    """Start ``command`` in its own session with stdio to ``log``."""

    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("ab") as output:
            return subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=dict(env),
                cwd=cwd,
                start_new_session=True,
                close_fds=True,
            )
    except OSError as exc:
        raise AgentFailure(START_FAILED, f"{name}: {command[0]}: {exc}") from exc


def listening(port: int, host: str = LOOPBACK_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def wait_listening(
    process: subprocess.Popen,
    *,
    port: int,
    name: str,
    deadline: float,
    log: Path,
) -> None:
    """Return once ``process`` listens on ``port``; fail if it exits or the
    deadline (``time.monotonic()``) passes first."""

    while True:
        code = process.poll()
        if code is not None:
            raise AgentFailure(
                exited(name),
                f"pid {process.pid} exited with {code} before listening on "
                f"{LOOPBACK_HOST}:{port}; see {log}",
            )
        if listening(port):
            return
        if time.monotonic() >= deadline:
            raise AgentFailure(
                not_listening(name),
                f"pid {process.pid} did not listen on {LOOPBACK_HOST}:{port} "
                f"in time; see {log}",
            )
        time.sleep(POLL_SECONDS)


def pid_alive(pid: int, proc_root: Path = PROC_ROOT) -> bool:
    """Whether ``pid`` runs, read from ``/proc``; a zombie does not run."""

    if proc_root.is_dir():
        try:
            stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            return False
        # pid (comm) state ...; comm may hold spaces and parentheses.
        fields = stat.rpartition(")")[2].split()
        return bool(fields) and fields[0] not in ("Z", "X")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_identity(pid: int, proc_root: Path = PROC_ROOT) -> str | None:
    """What tells the process holding ``pid`` from a later one reusing it.

    With ``/proc``, the start time in clock ticks since boot (field 22 of
    ``/proc/<pid>/stat``); without, the start time ``ps`` reports (macOS, in
    tests). ``None`` when neither can be read, e.g. the process is gone.
    """

    if proc_root.is_dir():
        try:
            stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # Field 3 (state) is the first after the comm; field 22 is 19 later.
        fields = stat.rpartition(")")[2].split()
        return f"proc-start:{fields[19]}" if len(fields) > 19 else None
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    started = result.stdout.strip()
    return f"ps-lstart:{started}" if result.returncode == 0 and started else None


def stop(process: subprocess.Popen, *, timeout: float = STOP_SECONDS) -> None:
    """Stop a server this run started, with its whole session: TERM, then KILL.

    Used when the first turn fails; a later turn never stops a server.
    """

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
