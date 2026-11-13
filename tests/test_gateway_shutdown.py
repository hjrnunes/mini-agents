"""The gateway's shutdown is bounded, whatever OGX does with SIGTERM.

OGX is a stand-in (``tests/fixtures/stub_ogx.py``) that holds its port as OGX
does, so these tests need no ``gateway`` extra. The gateway runs in its own
process with a short grace period, in a session of its own like a run's
service, and with its scratch directory under the test's ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from target_lifecycle import ROOT, Process, free_port, listening

STUB = ROOT / "tests" / "fixtures" / "stub_ogx.py"
GRACE = 1.0
# The grace, the shim's own stop and the stand-in's exit, with headroom.
BOUND = 10.0

# The gateway with the stand-in in place of OGX and a short grace period.
MAIN = """
import sys
from mini_agents.kernel import gateway

stub, ready, grace = sys.argv[1:4]
gateway.ogx_command = lambda config: [sys.executable, stub, str(config), ready]
gateway.TERMINATE_GRACE_SECONDS = float(grace)
sys.exit(gateway.main(sys.argv[4:]))
"""


class Running:
    """One gateway over a stand-in OGX, and what it needs checked after."""

    def __init__(self, process: Process, ready: Path, scratch: Path) -> None:
        self.process = process
        self.ready = ready
        self.scratch = scratch

    def wait_ready(self) -> dict:
        deadline = time.monotonic() + 60
        while not self.ready.exists():
            assert self.process.popen.poll() is None, "the gateway exited too early"
            assert time.monotonic() < deadline, "the stand-in OGX never became ready"
            time.sleep(0.1)
        return json.loads(self.ready.read_text())

    def signal_and_wait(self, signum: int) -> tuple[int, float]:
        """Send ``signum`` to the gateway; return its exit code and the time it took."""
        started = time.monotonic()
        self.process.popen.send_signal(signum)
        try:
            code = self.process.popen.wait(timeout=BOUND)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the gateway was still running {BOUND:.0f}s after the signal")
        return code, time.monotonic() - started

    def scratch_directories(self) -> list[Path]:
        return list(self.scratch.glob("mini-agents-gateway-*"))


@pytest.fixture
def start(tmp_path, request):
    """Start a gateway over a stand-in OGX in ``mode``; kill what is left at the end."""
    started: list[Process] = []

    def _start(mode: str) -> Running:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        ready = tmp_path / "ready.json"
        process = Process(
            [
                sys.executable,
                "-c",
                MAIN,
                str(STUB),
                str(ready),
                str(GRACE),
                "--connector",
                "stub",
                "--mcp-url",
                "http://127.0.0.1:9/sse",
                "--port",
                str(free_port()),
            ],
            env={
                "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
                "OPENAI_API_KEY": "sk-shutdown-test-key-0123456789abcdef",
                "MODEL_ID": "stub-model",
                "STUB_OGX_MODE": mode,
                "TMPDIR": str(scratch),
            },
            log=tmp_path / "gateway.log",
        )
        started.append(process)
        return Running(process, ready, scratch)

    request.addfinalizer(lambda: [process.kill() for process in started])
    return _start


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _shim_port(state: dict) -> int:
    port = urlsplit(state["shim_url"]).port
    assert port is not None
    return port


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_an_ogx_that_ignores_sigterm_is_killed_and_everything_is_cleaned_up(
    start, tmp_path, signum
):
    gateway = start("ignore-sigterm")
    state = gateway.wait_ready()
    shim_port = _shim_port(state)
    # Something to clean up: OGX and the shim listen, the scratch exists.
    assert listening(state["port"]) and listening(shim_port)
    assert len(gateway.scratch_directories()) == 1
    assert _alive(state["pid"])

    code, took = gateway.signal_and_wait(signum)

    assert code == 0
    assert took < BOUND
    assert "killing it" in (tmp_path / "gateway.log").read_text()
    assert not _alive(state["pid"])
    assert not listening(state["port"])
    assert not listening(shim_port)
    assert gateway.process.group_is_gone()
    assert gateway.scratch_directories() == []


def test_an_ogx_that_exits_on_sigterm_is_not_killed_and_the_gateway_is_quick(
    start, tmp_path
):
    gateway = start("exit-on-sigterm")
    state = gateway.wait_ready()

    code, took = gateway.signal_and_wait(signal.SIGTERM)

    assert code == 0
    assert took < GRACE
    assert "killing it" not in (tmp_path / "gateway.log").read_text()
    assert not _alive(state["pid"])
    assert not listening(state["port"])
    assert gateway.process.group_is_gone()
    assert gateway.scratch_directories() == []


def test_an_ogx_that_exits_by_itself_gives_the_gateway_its_exit_code(start):
    gateway = start("exit:3")
    state = gateway.wait_ready()

    code = gateway.process.popen.wait(timeout=BOUND)

    assert code == 3
    assert not listening(_shim_port(state))
    assert gateway.process.group_is_gone()
    assert gateway.scratch_directories() == []
