"""A fresh start is the reset: the same seed file gives the same initial ledger.

A server keeps its ledger in memory only and plants a seed only when it starts,
so a caller resets a target by stopping it and starting it again with the same
arguments. The in-process tests drive ``main()`` with a builder that keeps the
state it is given, so no port is bound. The last test restarts a real server
on loopback and reads its ledger over SSE. No model calls.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mini_agents.kernel import mcp_server
from mini_agents.kernel.seeds import ARTIFACT_SEED_VERSION, SEED_SCHEMA_VERSION

DOMAINS = mcp_server.DOMAINS
STATE_CLASS = {"klarna": "KlarnaState", "airbnb": "AirbnbState", "occiai": "OcciAIState"}
WRITES = {
    "klarna": ("escalate_to_human", {"topic": "fees", "reason": "r"}),
    "airbnb": (
        "escalate_trust_safety",
        {"reservation_id": "RES-101", "topic": "scam", "reason": "r"},
    ),
    "occiai": (
        "submit_questionnaire",
        {"patient_id": "PAT-104", "answers": {"symptoms": "blurred vision"}},
    ),
}
PLANTED = "Planted for the reset test."


class FakeServer:
    def __init__(self):
        self.settings = SimpleNamespace(host=None, port=None)

    def run(self, transport):
        pass


def _module(domain, name):
    return importlib.import_module(f"mini_agents.domains.{domain}.{name}")


def _ledger(state):
    """Every record; each start draws its own session id."""
    environment = state.to_environment()
    environment.pop("session_id")
    return environment


def _seed_file(tmp_path, domain, version):
    slot = _module(domain, "seeds").SLOTS[0]
    document = {
        "schema_version": version,
        "carrier_operation": slot.carrier_operation,
        "items": [
            {
                "slot": slot.id,
                "record_ref": slot.records[0],
                "mode": "replace",
                "content": PLANTED,
            }
        ],
    }
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(document))
    return path


def _start(monkeypatch, domain, *args):
    """Run ``main()`` once and return the state the server was built with."""
    states = []
    monkeypatch.setitem(
        mcp_server.BUILDERS, domain, lambda state=None: states.append(state) or FakeServer()
    )
    mcp_server.main(["--domain", domain, "--port", "1", *args])
    (state,) = states
    return state


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_snapshot_holds_every_state_attribute(domain):
    state = getattr(_module(domain, "state"), STATE_CLASS[domain])()

    assert set(vars(state)) == set(type(state).ENVIRONMENT_ATTRIBUTES)


@pytest.mark.parametrize("version", [ARTIFACT_SEED_VERSION, SEED_SCHEMA_VERSION])
@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_a_fresh_start_with_the_same_seed_file_gives_the_same_initial_ledger(
    tmp_path, monkeypatch, domain, unsafe, version
):
    path = _seed_file(tmp_path, domain, version)
    flags = ["--seed-file", str(path), *(["--unsafe"] if unsafe else [])]

    first = _start(monkeypatch, domain, *flags)
    initial = _ledger(first)
    tools = _module(domain, "tools_unsafe" if unsafe else "tools")
    executor = next(v for k, v in vars(tools).items() if k.endswith("ToolExecutor"))
    tool, arguments = WRITES[domain]
    assert "error" not in executor(first).execute(tool, arguments)
    assert _ledger(first) != initial

    second = _start(monkeypatch, domain, *flags)

    assert second is not first
    assert _ledger(second) == initial
    assert PLANTED in json.dumps(initial)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(port, seed_file):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mini_agents",
            "--domain",
            "klarna",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--seed-file",
            str(seed_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        assert process.poll() is None, "the server exited before it listened"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return process
        time.sleep(0.1)
    process.kill()
    raise AssertionError("the server did not listen in time")


async def _calls(url, calls):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    results = []
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for name, arguments in calls:
                result = await session.call_tool(name, arguments)
                results.append(json.loads(result.content[0].text))
    return results


def _summary_without_session(summary):
    return {key: value for key, value in summary.items() if key != "session_id"}


def test_restarting_a_real_server_resets_its_ledger(tmp_path):
    seed_file = _seed_file(tmp_path, "klarna", ARTIFACT_SEED_VERSION)
    summary = ("get_klarna_state_summary", {})
    tool, arguments = WRITES["klarna"]

    port = _free_port()
    process = _serve(port, seed_file)
    try:
        before, written, after = asyncio.run(
            _calls(f"http://127.0.0.1:{port}/sse", [summary, (tool, arguments), summary])
        )
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert "error" not in written
    assert after["escalations"] and not before["escalations"]

    port = _free_port()
    process = _serve(port, seed_file)
    try:
        restarted, order = asyncio.run(
            _calls(
                f"http://127.0.0.1:{port}/sse",
                [summary, ("lookup_order", {"order_id": "ORD-101"})],
            )
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert _summary_without_session(restarted) == _summary_without_session(before)
    assert order["item"] == PLANTED
