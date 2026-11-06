"""A fresh start is the reset: two starts give the same initial ledger.

A server keeps its ledger in memory only and reads nothing back from disk, so a
caller resets a target by stopping it and starting it again with the same
arguments. A seed the kernel API plants into a running state does not survive
the restart. The in-process tests drive ``main()`` with a builder that keeps the
state it creates, so no port is bound. The last test restarts a real server
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
from mini_agents.kernel.seeds import parse_seed

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


def _state(domain):
    return getattr(_module(domain, "state"), STATE_CLASS[domain])()


def _ledger(state):
    """Every record; each start draws its own session id."""
    environment = state.to_environment()
    environment.pop("session_id")
    return environment


def _start(monkeypatch, domain, *args):
    """Run ``main()`` once and return the state its builder created."""
    states = []

    def builder():
        states.append(_state(domain))
        return FakeServer()

    monkeypatch.setitem(mcp_server.BUILDERS, domain, builder)
    mcp_server.main(["--domain", domain, "--port", "1", *args])
    (state,) = states
    return state


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_snapshot_holds_every_state_attribute(domain):
    state = _state(domain)

    assert set(vars(state)) == set(type(state).ENVIRONMENT_ATTRIBUTES)


@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_a_fresh_start_gives_the_same_initial_ledger(monkeypatch, domain, unsafe):
    flags = ("--unsafe",) if unsafe else ()

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


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_seed_planted_after_start_does_not_survive_a_restart(monkeypatch, domain):
    seeds = _module(domain, "seeds")
    slot = seeds.SLOTS[0]
    document = parse_seed(
        json.dumps(
            {
                "schema_version": "mini-agents-seed-v1",
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
        ).encode()
    )
    planted = seeds.new_state()
    seeds.apply_seed(planted, document.items, carrier=document.carrier_operation)
    assert PLANTED in json.dumps(_ledger(planted))

    restarted = _start(monkeypatch, domain)

    assert _ledger(restarted) == _ledger(seeds.new_state())
    assert PLANTED not in json.dumps(_ledger(restarted))


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(port):
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


def test_restarting_a_real_server_resets_its_ledger():
    summary = ("get_klarna_state_summary", {})
    tool, arguments = WRITES["klarna"]

    port = _free_port()
    process = _serve(port)
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
    process = _serve(port)
    try:
        (restarted,) = asyncio.run(_calls(f"http://127.0.0.1:{port}/sse", [summary]))
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert _summary_without_session(restarted) == _summary_without_session(before)
