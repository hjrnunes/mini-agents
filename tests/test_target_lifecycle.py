"""A target starts, resets and stops from its config alone.

``target_lifecycle`` stands in for the testing tool: it reads a ``target.yaml``,
fills the placeholders and runs the commands the file names. The six
mini-agents targets name only a start (a fresh start is their reset, and they
serve once their port accepts a connection). The fixture target in
``tests/fixtures/stub_target`` is no mini-agents domain and names a health and
a reset command too, so the same driver runs both shapes. No model calls.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import urllib.request
from pathlib import Path

import pytest
from target_lifecycle import (
    ROOT,
    Process,
    authorize,
    fill,
    free_port,
    listening,
    python_values,
    read_config,
    start_target,
)
from test_reset import WRITES, _calls, _summary_without_session

from mini_agents.kernel import declarations, mcp_server
from mini_agents.kernel.target_config import TARGET_FILE

DOMAINS = mcp_server.DOMAINS
CASES = [(d, v) for d in DOMAINS for v in declarations.VARIANTS]
IDS = [f"{d}-{v}" for d, v in CASES]
STUB = ROOT / "tests/fixtures/stub_target/target.yaml"


def _config(domain, variant):
    return read_config(declarations.declaration_dir(domain, variant) / TARGET_FILE)


_start = start_target


async def _tool_names(url):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return {tool.name for tool in (await session.list_tools()).tools}


def _url(config, port):
    return fill([config["endpoint"]["mcp_url"]], {"port": port})[0]


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_a_target_starts_serves_its_declared_tools_and_stops(domain, variant):
    config = _config(domain, variant)
    declared = {
        tool["name"]
        for tool in json.loads(
            (
                declarations.declaration_dir(domain, variant)
                / declarations.TOOLS_LIST_FILE
            ).read_text()
        )["tools"]
    }

    process, port = _start(config)
    try:
        served = asyncio.run(_tool_names(_url(config, port)))
    finally:
        process.stop()

    assert served == declared
    assert not listening(port)
    assert process.group_is_gone()


def test_the_discover_stage_lists_the_state_summary_and_no_other_stage_does():
    config = _config("airbnb", "safe")
    (observer,) = config["endpoint"]["discover_only_tools"]

    plain, plain_port = _start(config)
    try:
        listed_plain = asyncio.run(_tool_names(_url(config, plain_port)))
    finally:
        plain.stop()
    discover, discover_port = _start(config, stage="discover")
    try:
        listed_discover = asyncio.run(_tool_names(_url(config, discover_port)))
    finally:
        discover.stop()

    assert observer not in listed_plain
    assert listed_discover == listed_plain | {observer}


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_restart_from_the_config_gives_the_initial_ledger(domain):
    config = _config(domain, "safe")
    summary = (f"get_{domain}_state_summary", {})
    tool, arguments = WRITES[domain]

    first, port = _start(config)
    try:
        before, written, after = asyncio.run(
            _calls(_url(config, port), [summary, (tool, arguments), summary])
        )
    finally:
        first.stop()
    assert "error" not in written
    assert _summary_without_session(after) != _summary_without_session(before)

    second, port = _start(config)
    try:
        (restarted,) = asyncio.run(_calls(_url(config, port), [summary]))
    finally:
        second.stop()

    assert _summary_without_session(restarted) == _summary_without_session(before)


def _stub_state(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=5) as reply:
        return json.load(reply)["counter"]


def _bump(port):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/bump", method="POST")
    urllib.request.urlopen(request, timeout=5).read()


def _run(config, name, port):
    command = fill(config["endpoint"][name], python_values(port=port))
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, check=False
    ).returncode


def test_the_fixture_target_starts_resets_and_stops_from_its_config():
    config = read_config(STUB)
    authorize(config)
    assert "gateway" not in config
    port = free_port()

    assert _run(config, "health", port) != 0
    process = Process(fill(config["endpoint"]["start"], python_values(port=port)))
    try:
        for _ in range(100):
            if _run(config, "health", port) == 0:
                break
        else:
            pytest.fail("the health command never passed")
        _bump(port)
        _bump(port)
        assert _stub_state(port) == 2

        assert _run(config, "reset", port) == 0
        assert _stub_state(port) == 0
        assert process.popen.poll() is None, "a reset command does not restart"
    finally:
        process.stop()

    assert not listening(port)
    assert process.group_is_gone()


def test_the_fixture_target_restarts_to_its_start_state_too():
    config = read_config(STUB)
    port = free_port()
    start = fill(config["endpoint"]["start"], python_values(port=port))

    first = Process(start)
    try:
        first.wait_serving(port, timeout=30)
        _bump(port)
        assert _stub_state(port) == 1
    finally:
        first.stop()
    second = Process(start)
    try:
        second.wait_serving(port, timeout=30)
        assert _stub_state(port) == 0
    finally:
        second.stop()


@pytest.mark.parametrize("claim", [None, "production"])
def test_a_config_that_does_not_say_mock_is_refused(claim):
    config = {k: v for k, v in read_config(STUB).items() if k != "authorized_as"}
    if claim is not None:
        config["authorized_as"] = claim

    with pytest.raises(PermissionError):
        authorize(config)


def test_the_fixture_config_sits_outside_the_declarations_directory():
    assert Path(declarations.DECLARATIONS_DIR) not in STUB.parents
