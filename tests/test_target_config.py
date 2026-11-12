"""Each variant's ``target.yaml`` says how to run it and holds no secret.

The files are generated (``tests/test_declarations.py`` checks them against the
generator); these tests check what a testing tool reads from them: the target
is a mock, its ports match the stack's, its commands use only known
placeholders, and the gateway's model wiring names the variables the gateway
reads and holds placeholders, never values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from target_lifecycle import PLACEHOLDER, agent_url, fill, read_config

from mini_agents.kernel import declarations, mcp_server
from mini_agents.kernel.gateway import API_KEY_ENV, BASE_URL_ENV, MODEL_ID_ENV
from mini_agents.kernel.stack import MCP_SERVERS
from mini_agents.kernel.target_config import (
    GATEWAY_PORT,
    SCHEMA_VERSION,
    TARGET_FILE,
    target_id,
)

CASES = [(d, v) for d in mcp_server.DOMAINS for v in declarations.VARIANTS]
IDS = [f"{d}-{v}" for d, v in CASES]
PLACEHOLDERS = {
    "python",
    "port",
    "mcp_url",
    "gateway_port",
    "model",
    "model_url",
    "model_key",
}
KEYS = {
    "schema_version",
    "target",
    "kind",
    "authorized_as",
    "endpoint",
    "gateway",
    "discovery",
}


def _config(domain, variant):
    return read_config(declarations.declaration_dir(domain, variant) / TARGET_FILE)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_config_states_a_live_mock_target(domain, variant):
    config = _config(domain, variant)

    assert set(config) == KEYS
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["kind"] == "live"
    assert config["authorized_as"] == "mock"
    assert config["target"] == target_id(domain, variant)


def test_the_six_targets_have_distinct_ids_and_ports():
    configs = [_config(domain, variant) for domain, variant in CASES]

    assert len({config["target"] for config in configs}) == len(CASES)
    assert len({config["endpoint"]["port"] for config in configs}) == len(CASES)
    assert {config["endpoint"]["port"] for config in configs} == {
        port for _, port, _ in MCP_SERVERS
    }


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_start_command_runs_this_domain_and_variant_on_the_given_port(
    domain, variant
):
    endpoint = _config(domain, variant)["endpoint"]

    command = fill(endpoint["start"], {"python": "py", "port": 1234})

    assert command[:3] == ["py", "-m", "mini_agents"]
    assert command[command.index("--domain") + 1] == domain
    assert command[command.index("--port") + 1] == "1234"
    assert ("--unsafe" in command) == (variant == "unsafe")
    assert endpoint["mcp_url"] == "http://127.0.0.1:{port}/sse"
    assert (domain, endpoint["port"], variant == "unsafe") in MCP_SERVERS


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_state_summary_is_a_discover_only_observer(domain, variant):
    config = _config(domain, variant)
    tools_list = (
        declarations.declaration_dir(domain, variant) / declarations.TOOLS_LIST_FILE
    )
    listed = {tool["name"] for tool in json.loads(tools_list.read_text())["tools"]}

    assert config["endpoint"]["discover_only_tools"] == [f"get_{domain}_state_summary"]
    assert not set(config["endpoint"]["discover_only_tools"]) & listed
    assert config["endpoint"]["start_stage"] == {"discover": ["--state-summary"]}
    assert config["endpoint"]["unset_env"] == ["MINI_AGENTS_STATE_SUMMARY"]


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_discovery_operations_are_the_runtime_specs(domain, variant):
    config = _config(domain, variant)
    operations = declarations.read_runtime_spec(
        declarations.declaration_dir(domain, variant) / declarations.RUNTIME_SPEC_FILE
    )["operations"]

    assert config["discovery"]["read_only_operations"] == sorted(
        operations["read_only"]
    )
    assert config["discovery"]["mutating_operations"] == sorted(operations["mutating"])


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_gateway_block_names_its_port_start_and_model_wiring(domain, variant):
    config = _config(domain, variant)
    gateway = config["gateway"]

    assert set(gateway) == {"port", "start", "env"}
    assert gateway["port"] == GATEWAY_PORT
    assert config["endpoint"]["agent_url"] == f"http://127.0.0.1:{GATEWAY_PORT}/v1"
    command = fill(
        gateway["start"],
        {"python": "py", "mcp_url": "http://127.0.0.1:9/sse", "gateway_port": 4321},
    )
    assert command[:3] == ["py", "-m", "mini_agents.kernel.gateway"]
    assert command[command.index("--connector") + 1] == config["target"]
    assert command[command.index("--mcp-url") + 1] == "http://127.0.0.1:9/sse"
    assert command[command.index("--port") + 1] == "4321"


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_model_variables_hold_placeholders_and_never_a_value(domain, variant):
    env = _config(domain, variant)["gateway"]["env"]

    assert env == {
        BASE_URL_ENV: "{model_url}",
        API_KEY_ENV: "{model_key}",
        MODEL_ID_ENV: "{model}",
    }


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_every_placeholder_is_one_the_header_explains(domain, variant):
    path = declarations.declaration_dir(domain, variant) / TARGET_FILE
    config = read_config(path)
    header = "\n".join(
        line for line in path.read_text().splitlines() if line.startswith("#")
    )

    used = {name for text in _strings(config) for name in PLACEHOLDER.findall(text)}

    assert used <= PLACEHOLDERS
    for name in PLACEHOLDERS:
        assert "{" + name + "}" in header


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_no_string_in_the_file_looks_like_a_credential(domain, variant):
    path = declarations.declaration_dir(domain, variant) / TARGET_FILE
    text = path.read_text()

    assert not re.search(r"sk-[A-Za-z0-9]|Bearer |api[_-]?key\s*[:=]\s*\S", text)
    assert "https://" not in text


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_agent_url_follows_the_gateway_port_a_run_picks(domain, variant):
    config = _config(domain, variant)
    declared = config["gateway"]["port"]

    assert agent_url(config, declared) == config["endpoint"]["agent_url"]
    assert agent_url(config, 43127) == "http://127.0.0.1:43127/v1"


def test_the_gateway_port_is_not_a_target_port():
    assert GATEWAY_PORT not in {port for _, port, _ in MCP_SERVERS}


def test_the_declarations_directory_holds_one_config_per_variant():
    found = sorted(
        str(path.relative_to(declarations.DECLARATIONS_DIR))
        for path in Path(declarations.DECLARATIONS_DIR).rglob(TARGET_FILE)
    )

    assert found == sorted(f"{d}/{v}/{TARGET_FILE}" for d, v in CASES)
