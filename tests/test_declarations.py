"""Each domain ships tools-list.json and runtime-spec.yaml per variant.

``tools-list.json`` must be the ``tools/list`` result a client of the default
server receives, so the test starts each server in-process and lists its tools
over an in-memory MCP session. ``runtime-spec.yaml`` must hold what a
consumer's declared-target reader accepts (``declared-runtime-v1``) and agree
with the seed hook: the records it names are the ones that variant accepts,
and each slot's ``max_chars`` is the limit ``--seed-file`` enforces.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.resources
import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mini_agents.kernel import declarations, mcp_server
from mini_agents.kernel.seeds import (
    ARTIFACT_SEED_VERSION,
    MAX_CONTENT_CHARS,
    SeedError,
    parse_seed,
)

DOMAINS = mcp_server.DOMAINS
VARIANTS = declarations.VARIANTS
PACKAGE = Path(mcp_server.__file__).resolve().parent.parent

# What a declared-target reader accepts in a runtime spec: its top-level keys,
# a seed slot's keys and the ones it requires, and the operation classes.
SPEC_KEYS = {"schema_version", "delivery", "limits", "seed_slots", "operations"}
SLOT_KEYS = {
    "id", "carrier_operation", "field", "controller", "content_kind",
    "max_chars", "mode", "records",
}
SLOT_REQUIRED = SLOT_KEYS - {"mode", "records"}
OPERATION_KEYS = {"read_only", "mutating"}

CASES = [(domain, variant) for domain in DOMAINS for variant in VARIANTS]
IDS = [f"{domain}-{variant}" for domain, variant in CASES]


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _shipped_tools(domain, variant):
    path = declarations.declaration_dir(domain, variant) / declarations.TOOLS_LIST_FILE
    return json.loads(path.read_text())


def _shipped_spec(domain, variant):
    path = declarations.declaration_dir(domain, variant) / declarations.RUNTIME_SPEC_FILE
    return declarations.read_runtime_spec(path)


def _live_tools_list(domain, variant, monkeypatch):
    monkeypatch.setattr(mcp_server, "UNSAFE_MODE", variant == "unsafe")
    server = mcp_server.BUILDERS[domain]()

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            return await client.list_tools()

    result = asyncio.run(run())
    return result.model_dump(mode="json", by_alias=True, exclude_none=True)


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_files_ship_inside_the_package(domain, variant):
    shipped = importlib.resources.files("mini_agents") / "declarations" / domain / variant

    assert (shipped / "tools-list.json").is_file()
    assert (shipped / "runtime-spec.yaml").is_file()


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_tools_list_is_what_the_live_server_lists(domain, variant, monkeypatch):
    live = _live_tools_list(domain, variant, monkeypatch)

    shipped = _shipped_tools(domain, variant)

    assert shipped == live
    assert [tool["name"] for tool in shipped["tools"]]
    for tool in shipped["tools"]:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_tools_list_leaves_out_the_state_summary(domain, variant):
    names = {tool["name"] for tool in _shipped_tools(domain, variant)["tools"]}

    assert f"get_{domain}_state_summary" not in names
    assert names == set(_seeds(domain).TOOL_ACCESS) - {f"get_{domain}_state_summary"}


def test_the_committed_files_are_what_the_generator_writes():
    files = declarations.rendered()

    assert len(files) == len(CASES) * 2
    stale = [str(path) for path, text in files.items() if path.read_text() != text]
    assert stale == [], "run scripts/gen_declarations.py"


def test_nothing_else_is_in_the_declarations_directory():
    present = {
        path for path in declarations.DECLARATIONS_DIR.rglob("*") if path.is_file()
    }

    assert present == set(declarations.rendered())


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_runtime_spec_is_a_declared_runtime_spec(domain, variant):
    spec = _shipped_spec(domain, variant)
    tools = [tool["name"] for tool in _shipped_tools(domain, variant)["tools"]]

    assert set(spec) <= SPEC_KEYS
    assert spec["schema_version"] == "declared-runtime-v1"
    ids = [slot["id"] for slot in spec["seed_slots"]]
    assert ids and len(ids) == len(set(ids))
    for slot in spec["seed_slots"]:
        assert SLOT_REQUIRED <= set(slot) <= SLOT_KEYS
        for key in ("id", "carrier_operation", "field", "controller", "content_kind"):
            assert isinstance(slot[key], str) and slot[key].strip()
        assert slot["carrier_operation"] in tools
        assert type(slot["max_chars"]) is int and slot["max_chars"] >= 1
        assert slot["records"] and len(slot["records"]) == len(set(slot["records"]))
    # No tool carries annotations, so the spec classifies every one, once.
    assert set(spec["operations"]) == OPERATION_KEYS
    classified = spec["operations"]["read_only"] + spec["operations"]["mutating"]
    assert sorted(classified) == sorted(tools)


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_operations_follow_the_tool_access_table(domain, variant):
    spec = _shipped_spec(domain, variant)
    access = _seeds(domain).TOOL_ACCESS

    assert {access[name] for name in spec["operations"]["read_only"]} == {"read"}
    assert {access[name] for name in spec["operations"]["mutating"]} == {"write"}


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_slots_are_the_domain_slots_with_this_variants_records(domain, variant):
    spec = _shipped_spec(domain, variant)
    slots = _seeds(domain).SLOTS

    assert [slot["id"] for slot in spec["seed_slots"]] == [slot.id for slot in slots]
    for declared, slot in zip(spec["seed_slots"], slots):
        assert declared["carrier_operation"] == slot.carrier_operation
        assert declared["field"] == slot.field
        assert declared["records"] == list(slot.accepted(variant == "unsafe"))


def _artifact_seed(slot, record, content):
    return json.dumps(
        {
            "schema_version": ARTIFACT_SEED_VERSION,
            "carrier_operation": slot["carrier_operation"],
            "items": [
                {
                    "slot": slot["id"],
                    "record_ref": record,
                    "mode": "replace",
                    "content": content,
                }
            ],
        }
    ).encode()


def _plant(domain, variant, raw):
    seeds = _seeds(domain)
    document = parse_seed(raw)
    state = seeds.new_state()
    seeds.apply_seed(
        state,
        document.items,
        unsafe=variant == "unsafe",
        carrier=document.carrier_operation,
    )
    return state


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_slot_max_chars_is_the_limit_the_seed_hook_enforces(domain, variant):
    schema = json.loads((PACKAGE / "seed.schema.json").read_text())
    schema_limit = schema["properties"]["items"]["items"]["properties"]["content"][
        "maxLength"
    ]

    for slot in _shipped_spec(domain, variant)["seed_slots"]:
        limit = slot["max_chars"]
        assert limit == schema_limit == MAX_CONTENT_CHARS
        for record in slot["records"]:
            _plant(domain, variant, _artifact_seed(slot, record, "x" * limit))
            with pytest.raises(SeedError) as refused:
                _plant(domain, variant, _artifact_seed(slot, record, "x" * (limit + 1)))
            assert refused.value.reason == "content_too_long"
