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
import os
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mini_agents.kernel import declarations, mcp_server
from mini_agents.kernel.seeds import (
    ARTIFACT_SEED_VERSION,
    MAX_CONTENT_CHARS,
    SeedError,
    SeedItem,
    parse_seed,
)

DOMAINS = mcp_server.DOMAINS
VARIANTS = declarations.VARIANTS
PACKAGE = Path(mcp_server.__file__).resolve().parent.parent

# What a declared-target reader accepts in a runtime spec: its top-level keys,
# a seed slot's keys and the ones it requires, and the operation classes.
SPEC_KEYS = {"schema_version", "delivery", "limits", "seed_slots", "operations"}
SLOT_KEYS = {
    "id", "carrier_operation", "record_key", "field", "controller", "content_kind",
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


def _live_call(domain, variant, state, tool, arguments, monkeypatch):
    monkeypatch.setattr(mcp_server, "UNSAFE_MODE", variant == "unsafe")
    server = mcp_server.BUILDERS[domain](state)

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(run())


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
        for key in (
            "id",
            "carrier_operation",
            "record_key",
            "field",
            "controller",
            "content_kind",
        ):
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
        assert declared["record_key"] == slot.record_key
        assert declared["field"] == slot.field
        assert declared["records"] == list(slot.accepted(variant == "unsafe"))


# Each carrier's arguments, as agents call it, for the record a seed names. A
# policy query matches the record's title, which a planted body leaves alone.
POLICY_QUERY = {
    "POL-REFUND": "refund",
    "POL-SCHEDULE": "schedule",
    "POL-FEES-DRAFT": "fees",
    "POL-MOD": "modification",
    "POL-HOST-DRAFT": "house-rules",
}
PATIENT_OF_RECORD = {
    "REF-101": "PAT-101",
    "REF-104": "PAT-104",
    "REF-201": "PAT-201",
    "EDU-CATARACT": "PAT-104",
    "EDU-GYNAE": "PAT-201",
}
CARRIER_CALLS = {
    "lookup_order": lambda record: {"order_id": record},
    "retrieve_policy": lambda record: {"query": POLICY_QUERY[record]},
    "get_listing": lambda record: {"listing_id": record},
    "lookup_policy": lambda record: {"query": POLICY_QUERY[record]},
    "get_referral": lambda record: {"patient_id": PATIENT_OF_RECORD[record]},
    "get_education": lambda record: {"patient_id": PATIENT_OF_RECORD[record]},
}


def _list_field(field):
    """Split ``documents[].body`` into the list and the item property, else None."""
    if "[]." not in field:
        return None
    return tuple(field.split("[].", 1))


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_carrier_takes_the_arguments_agents_already_send(domain, variant):
    tools = {tool["name"]: tool for tool in _shipped_tools(domain, variant)["tools"]}
    for slot in _seeds(domain).SLOTS:
        inputs = tools[slot.carrier_operation]["inputSchema"]
        expected = set(CARRIER_CALLS[slot.carrier_operation](slot.records[0]))
        assert set(inputs["properties"]) == expected, slot.id
        assert set(inputs["required"]) == expected, slot.id


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_carrier_returns_structured_content_and_its_json_text_copy(
    domain, variant, monkeypatch
):
    unsafe = variant == "unsafe"
    seeds = _seeds(domain)
    for slot in seeds.SLOTS:
        for record in slot.accepted(unsafe):
            marker = f"STRUCTURED[{slot.id}:{record}:{variant}]"
            state = seeds.new_state()
            seeds.apply_seed(
                state,
                [SeedItem(slot.id, record, "replace", marker)],
                unsafe=unsafe,
            )
            arguments = CARRIER_CALLS[slot.carrier_operation](record)

            result = _live_call(
                domain, variant, state, slot.carrier_operation, arguments, monkeypatch
            )

            structured = result.structuredContent
            assert result.isError is False
            assert structured is not None and "error" not in structured, structured
            assert len(result.content) == 1 and result.content[0].type == "text"
            assert result.content[0].text == json.dumps(structured)
            listed = _list_field(slot.field)
            if listed:
                name, prop = listed
                items = [
                    item for item in structured[name]
                    if item[slot.record_key] == record
                ]
                assert [item[prop] for item in items] == [marker], slot.id
            else:
                assert structured[slot.record_key] == record, slot.id
                assert structured[slot.field] == marker, slot.id


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_carrier_schema_declares_the_record_key_and_slot_field(domain, variant):
    tools = {tool["name"]: tool for tool in _shipped_tools(domain, variant)["tools"]}
    for slot in _seeds(domain).SLOTS:
        carrier = tools[slot.carrier_operation]
        output = carrier["outputSchema"]
        assert output["type"] == "object"
        listed = _list_field(slot.field)
        if listed:
            name, prop = listed
            assert output["properties"][name]["type"] == "array"
            item = output["properties"][name]["items"]
            assert item["properties"][slot.record_key]["type"] == "string"
            assert item["properties"][prop]["type"] == "string"
        else:
            assert output["properties"][slot.record_key]["type"] == "string"
            assert output["properties"][slot.field]["type"] == "string"


def _artifact_generator_contract_files():
    """Find the artifact generator's runtime-contract folder.

    Set ``ARTIFACT_GENERATOR_RUNTIME_CONTRACT`` to its
    ``contracts/runtime-contract`` folder; the test skips without it.
    """
    location = os.environ.get("ARTIFACT_GENERATOR_RUNTIME_CONTRACT")
    if not location:
        pytest.skip("ARTIFACT_GENERATOR_RUNTIME_CONTRACT is not set")
    contract = Path(location)
    schema = contract / "artifact-runtime-contract-v2.schema.json"
    fixture = (
        contract
        / "artifact-runtime-contract-v2"
        / "valid"
        / "slot-with-record-key.json"
    )
    if not schema.is_file() or not fixture.is_file():
        pytest.skip("artifact-generator runtime-contract v2 files are absent")
    return schema, fixture


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_runtime_spec_slot_validates_against_runtime_contract_v2(domain, variant):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path, fixture_path = _artifact_generator_contract_files()
    schema = json.loads(schema_path.read_text())
    contract = json.loads(fixture_path.read_text())
    contract["seed_slots"] = [
        {**slot, "mode": "replace"}
        for slot in _shipped_spec(domain, variant)["seed_slots"]
    ]

    jsonschema.Draft202012Validator(schema).validate(contract)


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
