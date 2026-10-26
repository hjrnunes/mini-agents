"""Every seed slot plants text that its carrier tool returns, in both executors.

The tests instantiate the executors directly and read tool results; no
transport is involved. The call tables below are hand-kept so a new slot or tool
that nobody wired into a test fails here.
"""

from __future__ import annotations

import importlib
import json

import pytest

from mini_agents.domains.airbnb.state import AirbnbState
from mini_agents.domains.airbnb.tools import AirbnbToolExecutor
from mini_agents.domains.airbnb.tools_unsafe import (
    AirbnbToolExecutor as UnsafeAirbnbToolExecutor,
)
from mini_agents.domains.klarna.state import KlarnaState
from mini_agents.domains.klarna.tools import KlarnaToolExecutor
from mini_agents.domains.klarna.tools_unsafe import (
    KlarnaToolExecutor as UnsafeKlarnaToolExecutor,
)
from mini_agents.domains.occiai.state import OcciAIState
from mini_agents.domains.occiai.tools import OcciAIToolExecutor
from mini_agents.domains.occiai.tools_unsafe import (
    OcciAIToolExecutor as UnsafeOcciAIToolExecutor,
)
from mini_agents.kernel.seeds import SeedItem, seeded_state

DOMAINS = ("klarna", "airbnb", "occiai")

STATES = {"klarna": KlarnaState, "airbnb": AirbnbState, "occiai": OcciAIState}
EXECUTORS = {
    "klarna": {False: KlarnaToolExecutor, True: UnsafeKlarnaToolExecutor},
    "airbnb": {False: AirbnbToolExecutor, True: UnsafeAirbnbToolExecutor},
    "occiai": {False: OcciAIToolExecutor, True: UnsafeOcciAIToolExecutor},
}

# Read-tool arguments, keyed by tool name. A callable takes the seeded record id.
READ_ARGS = {
    "lookup_order": lambda record: {"order_id": record},
    "get_account_details": lambda record: {"customer_id": "CUST001"},
    "retrieve_policy": lambda record: {"doc_id": record},
    "get_klarna_state_summary": lambda record: {},
    "get_reservation": lambda record: {"reservation_id": "RES-101"},
    "get_listing": lambda record: {"listing_id": record},
    "check_availability": lambda record: {"listing_id": "LST-101", "date": "2026-10-01"},
    "lookup_policy": lambda record: {"doc_id": record},
    "get_airbnb_state_summary": lambda record: {},
    "get_patient_profile": lambda record: {},
    "get_referral": lambda record: {"referral_id": record},
    "get_education": lambda record: {"template_id": record},
    "ask_clinical_question": lambda record: {
        "patient_id": "PAT-104",
        "question": "What happens at the clinic?",
    },
    "get_occiai_state_summary": lambda record: {},
}


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _cases():
    for domain in DOMAINS:
        for slot in _seeds(domain).SLOTS:
            for unsafe in (False, True):
                records = slot.unsafe_records if unsafe else slot.records
                for record in records:
                    for mode in ("replace", "append"):
                        yield pytest.param(
                            domain,
                            slot,
                            unsafe,
                            record,
                            mode,
                            id=f"{slot.id}-{'unsafe' if unsafe else 'safe'}-{record}-{mode}",
                        )


def _values(result, path):
    nodes = [result]
    for part in path.split("."):
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nodes = [node[key] for node in nodes if key in node]
        if many:
            nodes = [element for node in nodes for element in node]
    return nodes


def _marker(slot, mode):
    return f"PLANTED[{slot.id}:{mode}]"


def _carrier_args(slot, record):
    return READ_ARGS[slot.carrier_operation](record)


@pytest.mark.parametrize("domain, slot, unsafe, record, mode", list(_cases()))
def test_carrier_returns_the_planted_text(domain, slot, unsafe, record, mode):
    executor_class = EXECUTORS[domain][unsafe]
    args = _carrier_args(slot, record)
    untouched = executor_class(STATES[domain]()).execute(slot.carrier_operation, args)
    state = STATES[domain]()
    marker = _marker(slot, mode)

    applied = _seeds(domain).apply_seed(
        state, [SeedItem(slot.id, record, mode, marker)], unsafe=unsafe
    )
    result = executor_class(state).execute(slot.carrier_operation, args)

    before = _values(untouched, slot.field)
    after = _values(result, slot.field)
    added = [value for value in after if value not in before]
    removed = [value for value in before if value not in after]
    assert applied == [slot.id]
    assert len(added) == 1 and len(removed) == 1
    expected = marker if mode == "replace" else f"{removed[0]} {marker}"
    assert added[0] == expected


@pytest.mark.parametrize("domain", DOMAINS)
def test_readers_of_each_slot_are_exactly_the_tools_that_return_the_text(domain):
    seeds = _seeds(domain)
    read_tools = [name for name, access in seeds.TOOL_ACCESS.items() if access == "read"]
    for slot in seeds.SLOTS:
        seen = set()
        for unsafe in (False, True):
            for record in slot.unsafe_records if unsafe else slot.records:
                state = STATES[domain]()
                marker = _marker(slot, "replace")
                seeds.apply_seed(
                    state, [SeedItem(slot.id, record, "replace", marker)], unsafe=unsafe
                )
                executor = EXECUTORS[domain][unsafe](state)
                for tool in read_tools:
                    result = executor.execute(tool, READ_ARGS[tool](record))
                    if marker in json.dumps(result):
                        seen.add(tool)
        assert seen == {slot.carrier_operation, *slot.other_readers}, slot.id


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_seed_changes_only_the_named_record(domain):
    seeds = _seeds(domain)
    for slot in seeds.SLOTS:
        state, pristine = STATES[domain](), STATES[domain]()
        record = slot.records[0]
        seeds.apply_seed(state, [SeedItem(slot.id, record, "replace", "x")])
        container = getattr(state, slot.container)
        untouched = getattr(pristine, slot.container)
        assert {k: v for k, v in container.items() if k != record} == {
            k: v for k, v in untouched.items() if k != record
        }, slot.id


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_slot_is_exercised_by_the_call_tables(domain):
    seeds = _seeds(domain)
    for slot in seeds.SLOTS:
        assert slot.carrier_operation in READ_ARGS, slot.id
        assert seeds.TOOL_ACCESS[slot.carrier_operation] == "read", slot.id
        assert (
            slot.record_key in READ_ARGS[slot.carrier_operation](slot.records[0])
        ), slot.id
        assert slot.records and slot.unsafe_records, slot.id
        assert len(slot.id.split(".")) == 3 and slot.id.startswith(f"{domain}."), slot.id
    assert set(READ_ARGS) >= {
        name for name, access in seeds.TOOL_ACCESS.items() if access == "read"
    }


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_twin_accepts_every_record_the_safe_target_accepts(domain):
    for slot in _seeds(domain).SLOTS:
        missing = sorted(set(slot.records) - set(slot.unsafe_records))
        assert missing == [], slot.id


def test_a_safe_education_seed_reaches_the_twin_agent(tmp_path):
    seed_file = tmp_path / "seed.json"
    seed_file.write_text(
        json.dumps(
            {
                "schema_version": "mini-agents-seed-v1",
                "carrier_operation": "get_education",
                "items": [
                    {
                        "slot": "occiai.education.script",
                        "record_ref": "EDU-CATARACT",
                        "mode": "replace",
                        "content": "PLANTED-ON-THE-TWIN",
                    }
                ],
            }
        )
    )

    state = seeded_state("occiai", str(seed_file), unsafe=True)
    result = UnsafeOcciAIToolExecutor(state).execute(
        "get_education", {"template_id": "EDU-CATARACT"}
    )

    assert result["script"] == "PLANTED-ON-THE-TWIN"
