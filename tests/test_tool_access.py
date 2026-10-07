"""A tool's manifest access matches what it does to the ledger.

"read" means the safe executor changes no record; "write" means the unsafe
executor, which has no guardrails to refuse, changes one. Audit-log entries and
the session id do not count: lookups write them in every domain.
"""

from __future__ import annotations

import importlib

import pytest

from mini_agents.kernel.mcp_server import DOMAINS

STATE_CLASS = {"klarna": "KlarnaState", "airbnb": "AirbnbState", "occiai": "OcciAIState"}
EXECUTOR_CLASS = {
    "klarna": "KlarnaToolExecutor",
    "airbnb": "AirbnbToolExecutor",
    "occiai": "OcciAIToolExecutor",
}

ARGS = {
    "lookup_order": lambda state: {"order_id": "ORD-101"},
    "get_account_details": lambda state: {"customer_id": "CUST001"},
    "retrieve_policy": lambda state: {"query": "refund"},
    "get_klarna_state_summary": lambda state: {},
    "schedule_payment": lambda state: {"plan_id": "PLAN-12", "next_due": "2027-01-01"},
    "process_refund": lambda state: {"order_id": "ORD-101", "amount": 10.0, "reason": "r"},
    "escalate_to_human": lambda state: {"topic": "fees", "reason": "r"},
    "get_reservation": lambda state: {"reservation_id": "RES-101"},
    "get_listing": lambda state: {"listing_id": "LST-101"},
    "check_availability": lambda state: {"listing_id": "LST-101", "date": "2026-10-01"},
    "lookup_policy": lambda state: {"query": "modification"},
    "get_airbnb_state_summary": lambda state: {},
    "modify_booking": lambda state: {
        "reservation_id": "RES-101",
        "check_in": "2026-10-02",
        "check_out": "2026-10-06",
        "reason": "r",
    },
    "update_listing": lambda state: {"listing_id": "LST-101", "occupancy": 3},
    "moderate_message": lambda state: {"reservation_id": "RES-101", "body": "hello"},
    "escalate_trust_safety": lambda state: {
        "reservation_id": "RES-101",
        "topic": "scam",
        "reason": "r",
    },
    "get_referral": lambda state: {"patient_id": "PAT-104"},
    "get_education": lambda state: {"patient_id": "PAT-104"},
    "ask_clinical_question": lambda state: {"patient_id": "PAT-104", "question": "What?"},
    "get_occiai_state_summary": lambda state: {},
    "submit_questionnaire": lambda state: {
        "patient_id": "PAT-104",
        "answers": {"symptoms": "blurred vision"},
    },
    "summarize_for_ehr": lambda state: {"patient_id": "PAT-104", "text": "draft"},
    "commit_to_ehr": lambda state: {
        "patient_id": "PAT-104",
        "draft_id": state.ehr_drafts[0].draft_id,
    },
    "generate_education_video": lambda state: {"patient_id": "PAT-104"},
    "escalate_to_clinician": lambda state: {"patient_id": "PAT-104", "reason": "r"},
}


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _fresh(domain, unsafe):
    state = getattr(
        importlib.import_module(f"mini_agents.domains.{domain}.state"), STATE_CLASS[domain]
    )()
    module = f"mini_agents.domains.{domain}.tools{'_unsafe' if unsafe else ''}"
    executor = getattr(importlib.import_module(module), EXECUTOR_CLASS[domain])(state)
    return state, executor


def _ledger(state):
    return {
        name: repr(value)
        for name, value in vars(state).items()
        if name not in {"audit_logs", "session_id"}
    }


def _tools(domain, access):
    return [name for name, value in _seeds(domain).TOOL_ACCESS.items() if value == access]


def _cases(access):
    return [
        pytest.param(domain, tool, id=f"{domain}-{tool}")
        for domain in DOMAINS
        for tool in _tools(domain, access)
    ]


def test_the_call_table_covers_every_tool():
    named = {tool for domain in DOMAINS for tool in _seeds(domain).TOOL_ACCESS}

    assert named == set(ARGS)


@pytest.mark.parametrize("domain, tool", _cases("read"))
def test_a_read_tool_leaves_the_safe_ledger_unchanged(domain, tool):
    state, executor = _fresh(domain, unsafe=False)
    before = _ledger(state)

    result = executor.execute(tool, ARGS[tool](state))

    assert "Unknown tool" not in str(result)
    assert _ledger(state) == before


@pytest.mark.parametrize("domain, tool", _cases("write"))
def test_a_write_tool_changes_the_unsafe_ledger(domain, tool):
    state, executor = _fresh(domain, unsafe=True)
    if tool == "commit_to_ehr":
        executor.execute("summarize_for_ehr", ARGS["summarize_for_ehr"](state))
    before = _ledger(state)

    executor.execute(tool, ARGS[tool](state))

    assert _ledger(state) != before
