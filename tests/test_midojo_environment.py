"""Each domain's ledger round-trips through its MiDojo environment."""

from __future__ import annotations

import importlib
import json

import pytest

from mini_agents.kernel import mcp_server
from mini_agents.kernel.seeds import SeedItem

DOMAINS = mcp_server.DOMAINS
STATE_CLASS = {
    "klarna": "KlarnaState",
    "airbnb": "AirbnbState",
    "occiai": "OcciAIState",
}
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

WRITTEN = {"klarna": "escalations", "airbnb": "trust_cases", "occiai": "questionnaires"}


def _module(domain, name):
    return importlib.import_module(f"mini_agents.domains.{domain}.{name}")


def _state(domain):
    return getattr(_module(domain, "state"), STATE_CLASS[domain])()


def _executor(domain, state):
    tools = _module(domain, "tools")
    return next(v for k, v in vars(tools).items() if k.endswith("ToolExecutor"))(state)


def _round_trip(state):
    return type(state).from_environment(state.to_environment())


def _seeded_cases():
    for domain in DOMAINS:
        for slot in _module(domain, "seeds").SLOTS:
            for record in slot.records:
                for mode in ("replace", "append"):
                    yield pytest.param(
                        domain, slot, record, mode, id=f"{slot.id}-{record}-{mode}"
                    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_default_ledger_round_trips(domain):
    state = _state(domain)

    assert _round_trip(state).to_environment() == state.to_environment()


@pytest.mark.parametrize("domain, slot, record, mode", list(_seeded_cases()))
def test_every_seeded_ledger_round_trips(domain, slot, record, mode):
    state = _state(domain)
    item = SeedItem(slot.id, record, mode, "PLANTED text")
    _module(domain, "seeds").apply_seed(state, [item])

    assert _round_trip(state).to_environment() == state.to_environment()


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_ledger_after_writes_round_trips(domain):
    state = _state(domain)
    tool, args = WRITES[domain]

    result = _executor(domain, state).execute(tool, args)

    assert "error" not in result, result
    assert getattr(state, WRITTEN[domain])
    assert _round_trip(state).to_environment() == state.to_environment()


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_round_trip_keeps_the_session_id(domain):
    state = _state(domain)

    assert _round_trip(state).session_id == state.session_id


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_environment_survives_json(domain):
    environment = _state(domain).to_environment()

    assert json.loads(json.dumps(environment)) == environment


@pytest.mark.parametrize("domain", DOMAINS)
def test_a_round_tripped_state_gives_the_same_state_summary(domain):
    state = _state(domain)
    tool, args = WRITES[domain]
    _executor(domain, state).execute(tool, args)
    summary = f"get_{domain}_state_summary"

    before = _executor(domain, state).execute(summary, {})
    after = _executor(domain, _round_trip(state)).execute(summary, {})

    assert after == before


def _json_type(value):
    return type(value).__name__


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_record_container_has_one_shape(domain):
    environment = _state(domain).to_environment()

    for name, container in environment.items():
        if not isinstance(container, dict) or not container:
            continue
        if not all(isinstance(v, dict) for v in container.values()):
            continue
        records = list(container.values())
        shape = {k: _json_type(v) for k, v in records[0].items()}
        for key, record in container.items():
            assert {k: _json_type(v) for k, v in record.items()} == shape, (
                f"{name}.{key}"
            )


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_state_attribute_is_in_the_environment(domain):
    state = _state(domain)

    assert set(vars(state)) == set(type(state).ENVIRONMENT_ATTRIBUTES)
