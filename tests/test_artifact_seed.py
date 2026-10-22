"""--seed-file reads artifact-seed-v1 as well as mini-agents-seed-v1.

An artifact package's ``seed.json`` names a seed slot, one of its records, a
mode and the content, and always its carrier operation. It plants the same
records the mini-agents format plants, its content is bounded by the slot's
``max_chars``, and it writes no ``<seed-file>.applied.json``.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

from mini_agents.kernel import mcp_server
from mini_agents.kernel.seeds import (
    ARTIFACT_SEED_VERSION,
    EXIT_CODE,
    MAX_ITEMS,
    SEED_SCHEMA_VERSION,
    SeedError,
    parse_seed,
)

DOMAINS = mcp_server.DOMAINS

ITEM = {
    "slot": "klarna.order.item",
    "record_ref": "ORD-101",
    "mode": "replace",
    "content": "planted",
}


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _document(items=None, *, version=ARTIFACT_SEED_VERSION, **fields):
    document = {
        "schema_version": version,
        "carrier_operation": "lookup_order",
        "items": [ITEM] if items is None else items,
        **fields,
    }
    return json.dumps(document).encode()


def _apply(domain, raw, *, unsafe=False):
    document = parse_seed(raw)
    seeds = _seeds(domain)
    state = seeds.new_state()
    seeds.apply_seed(
        state, document.items, unsafe=unsafe, carrier=document.carrier_operation
    )
    return state


def _ledger(state):
    """The state's records; each new state draws its own session id."""
    environment = state.to_environment()
    environment.pop("session_id")
    return environment


def _slot_items(slot, records, content="planted text"):
    modes = ("replace", "append")
    return [
        {
            "slot": slot.id,
            "record_ref": record,
            "mode": modes[index % 2],
            "content": content,
        }
        for index, record in enumerate(records)
    ]


def test_an_artifact_seed_parses_to_items():
    document = parse_seed(_document())

    assert document.schema_version == ARTIFACT_SEED_VERSION
    assert document.carrier_operation == "lookup_order"
    assert [(i.slot, i.record_ref, i.mode, i.content) for i in document.items] == [
        ("klarna.order.item", "ORD-101", "replace", "planted")
    ]


def test_a_mini_agents_seed_keeps_its_own_version():
    document = parse_seed(_document(version=SEED_SCHEMA_VERSION))

    assert document.schema_version == SEED_SCHEMA_VERSION


@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_both_formats_plant_the_same_records(domain, unsafe):
    for slot in _seeds(domain).SLOTS:
        records = slot.accepted(unsafe)
        for start in range(0, len(records), MAX_ITEMS):
            items = _slot_items(slot, records[start : start + MAX_ITEMS])
            fields = {"carrier_operation": slot.carrier_operation}

            ours = _apply(
                domain, _document(items, version=SEED_SCHEMA_VERSION, **fields),
                unsafe=unsafe,
            )
            artifact = _apply(domain, _document(items, **fields), unsafe=unsafe)

            assert _ledger(artifact) == _ledger(ours)
            assert _ledger(artifact) != _ledger(_seeds(domain).new_state())


@pytest.mark.parametrize("domain", DOMAINS)
def test_content_of_exactly_the_slot_limit_is_planted(domain):
    for slot in _seeds(domain).SLOTS:
        record = slot.records[0]
        content = "x" * slot.max_chars
        items = [{**_slot_items(slot, [record])[0], "mode": "replace", "content": content}]

        state = _apply(
            domain, _document(items, carrier_operation=slot.carrier_operation)
        )

        planted = getattr(state, slot.container)[record]
        assert getattr(planted, slot.attribute) == content


@pytest.mark.parametrize("domain", DOMAINS)
def test_content_over_the_slot_limit_is_refused(domain):
    for slot in _seeds(domain).SLOTS:
        items = _slot_items(slot, [slot.records[0]], content="x" * (slot.max_chars + 1))
        raw = _document(items, carrier_operation=slot.carrier_operation)

        with pytest.raises(SeedError) as refused:
            _apply(domain, raw)

        assert refused.value.reason == "content_too_long"
        assert refused.value.detail == (
            f"{slot.max_chars + 1} characters, limit {slot.max_chars} for slot {slot.id}"
        )


def test_an_over_long_item_leaves_every_record_untouched():
    from mini_agents.domains.klarna.state import KlarnaState

    items = [ITEM, {**ITEM, "record_ref": "ORD-102", "content": "x" * 2001}]
    seeds = _seeds("klarna")
    state = KlarnaState()
    before = _ledger(state)

    with pytest.raises(SeedError, match="content_too_long"):
        seeds.apply_seed(state, parse_seed(_document(items)).items, carrier="lookup_order")

    assert _ledger(state) == before


def test_an_artifact_seed_has_no_item_cap():
    items = [ITEM] * (MAX_ITEMS + 1)

    state = _apply("klarna", _document(items))

    assert state.orders["ORD-101"].item == "planted"


def _without(key):
    document = json.loads(_document())
    del document[key]
    return json.dumps(document).encode()


def _item_without(key):
    item = dict(ITEM)
    del item[key]
    return _document([item])


# The refusals of the artifact-seed-v1 schema's invalid examples, on this
# package's own records.
@pytest.mark.parametrize(
    "raw, detail",
    [
        (_without("carrier_operation"), "document keys"),
        (_document(note="x"), "document keys"),
        (_document(carrier_operation=""), "carrier_operation"),
        (_document(carrier_operation=None), "carrier_operation"),
        (_document([]), "items is empty"),
        (_document({"slot": "x"}), "items"),
        (_document([{**ITEM, "content": ""}]), "content"),
        (_document([{**ITEM, "mode": "prepend"}]), "mode"),
        (_document([{**ITEM, "max_chars": 2000}]), "item keys"),
        (_item_without("record_ref"), "item keys"),
        (_document([{**ITEM, "record_ref": ""}]), "record_ref"),
        (_document([{**ITEM, "slot": ""}]), "slot"),
        (_document(version="artifact-seed-v2"), "schema_version"),
    ],
)
def test_a_malformed_artifact_seed_is_a_schema_mismatch(raw, detail):
    with pytest.raises(SeedError) as refused:
        parse_seed(raw)

    assert (refused.value.reason, refused.value.detail) == ("schema_mismatch", detail)


def test_an_artifact_seed_on_another_carrier_is_refused():
    with pytest.raises(SeedError) as refused:
        _apply("klarna", _document(carrier_operation="retrieve_policy"))

    assert refused.value.reason == "carrier_mismatch"


def test_an_artifact_seed_on_an_unknown_record_is_refused():
    with pytest.raises(SeedError) as refused:
        _apply("klarna", _document([{**ITEM, "record_ref": "ORD 999"}]))

    assert refused.value.reason == "unknown_record"


class FakeServer:
    def __init__(self):
        self.settings = SimpleNamespace(host=None, port=None)

    def run(self, transport):
        pass


@pytest.fixture
def built(monkeypatch):
    states = []

    def builder(state=None):
        states.append(state)
        return FakeServer()

    monkeypatch.setitem(mcp_server.BUILDERS, "klarna", builder)
    return states


def _argv(path, *extra):
    return ["--domain", "klarna", "--port", "1", "--seed-file", str(path), *extra]


def test_the_server_starts_with_the_artifact_seed_planted(tmp_path, built):
    path = tmp_path / "seed.json"
    path.write_bytes(_document())

    mcp_server.main(_argv(path))

    assert built[0].orders["ORD-101"].item == "planted"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed.json"]


def test_an_over_long_artifact_seed_stops_the_server(tmp_path, built, capsys):
    path = tmp_path / "seed.json"
    path.write_bytes(_document([{**ITEM, "content": "x" * 2001}]))

    with pytest.raises(SystemExit) as stopped:
        mcp_server.main(_argv(path))

    assert stopped.value.code == EXIT_CODE
    assert built == []
    assert capsys.readouterr().err == (
        "mini-agents seed-error reason=content_too_long "
        "detail=2001 characters, limit 2000 for slot klarna.order.item\n"
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed.json"]
