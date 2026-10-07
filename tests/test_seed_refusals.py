"""Each refusal reason fires on its own input and leaves the state untouched."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from mini_agents.domains.klarna import seeds as klarna_seeds
from mini_agents.domains.klarna.state import KlarnaState
from mini_agents.kernel.seeds import (
    MAX_CONTENT_CHARS,
    MAX_ITEMS,
    REASONS,
    SeedError,
    parse_seed,
)

ITEM = {
    "slot": "klarna.order.item",
    "record_ref": "ORD-101",
    "mode": "replace",
    "content": "planted",
}


def _document(items=None, **extra):
    document = {
        "schema_version": "mini-agents-seed-v1",
        "items": [ITEM] if items is None else items,
    }
    document.update(extra)
    return json.dumps(document).encode()


def _apply(raw, *, unsafe=False):
    document = parse_seed(raw)
    state = KlarnaState()
    klarna_seeds.apply_seed(
        state, document.items, unsafe=unsafe, carrier=document.carrier_operation
    )
    return state


def _with(**changes):
    return {**ITEM, **changes}


def test_a_valid_document_parses_to_items():
    document = parse_seed(_document(carrier_operation="lookup_order"))

    assert document.carrier_operation == "lookup_order"
    assert [(i.slot, i.record_ref, i.mode, i.content) for i in document.items] == [
        ("klarna.order.item", "ORD-101", "replace", "planted")
    ]


def _schema_validator():
    path = Path(parse_seed.__code__.co_filename).parent.parent / "seed.schema.json"
    return jsonschema.Draft202012Validator(json.loads(path.read_text()))


def test_the_seed_schema_accepts_what_the_process_accepts():
    validator = _schema_validator()
    accepted = [
        _document(),
        _document(carrier_operation="lookup_order"),
        _document(items=[ITEM] * MAX_ITEMS),
        _document(items=[_with(content="x" * MAX_CONTENT_CHARS)]),
        _document(items=[_with(mode="append")]),
    ]

    for raw in accepted:
        parse_seed(raw)
        validator.validate(json.loads(raw))


def test_the_seed_schema_rejects_what_the_process_refuses_before_it_reads_state():
    validator = _schema_validator()
    refused = [
        raw for name, raw in SCHEMA_MISMATCHES.items() if name != "not json"
    ] + [
        _document(items=[ITEM] * (MAX_ITEMS + 1)),
        _document(items=[_with(content="x" * (MAX_CONTENT_CHARS + 1))]),
    ]

    for raw in refused:
        assert not validator.is_valid(json.loads(raw)), raw


def test_the_reason_codes_are_a_closed_set():
    assert REASONS == (
        "seed_file_unreadable",
        "schema_mismatch",
        "too_many_items",
        "content_too_long",
        "unknown_slot",
        "unknown_record",
        "carrier_mismatch",
        "applied_write_failed",
    )


SCHEMA_MISMATCHES = {
    "not json": b"{not json",
    "not an object": b"[]",
    "wrong version": _document().replace(b"seed-v1", b"seed-v0"),
    "missing items": json.dumps({"schema_version": "mini-agents-seed-v1"}).encode(),
    "empty items": _document(items=[]),
    "extra top-level key": _document(extra="x"),
    "carrier not a string": _document(carrier_operation=3),
    "item not an object": _document(items=["x"]),
    "item missing key": _document(items=[{k: v for k, v in ITEM.items() if k != "mode"}]),
    "item extra key": _document(items=[_with(extra="x")]),
    "bad mode": _document(items=[_with(mode="prepend")]),
    "empty content": _document(items=[_with(content="")]),
    "content not a string": _document(items=[_with(content=7)]),
    "record ref with a space": _document(items=[_with(record_ref="ORD 101")]),
    "slot not a string": _document(items=[_with(slot=3)]),
}


@pytest.mark.parametrize("raw", SCHEMA_MISMATCHES.values(), ids=SCHEMA_MISMATCHES.keys())
def test_malformed_documents_are_a_schema_mismatch(raw):
    with pytest.raises(SeedError) as caught:
        parse_seed(raw)

    assert caught.value.reason == "schema_mismatch"


def test_more_than_four_items_is_too_many_items():
    with pytest.raises(SeedError) as caught:
        parse_seed(_document(items=[ITEM] * (MAX_ITEMS + 1)))

    assert caught.value.reason == "too_many_items"


def test_four_items_are_accepted():
    assert len(parse_seed(_document(items=[ITEM] * MAX_ITEMS)).items) == MAX_ITEMS


def test_content_over_the_limit_is_content_too_long():
    with pytest.raises(SeedError) as caught:
        parse_seed(_document(items=[_with(content="x" * (MAX_CONTENT_CHARS + 1))]))

    assert caught.value.reason == "content_too_long"


def test_content_at_the_limit_is_accepted():
    items = parse_seed(_document(items=[_with(content="x" * MAX_CONTENT_CHARS)])).items

    assert len(items[0].content) == MAX_CONTENT_CHARS


def test_an_unknown_slot_is_refused():
    with pytest.raises(SeedError) as caught:
        _apply(_document(items=[_with(slot="klarna.order.nothing")]))

    assert caught.value.reason == "unknown_slot"


def test_a_slot_of_another_domain_is_an_unknown_slot():
    with pytest.raises(SeedError) as caught:
        _apply(_document(items=[_with(slot="airbnb.listing.title")]))

    assert caught.value.reason == "unknown_slot"


def test_an_unknown_record_is_refused():
    with pytest.raises(SeedError) as caught:
        _apply(_document(items=[_with(record_ref="ORD-999")]))

    assert caught.value.reason == "unknown_record"


def test_a_record_the_carrier_cannot_return_in_this_mode_is_unknown():
    raw = _document(items=[_with(record_ref="ORD-201")])

    with pytest.raises(SeedError) as caught:
        _apply(raw, unsafe=False)

    assert caught.value.reason == "unknown_record"
    assert _apply(raw, unsafe=True).orders["ORD-201"].item == "planted"


def test_a_carrier_that_does_not_match_the_slot_is_refused():
    with pytest.raises(SeedError) as caught:
        _apply(_document(carrier_operation="retrieve_policy"))

    assert caught.value.reason == "carrier_mismatch"


def test_a_refused_document_leaves_every_record_untouched():
    pristine = KlarnaState()
    state = KlarnaState()
    document = parse_seed(_document(items=[ITEM, _with(record_ref="ORD-999")]))

    with pytest.raises(SeedError):
        klarna_seeds.apply_seed(state, document.items)

    assert state.orders == pristine.orders


def test_the_stderr_line_is_one_typed_line_without_the_content():
    error = SeedError("unknown_record", "record 'ORD-999'\nfor slot\tx")

    assert error.stderr_line() == (
        "mini-agents seed-error reason=unknown_record detail=record 'ORD-999' for slot x"
    )


def test_a_reason_outside_the_closed_set_is_a_programming_error():
    with pytest.raises(ValueError):
        SeedError("whatever")
