"""Startup seeding: plant consumer-authored text into one named record.

Two formats are read: this package's ``mini-agents-seed-v1``
(``seed.schema.json``), and ``artifact-seed-v1``, the ``seed.json`` of an
artifact package. They differ in their bounds: the first caps the item count
and every item's content, the second requires ``carrier_operation`` and bounds
each item's content by its slot's ``max_chars``, which the declaration files
state. Only the first writes ``<seed-file>.applied.json``.

A seed changes state only. No tool is added or changed, so the safe and unsafe
executors stay in lockstep and the hook works in both modes. Each domain keeps
its slot table in ``domains/<name>/seeds.py``; this module holds what the
domains share.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

SEED_SCHEMA_VERSION = "mini-agents-seed-v1"
ARTIFACT_SEED_VERSION = "artifact-seed-v1"
APPLIED_SCHEMA_VERSION = "mini-agents-seed-applied-v1"
EXIT_CODE = 3
MAX_ITEMS = 4
MAX_CONTENT_CHARS = 2000
MODES = ("replace", "append")

REASONS = (
    "seed_file_unreadable",
    "schema_mismatch",
    "too_many_items",
    "content_too_long",
    "unknown_slot",
    "unknown_record",
    "carrier_mismatch",
    "applied_write_failed",
)

_RECORD_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_DOCUMENT_KEYS = {"schema_version", "carrier_operation", "items"}
_ITEM_KEYS = {"slot", "record_ref", "mode", "content"}


class SeedError(Exception):
    """A seed the process refuses; ``reason`` is one of ``REASONS``."""

    def __init__(self, reason: str, detail: str = ""):
        if reason not in REASONS:
            raise ValueError(f"unknown seed refusal reason: {reason}")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = " ".join(detail.split())

    def stderr_line(self) -> str:
        line = f"mini-agents seed-error reason={self.reason}"
        return f"{line} detail={self.detail}" if self.detail else line


@dataclass(frozen=True)
class SeedItem:
    slot: str
    record_ref: str
    mode: str
    content: str


@dataclass(frozen=True)
class SeedDocument:
    carrier_operation: Optional[str]
    items: list[SeedItem]
    schema_version: str = SEED_SCHEMA_VERSION


@dataclass(frozen=True)
class Slot:
    """One text field of one record kind that a read tool returns to the agent.

    ``record_key`` names the record in the carrier's result: for a field inside
    a list (``documents[].body``) it is the list items' id property, otherwise
    it is an argument of the carrier that the result repeats.
    ``records`` and ``unsafe_records`` list the record ids the hook accepts in
    safe and unsafe mode. They differ where the two executors expose different
    records: a seed on a record the carrier never returns would reach no agent.
    ``max_chars`` bounds one item's content; ``seed.schema.json`` states the
    same limit for every slot.
    """

    id: str
    carrier_operation: str
    record_key: str
    other_readers: tuple[str, ...]
    field: str
    controller: str
    content_kind: str
    records: tuple[str, ...]
    unsafe_records: tuple[str, ...]
    container: str
    attribute: str
    max_chars: int = MAX_CONTENT_CHARS

    def accepted(self, unsafe: bool) -> tuple[str, ...]:
        return self.unsafe_records if unsafe else self.records


def parse_seed(raw: bytes) -> SeedDocument:
    """Validate the seed document's shape and bounds; touch no state.

    The ``schema_version`` picks the format. A slot's ``max_chars`` is checked
    when the item is applied, since only then is the slot known.
    """
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise SeedError(
            "schema_mismatch", f"not valid JSON ({type(exc).__name__})"
        ) from exc
    if isinstance(document, dict) and (
        document.get("schema_version") == ARTIFACT_SEED_VERSION
    ):
        items = _artifact_items(document)
        return SeedDocument(
            carrier_operation=document["carrier_operation"],
            items=[_parse_item(item, _ARTIFACT_ITEM_RULES) for item in items],
            schema_version=ARTIFACT_SEED_VERSION,
        )
    items = _document_items(document)
    return SeedDocument(
        carrier_operation=document.get("carrier_operation"),
        items=[_check_length(_parse_item(item, _ITEM_RULES)) for item in items],
    )


def _document_items(document: Any) -> list:
    if not isinstance(document, dict) or set(document) - _DOCUMENT_KEYS:
        raise SeedError("schema_mismatch", "document keys")
    if document.get("schema_version") != SEED_SCHEMA_VERSION:
        raise SeedError("schema_mismatch", "schema_version")
    carrier = document.get("carrier_operation")
    if carrier is not None and not isinstance(carrier, str):
        raise SeedError("schema_mismatch", "carrier_operation")
    items = document.get("items")
    if not isinstance(items, list):
        raise SeedError("schema_mismatch", "items")
    if len(items) > MAX_ITEMS:
        raise SeedError("too_many_items", f"{len(items)} items, limit {MAX_ITEMS}")
    if not items:
        raise SeedError("schema_mismatch", "items is empty")
    return items


def _artifact_items(document: dict) -> list:
    if set(document) != _DOCUMENT_KEYS:
        raise SeedError("schema_mismatch", "document keys")
    if not _is_text(document["carrier_operation"]):
        raise SeedError("schema_mismatch", "carrier_operation")
    items = document["items"]
    if not isinstance(items, list):
        raise SeedError("schema_mismatch", "items")
    if not items:
        raise SeedError("schema_mismatch", "items is empty")
    return items


def _parse_item(item: Any, rules: tuple) -> SeedItem:
    if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
        raise SeedError("schema_mismatch", "item keys")
    for field, is_valid in rules:
        if not is_valid(item[field]):
            raise SeedError("schema_mismatch", field)
    return SeedItem(item["slot"], item["record_ref"], item["mode"], item["content"])


def _check_length(
    item: SeedItem, limit: int = MAX_CONTENT_CHARS, where: str = ""
) -> SeedItem:
    if len(item.content) > limit:
        raise SeedError(
            "content_too_long", f"{len(item.content)} characters, limit {limit}{where}"
        )
    return item


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_record_ref(value: Any) -> bool:
    return isinstance(value, str) and bool(_RECORD_REF.match(value))


# Order fixes which field a multiply-malformed item reports.
_ITEM_RULES = (
    ("slot", lambda value: isinstance(value, str)),
    ("record_ref", _is_record_ref),
    ("mode", lambda value: value in MODES),
    ("content", _is_text),
)
# artifact-seed-v1 states no record pattern; an unknown record is refused on apply.
_ARTIFACT_ITEM_RULES = (
    ("slot", _is_text),
    ("record_ref", _is_text),
    ("mode", lambda value: value in MODES),
    ("content", _is_text),
)


def apply_items(
    slots: Sequence[Slot],
    state: Any,
    items: Sequence[SeedItem],
    *,
    unsafe: bool,
    carrier: Optional[str],
) -> list[str]:
    """Resolve every item, then write them; a refusal changes nothing."""
    by_id = {slot.id: slot for slot in slots}
    resolved = [(_resolve(by_id, item, unsafe, carrier), item) for item in items]
    for slot, item in resolved:
        record = getattr(state, slot.container)[item.record_ref]
        current = getattr(record, slot.attribute)
        text = item.content if item.mode == "replace" else f"{current} {item.content}"
        setattr(record, slot.attribute, text)
    return [item.slot for _, item in resolved]


def _resolve(
    by_id: dict[str, Slot], item: SeedItem, unsafe: bool, carrier: Optional[str]
) -> Slot:
    slot = by_id.get(item.slot)
    if slot is None:
        raise SeedError("unknown_slot", f"slot {item.slot!r}")
    if carrier is not None and carrier != slot.carrier_operation:
        raise SeedError("carrier_mismatch", f"{slot.id} is carried by {slot.carrier_operation}")
    if item.record_ref not in slot.accepted(unsafe):
        raise SeedError("unknown_record", f"record {item.record_ref!r} for slot {slot.id}")
    _check_length(item, slot.max_chars, f" for slot {slot.id}")
    return slot


def seeded_state(domain: str, seed_file: str, *, unsafe: bool) -> Any:
    """Read the seed file, build the domain state, plant the seed, confirm it."""
    path = Path(seed_file)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SeedError("seed_file_unreadable", type(exc).__name__) from exc
    document = parse_seed(raw)
    seeds = importlib.import_module(f"mini_agents.domains.{domain}.seeds")
    state = seeds.new_state()
    applied = seeds.apply_seed(
        state, document.items, unsafe=unsafe, carrier=document.carrier_operation
    )
    if document.schema_version == SEED_SCHEMA_VERSION:
        write_applied(path, domain=domain, unsafe=unsafe, raw=raw, slots=applied)
    return state


def applied_path(seed_file: Path) -> Path:
    return seed_file.with_name(seed_file.name + ".applied.json")


def write_applied(
    seed_file: Path, *, domain: str, unsafe: bool, raw: bytes, slots: list[str]
) -> None:
    confirmation = {
        "schema_version": APPLIED_SCHEMA_VERSION,
        "domain": domain,
        "mode": "unsafe" if unsafe else "safe",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "slots": slots,
    }
    target = applied_path(seed_file)
    scratch = target.with_name(target.name + ".part")
    try:
        scratch.write_text(json.dumps(confirmation, indent=2, sort_keys=True) + "\n")
        os.replace(scratch, target)
    except OSError as exc:
        scratch.unlink(missing_ok=True)
        raise SeedError("applied_write_failed", type(exc).__name__) from exc
