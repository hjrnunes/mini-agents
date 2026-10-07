"""Startup seeding: plant consumer-authored text into one named record.

A seed changes state only. No tool is added or changed, so the safe and unsafe
executors stay in lockstep and the hook works in both modes. Each domain keeps
its slot table in ``domains/<name>/seeds.py``; this module holds what the
domains share.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

SEED_SCHEMA_VERSION = "mini-agents-seed-v1"
APPLIED_SCHEMA_VERSION = "mini-agents-seed-applied-v1"
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


@dataclass(frozen=True)
class Slot:
    """One text field of one record kind that a read tool returns to the agent.

    ``records`` and ``unsafe_records`` list the record ids the hook accepts in
    safe and unsafe mode. They differ where the two executors expose different
    records: a seed on a record the carrier never returns would reach no agent.
    """

    id: str
    carrier_operation: str
    other_readers: tuple[str, ...]
    field: str
    controller: str
    content_kind: str
    records: tuple[str, ...]
    unsafe_records: tuple[str, ...]
    container: str
    attribute: str

    def accepted(self, unsafe: bool) -> tuple[str, ...]:
        return self.unsafe_records if unsafe else self.records


def parse_seed(raw: bytes) -> SeedDocument:
    """Validate the seed document's shape and bounds; touch no state."""
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise SeedError(
            "schema_mismatch", f"not valid JSON ({type(exc).__name__})"
        ) from exc
    items = _document_items(document)
    return SeedDocument(
        carrier_operation=document.get("carrier_operation"),
        items=[_parse_item(item) for item in items],
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


def _parse_item(item: Any) -> SeedItem:
    if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
        raise SeedError("schema_mismatch", "item keys")
    slot, record_ref = item["slot"], item["record_ref"]
    mode, content = item["mode"], item["content"]
    if not isinstance(slot, str):
        raise SeedError("schema_mismatch", "slot")
    if not isinstance(record_ref, str) or not _RECORD_REF.match(record_ref):
        raise SeedError("schema_mismatch", "record_ref")
    if mode not in MODES:
        raise SeedError("schema_mismatch", "mode")
    if not isinstance(content, str) or not content:
        raise SeedError("schema_mismatch", "content")
    if len(content) > MAX_CONTENT_CHARS:
        raise SeedError(
            "content_too_long", f"{len(content)} characters, limit {MAX_CONTENT_CHARS}"
        )
    return SeedItem(slot, record_ref, mode, content)



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
    return slot

