from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


def new_session_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_short_id(prefix: str) -> str:
    return f"{prefix}{str(uuid.uuid4())[:6].upper()}"


@dataclass
class AuditLogEntry:
    log_id: str
    timestamp: str
    action: str
    customer_id: str
    fields_accessed: list[str]
    session_id: str
    outcome: str


def ledger_to_environment(state: object, attributes: tuple[str, ...]) -> dict:
    """Every named attribute as a JSON-native value, records as plain dicts."""
    return {name: _plain(getattr(state, name)) for name in attributes}


def _plain(value):
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def rebuild_records(kind: str, record_class, raw):
    """Rebuild a container of records: ``dict``, ``list`` or ``dict_of_lists``."""
    if kind == "dict":
        return {key: record_class(**record) for key, record in raw.items()}
    if kind == "list":
        return [record_class(**record) for record in raw]
    return {
        key: [record_class(**record) for record in records]
        for key, records in raw.items()
    }
