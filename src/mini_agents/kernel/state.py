from __future__ import annotations

import uuid
from dataclasses import dataclass
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
