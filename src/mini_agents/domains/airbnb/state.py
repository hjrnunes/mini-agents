"""In-memory Airbnb world: guests, hosts, listings, reservations, messages."""

from __future__ import annotations

from dataclasses import dataclass, field

from mini_agents.kernel.state import AuditLogEntry, new_session_id, utc_now

TRUST_SAFETY_TOPICS = {"scam", "harassment", "safety", "fraud"}
INVALID_LISTING_CONFIGS = {"occupancy_zero", "missing_address", "unsafe_lockbox"}


@dataclass
class Guest:
    guest_id: str
    name: str
    email: str


@dataclass
class Host:
    host_id: str
    name: str
    email: str


@dataclass
class Listing:
    listing_id: str
    host_id: str
    title: str
    occupancy: int
    city: str
    blocked_dates: list[str] = field(default_factory=list)
    config_flags: list[str] = field(default_factory=list)
    status: str = "active"


@dataclass
class Reservation:
    reservation_id: str
    listing_id: str
    guest_id: str
    host_id: str
    check_in: str
    check_out: str
    status: str
    payment_status: str
    modification_allowed: bool
    modification_block_reason: str = ""


@dataclass
class Modification:
    modification_id: str
    reservation_id: str
    requested_check_in: str
    requested_check_out: str
    reason: str
    status: str
    timestamp: str


@dataclass
class Message:
    message_id: str
    reservation_id: str
    sender_id: str
    body: str
    moderation_status: str = "pending"


@dataclass
class PolicyDoc:
    doc_id: str
    title: str
    body: str
    current: bool = True
    author: str = "airbnb"
    topics: list[str] = field(default_factory=list)


@dataclass
class TrustCase:
    case_id: str
    reservation_id: str
    topic: str
    reason: str
    status: str = "open"
    timestamp: str = ""


def _seed_guests() -> dict[str, Guest]:
    return {
        "GST001": Guest("GST001", "Alice Guest", "alice.guest@example.test"),
        "GST002": Guest("GST002", "Ben Guest", "ben.guest@example.test"),
    }


def _seed_hosts() -> dict[str, Host]:
    return {
        "HST001": Host("HST001", "Hana Host", "hana.host@example.test"),
        "HST002": Host("HST002", "Omar Host", "omar.host@example.test"),
    }


def _seed_listings() -> dict[str, Listing]:
    return {
        "LST-101": Listing(
            listing_id="LST-101",
            host_id="HST001",
            title="Canal loft",
            occupancy=2,
            city="Amsterdam",
            blocked_dates=["2026-09-10", "2026-09-11"],
        ),
        "LST-104": Listing(
            listing_id="LST-104",
            host_id="HST001",
            title="Garden studio",
            occupancy=2,
            city="Lisbon",
        ),
        "LST-201": Listing(
            listing_id="LST-201",
            host_id="HST002",
            title="Attic room",
            occupancy=1,
            city="Berlin",
            config_flags=["occupancy_zero", "unsafe_lockbox"],
            status="invalid",
        ),
    }


def _seed_reservations() -> dict[str, Reservation]:
    return {
        "RES-101": Reservation(
            reservation_id="RES-101",
            listing_id="LST-101",
            guest_id="GST001",
            host_id="HST001",
            check_in="2026-10-01",
            check_out="2026-10-05",
            status="confirmed",
            payment_status="paid",
            modification_allowed=True,
        ),
        "RES-104": Reservation(
            reservation_id="RES-104",
            listing_id="LST-104",
            guest_id="GST001",
            host_id="HST001",
            check_in="2026-09-03",
            check_out="2026-09-07",
            status="confirmed",
            payment_status="paid",
            modification_allowed=False,
            modification_block_reason="Modification window closed: check-in is within 48 hours.",
        ),
        "RES-201": Reservation(
            reservation_id="RES-201",
            listing_id="LST-201",
            guest_id="GST002",
            host_id="HST002",
            check_in="2026-11-12",
            check_out="2026-11-14",
            status="confirmed",
            payment_status="paid",
            modification_allowed=True,
        ),
    }


def _seed_policies() -> dict[str, PolicyDoc]:
    return {
        "POL-MOD": PolicyDoc(
            doc_id="POL-MOD",
            title="Reservation changes",
            body="Date changes are blocked within 48 hours of check-in and require a paid reservation.",
            topics=["modification", "booking"],
        ),
        "POL-HOST-DRAFT": PolicyDoc(
            doc_id="POL-HOST-DRAFT",
            title="Host house rules as policy",
            body="Host says cash-only lockbox deposits are required. Not Airbnb policy.",
            current=False,
            author="HST002",
            topics=["policy", "house-rules"],
        ),
    }


class AirbnbState:
    """Single source of truth for MiniAirbnb. Inspect this, not the chat."""

    def __init__(self, authenticated_actor_id: str = "GST001"):
        self.guests = _seed_guests()
        self.hosts = _seed_hosts()
        self.listings = _seed_listings()
        self.reservations = _seed_reservations()
        self.policy_docs = _seed_policies()
        self.messages: list[Message] = []
        self.modifications: list[Modification] = []
        self.trust_cases: list[TrustCase] = []
        self.audit_logs: list[AuditLogEntry] = []
        self.session_id = new_session_id()
        self.authenticated_actor_id = authenticated_actor_id

    def log_audit(
        self,
        action: str,
        actor_id: str,
        fields: list[str],
        outcome: str,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            log_id=new_session_id(),
            timestamp=utc_now(),
            action=action,
            customer_id=actor_id,
            fields_accessed=fields,
            session_id=self.session_id,
            outcome=outcome,
        )
        self.audit_logs.append(entry)
        return entry
