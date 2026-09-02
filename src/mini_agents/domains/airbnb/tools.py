"""Safe MiniAirbnb executor. Sensitive writes go through workflow rules."""

from __future__ import annotations

from typing import Any

from mini_agents.domains.airbnb.state import (
    INVALID_LISTING_CONFIGS,
    TRUST_SAFETY_TOPICS,
    AirbnbState,
    Message,
    Modification,
    TrustCase,
)
from mini_agents.kernel.state import new_short_id, utc_now


class AirbnbToolExecutor:
    def __init__(self, state: AirbnbState):
        self.state = state

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict:
        fn = getattr(self, f"_tool_{tool_name}", None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool_name}: {exc}"}

    def _is_party(self, reservation) -> bool:
        actor = self.state.authenticated_actor_id
        return actor in {reservation.guest_id, reservation.host_id}

    def _tool_get_reservation(self, reservation_id: str) -> dict:
        reservation = self.state.reservations.get(reservation_id)
        if not reservation:
            return {"error": f"Reservation '{reservation_id}' not found."}
        if not self._is_party(reservation):
            return {
                "error": "AUTHORIZATION: Reservation is not visible to this actor."
            }
        self.state.log_audit(
            "GET_RESERVATION",
            self.state.authenticated_actor_id,
            ["reservation"],
            "success",
        )
        return {
            "reservation_id": reservation.reservation_id,
            "listing_id": reservation.listing_id,
            "guest_id": reservation.guest_id,
            "host_id": reservation.host_id,
            "check_in": reservation.check_in,
            "check_out": reservation.check_out,
            "status": reservation.status,
            "payment_status": reservation.payment_status,
            "modification_allowed": reservation.modification_allowed,
        }

    def _tool_get_listing(self, listing_id: str) -> dict:
        listing = self.state.listings.get(listing_id)
        if not listing:
            return {"error": f"Listing '{listing_id}' not found."}
        actor = self.state.authenticated_actor_id
        party_ids = {
            reservation.guest_id
            for reservation in self.state.reservations.values()
            if reservation.listing_id == listing_id
        }
        party_ids.add(listing.host_id)
        if actor not in party_ids:
            return {"error": "AUTHORIZATION: Listing is not visible to this actor."}
        return {
            "listing_id": listing.listing_id,
            "host_id": listing.host_id,
            "title": listing.title,
            "occupancy": listing.occupancy,
            "city": listing.city,
            "blocked_dates": list(listing.blocked_dates),
            "status": listing.status,
        }

    def _tool_check_availability(self, listing_id: str, date: str) -> dict:
        listing = self.state.listings.get(listing_id)
        if not listing:
            return {"error": f"Listing '{listing_id}' not found."}
        available = date not in listing.blocked_dates and listing.status == "active"
        return {
            "listing_id": listing_id,
            "date": date,
            "available": available,
            "blocked_dates": list(listing.blocked_dates),
        }

    def _tool_lookup_policy(self, query: str) -> dict:
        needle = query.strip().lower()
        hits = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "body": doc.body,
            }
            for doc in self.state.policy_docs.values()
            if doc.current
            and doc.author == "airbnb"
            and (
                needle in doc.title.lower()
                or needle in doc.body.lower()
                or any(needle in topic for topic in doc.topics)
            )
        ]
        if not hits:
            return {
                "status": "NO_CURRENT_POLICY",
                "documents": [],
                "message": "No current Airbnb policy snippet matches this query.",
            }
        return {"status": "OK", "documents": hits}

    def _tool_modify_booking(
        self,
        reservation_id: str,
        check_in: str,
        check_out: str,
        reason: str,
    ) -> dict:
        reservation = self.state.reservations.get(reservation_id)
        if not reservation:
            return {"error": f"Reservation '{reservation_id}' not found."}
        if not self._is_party(reservation):
            return {
                "error": "AUTHORIZATION: Only a party to the stay may request a change."
            }

        blocked = (
            not reservation.modification_allowed
            or reservation.payment_status != "paid"
            or check_out <= check_in
        )
        if blocked:
            modification = Modification(
                modification_id=new_short_id("MOD"),
                reservation_id=reservation.reservation_id,
                requested_check_in=check_in,
                requested_check_out=check_out,
                reason=reason,
                status="REJECTED",
                timestamp=utc_now(),
            )
            self.state.modifications.append(modification)
            self.state.log_audit(
                "MODIFY_BOOKING",
                self.state.authenticated_actor_id,
                ["reservation", "dates"],
                "rejected",
            )
            return {
                "status": "REJECTED",
                "modification_id": modification.modification_id,
                "reservation_id": reservation.reservation_id,
                "check_in": reservation.check_in,
                "check_out": reservation.check_out,
                "message": reservation.modification_block_reason
                or "Modification workflow rejected this change.",
            }

        reservation.check_in = check_in
        reservation.check_out = check_out
        modification = Modification(
            modification_id=new_short_id("MOD"),
            reservation_id=reservation.reservation_id,
            requested_check_in=check_in,
            requested_check_out=check_out,
            reason=reason,
            status="COMPLETED",
            timestamp=utc_now(),
        )
        self.state.modifications.append(modification)
        return {
            "status": "COMPLETED",
            "modification_id": modification.modification_id,
            "reservation_id": reservation.reservation_id,
            "check_in": reservation.check_in,
            "check_out": reservation.check_out,
        }

    def _tool_update_listing(
        self,
        listing_id: str,
        occupancy: int,
        config_flags: list[str] | None = None,
    ) -> dict:
        listing = self.state.listings.get(listing_id)
        if not listing:
            return {"error": f"Listing '{listing_id}' not found."}
        if self.state.authenticated_actor_id != listing.host_id:
            return {"error": "AUTHORIZATION: Only the host may update a listing."}
        flags = list(config_flags or listing.config_flags)
        invalid = set(flags) & INVALID_LISTING_CONFIGS or occupancy < 1
        if invalid:
            return {
                "error": "TOOL_GUARDRAIL: Listing configuration is invalid or unsafe.",
                "listing_id": listing_id,
                "occupancy": listing.occupancy,
                "config_flags": list(listing.config_flags),
            }
        listing.occupancy = occupancy
        listing.config_flags = flags
        listing.status = "active"
        return {
            "status": "UPDATED",
            "listing_id": listing.listing_id,
            "occupancy": listing.occupancy,
            "config_flags": list(listing.config_flags),
        }

    def _tool_moderate_message(
        self,
        reservation_id: str,
        body: str,
        sender_id: str | None = None,
    ) -> dict:
        reservation = self.state.reservations.get(reservation_id)
        if not reservation:
            return {"error": f"Reservation '{reservation_id}' not found."}
        sender = sender_id or self.state.authenticated_actor_id
        lowered = body.lower()
        flagged = any(
            topic in lowered for topic in TRUST_SAFETY_TOPICS
        ) or "wire transfer" in lowered or "off-platform" in lowered
        status = "blocked" if flagged else "allowed"
        message = Message(
            message_id=new_short_id("MSG"),
            reservation_id=reservation_id,
            sender_id=sender,
            body=body,
            moderation_status=status,
        )
        self.state.messages.append(message)
        return {
            "status": status.upper(),
            "message_id": message.message_id,
            "reservation_id": reservation_id,
        }

    def _tool_escalate_trust_safety(
        self,
        reservation_id: str,
        topic: str,
        reason: str,
    ) -> dict:
        reservation = self.state.reservations.get(reservation_id)
        if not reservation:
            return {"error": f"Reservation '{reservation_id}' not found."}
        topic_key = topic.strip().lower()
        case = TrustCase(
            case_id=new_short_id("TS"),
            reservation_id=reservation_id,
            topic=topic_key,
            reason=reason,
            timestamp=utc_now(),
        )
        self.state.trust_cases.append(case)
        return {
            "status": "ESCALATED",
            "case_id": case.case_id,
            "required": topic_key in TRUST_SAFETY_TOPICS,
        }

    def _tool_get_airbnb_state_summary(self) -> dict:
        return {
            "session_id": self.state.session_id,
            "authenticated_actor_id": self.state.authenticated_actor_id,
            "reservations": {
                reservation_id: {
                    "listing_id": reservation.listing_id,
                    "guest_id": reservation.guest_id,
                    "host_id": reservation.host_id,
                    "check_in": reservation.check_in,
                    "check_out": reservation.check_out,
                    "status": reservation.status,
                    "payment_status": reservation.payment_status,
                    "modification_allowed": reservation.modification_allowed,
                }
                for reservation_id, reservation in self.state.reservations.items()
            },
            "listings": {
                listing_id: {
                    "host_id": listing.host_id,
                    "occupancy": listing.occupancy,
                    "blocked_dates": list(listing.blocked_dates),
                    "config_flags": list(listing.config_flags),
                    "status": listing.status,
                }
                for listing_id, listing in self.state.listings.items()
            },
            "modifications": [
                {
                    "modification_id": item.modification_id,
                    "reservation_id": item.reservation_id,
                    "requested_check_in": item.requested_check_in,
                    "requested_check_out": item.requested_check_out,
                    "status": item.status,
                    "reason": item.reason,
                }
                for item in self.state.modifications
            ],
            "messages": [
                {
                    "message_id": message.message_id,
                    "reservation_id": message.reservation_id,
                    "moderation_status": message.moderation_status,
                }
                for message in self.state.messages
            ],
            "trust_cases": [
                {
                    "case_id": case.case_id,
                    "reservation_id": case.reservation_id,
                    "topic": case.topic,
                    "status": case.status,
                }
                for case in self.state.trust_cases
            ],
            "audit_log": [
                {
                    "action": entry.action,
                    "actor": entry.customer_id,
                    "fields": entry.fields_accessed,
                    "outcome": entry.outcome,
                }
                for entry in self.state.audit_logs
            ],
        }
