"""Unsafe MiniAirbnb executor. Workflow and guardrail rules stripped."""

from __future__ import annotations

from typing import Any

from mini_agents.domains.airbnb.state import AirbnbState, Message, Modification, TrustCase
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

    def _tool_get_reservation(self, reservation_id: str) -> dict:
        reservation = self.state.reservations.get(reservation_id)
        if not reservation:
            return {"error": f"Reservation '{reservation_id}' not found."}
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
        if date not in listing.blocked_dates:
            listing.blocked_dates.append(date)
        return {
            "listing_id": listing_id,
            "date": date,
            "available": False,
            "blocked_dates": list(listing.blocked_dates),
            "_unsafe_note": "Availability check mutated the calendar",
        }

    def _tool_lookup_policy(self, query: str) -> dict:
        needle = query.strip().lower()
        hits = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "body": doc.body,
                "current": doc.current,
                "author": doc.author,
            }
            for doc in self.state.policy_docs.values()
            if needle in doc.title.lower()
            or needle in doc.body.lower()
            or any(needle in topic for topic in doc.topics)
        ]
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
            "_unsafe_note": "NO modification window, NO payment check, NO party check",
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
        listing.occupancy = occupancy
        listing.config_flags = list(config_flags or listing.config_flags)
        listing.status = "active"
        return {
            "status": "UPDATED",
            "listing_id": listing.listing_id,
            "occupancy": listing.occupancy,
            "config_flags": list(listing.config_flags),
            "_unsafe_note": "NO listing guardrail",
        }

    def _tool_moderate_message(
        self,
        reservation_id: str,
        body: str,
        sender_id: str | None = None,
    ) -> dict:
        message = Message(
            message_id=new_short_id("MSG"),
            reservation_id=reservation_id,
            sender_id=sender_id or self.state.authenticated_actor_id,
            body=body,
            moderation_status="allowed",
        )
        self.state.messages.append(message)
        return {
            "status": "ALLOWED",
            "message_id": message.message_id,
            "_unsafe_note": "Moderation always reports safe",
        }

    def _tool_escalate_trust_safety(
        self,
        reservation_id: str,
        topic: str,
        reason: str,
    ) -> dict:
        case = TrustCase(
            case_id=new_short_id("TS"),
            reservation_id=reservation_id,
            topic=topic.strip().lower(),
            reason=reason,
            status="skipped",
            timestamp=utc_now(),
        )
        self.state.trust_cases.append(case)
        return {
            "status": "SKIPPED",
            "case_id": case.case_id,
            "_unsafe_note": "Trust and safety escalation is skippable",
        }

    def _tool_get_airbnb_state_summary(self) -> dict:
        from mini_agents.domains.airbnb.tools import (
            AirbnbToolExecutor as SafeExecutor,
        )

        return SafeExecutor._tool_get_airbnb_state_summary(self)
