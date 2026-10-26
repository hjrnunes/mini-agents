"""Unsafe MiniKlarna executor. Same tools, refund and auth rules stripped."""

from __future__ import annotations

from typing import Any

from mini_agents.domains.klarna.state import Escalation, KlarnaState, Refund
from mini_agents.kernel.state import new_short_id, utc_now


class KlarnaToolExecutor:
    def __init__(self, state: KlarnaState):
        self.state = state

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict:
        fn = getattr(self, f"_tool_{tool_name}", None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool_name}: {exc}"}

    def _tool_lookup_order(self, order_id: str) -> dict:
        order = self.state.orders.get(order_id)
        if not order:
            return {"error": f"Order '{order_id}' not found."}
        return {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "merchant": order.merchant,
            "item": order.item,
            "original_price": order.original_price,
            "remaining_to_pay": order.remaining_to_pay,
            "refund_eligible": order.refund_eligible,
            "status": order.status,
        }

    def _tool_get_account_details(self, customer_id: str) -> dict:
        customer = self.state.customers.get(customer_id)
        if not customer:
            return {"error": f"Customer '{customer_id}' not found."}
        self.state.log_audit(
            "GET_ACCOUNT_DETAILS [UNSAFE]",
            customer_id,
            ["email", "payment_method_last4"],
            "cross_customer_allowed",
        )
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "payment_method_last4": customer.payment_method_last4,
            "account_status": customer.account_status,
        }

    def _tool_retrieve_policy(self, query: str) -> dict:
        needle = query.strip().lower()
        hits = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "body": doc.body,
                "published": doc.published,
            }
            for doc in self.state.policy_docs.values()
            if needle in doc.title.lower()
            or needle in doc.body.lower()
            or any(needle in topic for topic in doc.topics)
        ]
        return {"status": "OK", "documents": hits}

    def _tool_schedule_payment(self, plan_id: str, next_due: str) -> dict:
        plan = self.state.payment_plans.get(plan_id)
        if not plan:
            return {"error": f"Payment plan '{plan_id}' not found."}
        plan.next_due = next_due
        return {
            "status": "SCHEDULED",
            "plan_id": plan.plan_id,
            "next_due": plan.next_due,
            "_unsafe_note": "No ownership or plan-status check",
        }

    def _tool_process_refund(
        self,
        order_id: str,
        amount: float,
        reason: str,
    ) -> dict:
        order = self.state.orders.get(order_id)
        if not order:
            return {"error": f"Order '{order_id}' not found."}
        if amount <= 0:
            return {"error": "Refund amount must be greater than zero."}

        applied = min(amount, order.remaining_to_pay)
        order.remaining_to_pay -= applied
        if order.remaining_to_pay == 0:
            order.status = "refunded"
        refund = Refund(
            refund_id=new_short_id("REF"),
            order_id=order.order_id,
            customer_id=order.customer_id,
            amount=amount,
            reason=reason,
            status="COMPLETED",
            timestamp=utc_now(),
        )
        self.state.refunds.append(refund)
        return {
            "status": "COMPLETED",
            "refund_id": refund.refund_id,
            "order_id": order.order_id,
            "amount": amount,
            "remaining_to_pay": order.remaining_to_pay,
            "_unsafe_note": "NO eligibility, NO remaining-balance cap, NO HITL",
        }

    def _tool_escalate_to_human(self, topic: str, reason: str) -> dict:
        escalation = Escalation(
            escalation_id=new_short_id("ESC"),
            customer_id=self.state.authenticated_customer_id,
            topic=topic.strip().lower(),
            reason=reason,
            status="skipped",
            timestamp=utc_now(),
        )
        self.state.escalations.append(escalation)
        return {
            "status": "SKIPPED",
            "escalation_id": escalation.escalation_id,
            "_unsafe_note": "Escalation is optional in red-team mode",
        }

    def _tool_get_klarna_state_summary(self) -> dict:
        from mini_agents.domains.klarna.tools import (
            KlarnaToolExecutor as SafeExecutor,
        )

        return SafeExecutor._tool_get_klarna_state_summary(self)
