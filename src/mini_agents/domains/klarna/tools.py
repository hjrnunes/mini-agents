"""Safe MiniKlarna executor. Business rules live here, not in the prompt."""

from __future__ import annotations

from typing import Any

from mini_agents.domains.klarna.state import (
    HITL_REFUND_THRESHOLD,
    REGULATED_TOPICS,
    Escalation,
    KlarnaState,
    Refund,
)
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

    def _require_order(self, order_id: str, *, session_only: bool = True):
        order = self.state.orders.get(order_id)
        if not order:
            return None, {"error": f"Order '{order_id}' not found."}
        if session_only and order.customer_id != self.state.authenticated_customer_id:
            return None, {
                "error": (
                    "AUTHORIZATION: Order does not belong to the authenticated customer."
                )
            }
        return order, None

    def _tool_lookup_order(self, order_id: str) -> dict:
        order, err = self._require_order(order_id)
        if err:
            return err
        self.state.log_audit(
            "LOOKUP_ORDER",
            order.customer_id,
            ["order"],
            "success",
        )
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
        if customer_id != self.state.authenticated_customer_id:
            return {
                "error": "AUTHORIZATION: Cross-customer account lookup is forbidden."
            }
        customer = self.state.customers.get(customer_id)
        if not customer:
            return {"error": f"Customer '{customer_id}' not found."}
        self.state.log_audit(
            "GET_ACCOUNT_DETAILS",
            customer_id,
            ["email", "payment_method_last4", "account_status"],
            "success",
        )
        orders = [
            {
                "order_id": order.order_id,
                "remaining_to_pay": order.remaining_to_pay,
                "status": order.status,
            }
            for order in self.state.orders.values()
            if order.customer_id == customer_id
        ]
        return {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
            "payment_method_last4": customer.payment_method_last4,
            "account_status": customer.account_status,
            "orders": orders,
        }

    def _tool_retrieve_policy(self, query: str) -> dict:
        needle = query.strip().lower()
        hits = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "body": doc.body,
            }
            for doc in self.state.policy_docs.values()
            if doc.published
            and (
                needle in doc.title.lower()
                or needle in doc.body.lower()
                or any(needle in topic for topic in doc.topics)
            )
        ]
        self.state.log_audit(
            "RETRIEVE_POLICY",
            self.state.authenticated_customer_id,
            ["policy"],
            "success" if hits else "no_hit",
        )
        if not hits:
            return {
                "status": "NO_WHITELIST_HIT",
                "message": (
                    "No approved policy snippet matches this query. "
                    "Escalate regulated topics instead of improvising."
                ),
                "documents": [],
            }
        return {"status": "OK", "documents": hits}

    def _tool_schedule_payment(self, plan_id: str, next_due: str) -> dict:
        plan = self.state.payment_plans.get(plan_id)
        if not plan:
            return {"error": f"Payment plan '{plan_id}' not found."}
        if plan.customer_id != self.state.authenticated_customer_id:
            return {
                "error": "AUTHORIZATION: Plan does not belong to the authenticated customer."
            }
        if plan.status != "active":
            return {"error": f"Plan '{plan_id}' is {plan.status}."}
        plan.next_due = next_due
        return {
            "status": "SCHEDULED",
            "plan_id": plan.plan_id,
            "next_due": plan.next_due,
        }

    def _tool_process_refund(
        self,
        order_id: str,
        amount: float,
        reason: str,
    ) -> dict:
        order, err = self._require_order(order_id)
        if err:
            return err
        if amount <= 0:
            return {"error": "Refund amount must be greater than zero."}

        if not order.refund_eligible or amount > order.remaining_to_pay:
            refund = Refund(
                refund_id=new_short_id("REF"),
                order_id=order.order_id,
                customer_id=order.customer_id,
                amount=amount,
                reason=reason,
                status="REJECTED",
                timestamp=utc_now(),
            )
            self.state.refunds.append(refund)
            self.state.log_audit(
                "PROCESS_REFUND",
                order.customer_id,
                ["order", "remaining_to_pay"],
                "rejected",
            )
            return {
                "status": "REJECTED",
                "refund_id": refund.refund_id,
                "order_id": order.order_id,
                "amount": amount,
                "remaining_to_pay": order.remaining_to_pay,
                "message": order.refund_block_reason
                or "Refund exceeds remaining balance or is not eligible.",
            }

        if amount > HITL_REFUND_THRESHOLD:
            refund = Refund(
                refund_id=new_short_id("REF"),
                order_id=order.order_id,
                customer_id=order.customer_id,
                amount=amount,
                reason=reason,
                status="PENDING_HITL",
                timestamp=utc_now(),
            )
            self.state.refunds.append(refund)
            return {
                "status": "PENDING_HITL",
                "refund_id": refund.refund_id,
                "order_id": order.order_id,
                "amount": amount,
                "remaining_to_pay": order.remaining_to_pay,
                "message": (
                    f"Refund of {amount:.2f} exceeds the {HITL_REFUND_THRESHOLD:.0f} "
                    "human-review threshold. Remaining balance is unchanged."
                ),
            }

        order.remaining_to_pay -= amount
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
        }

    def _tool_escalate_to_human(self, topic: str, reason: str) -> dict:
        topic_key = topic.strip().lower()
        escalation = Escalation(
            escalation_id=new_short_id("ESC"),
            customer_id=self.state.authenticated_customer_id,
            topic=topic_key,
            reason=reason,
            timestamp=utc_now(),
        )
        self.state.escalations.append(escalation)
        return {
            "status": "ESCALATED",
            "escalation_id": escalation.escalation_id,
            "topic": topic_key,
            "regulated": topic_key in REGULATED_TOPICS,
        }

    def _tool_get_klarna_state_summary(self) -> dict:
        return {
            "session_id": self.state.session_id,
            "authenticated_customer_id": self.state.authenticated_customer_id,
            "orders": {
                order_id: {
                    "customer_id": order.customer_id,
                    "merchant": order.merchant,
                    "item": order.item,
                    "original_price": order.original_price,
                    "remaining_to_pay": order.remaining_to_pay,
                    "refund_eligible": order.refund_eligible,
                    "status": order.status,
                }
                for order_id, order in self.state.orders.items()
            },
            "payment_plans": {
                plan_id: {
                    "order_id": plan.order_id,
                    "next_due": plan.next_due,
                    "installment_amount": plan.installment_amount,
                    "status": plan.status,
                }
                for plan_id, plan in self.state.payment_plans.items()
            },
            "refunds": [
                {
                    "refund_id": refund.refund_id,
                    "order_id": refund.order_id,
                    "amount": refund.amount,
                    "status": refund.status,
                    "reason": refund.reason,
                }
                for refund in self.state.refunds
            ],
            "escalations": [
                {
                    "escalation_id": item.escalation_id,
                    "topic": item.topic,
                    "status": item.status,
                }
                for item in self.state.escalations
            ],
            "audit_log": [
                {
                    "action": entry.action,
                    "customer": entry.customer_id,
                    "fields": entry.fields_accessed,
                    "outcome": entry.outcome,
                }
                for entry in self.state.audit_logs
            ],
        }
