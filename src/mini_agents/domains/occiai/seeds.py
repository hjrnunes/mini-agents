"""Seed slots and tool access for MiniOcciAI.

The two executors read education material differently. The safe ``get_education``
returns the template of the authenticated patient's referral; the unsafe one
always returns ``EDU-UNREVIEWED``. The unsafe ``ask_clinical_question`` returns
no script at all. The record lists below follow what each executor returns.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mini_agents.domains.occiai.state import OcciAIState
from mini_agents.kernel.seeds import SeedItem, Slot, apply_items

SLOTS = (
    Slot(
        id="occiai.referral.reason",
        carrier_operation="get_referral",
        other_readers=("get_occiai_state_summary",),
        field="reason",
        controller="third_party",
        content_kind="clinical_note",
        records=("REF-104",),
        unsafe_records=("REF-101", "REF-104", "REF-201"),
        container="referrals",
        attribute="reason",
    ),
    Slot(
        id="occiai.education.script",
        carrier_operation="get_education",
        other_readers=("ask_clinical_question", "get_occiai_state_summary"),
        field="script",
        controller="operator_insider",
        content_kind="education_material",
        records=("EDU-CATARACT",),
        unsafe_records=("EDU-UNREVIEWED",),
        container="education",
        attribute="script",
    ),
)

# A tool is "write" when it creates or changes a ledger record. Audit-log
# entries do not count.
TOOL_ACCESS = {
    "ask_clinical_question": "read",
    "commit_to_ehr": "write",
    "escalate_to_clinician": "write",
    "generate_education_video": "write",
    "get_education": "read",
    "get_occiai_state_summary": "read",
    "get_referral": "read",
    "submit_questionnaire": "write",
    "summarize_for_ehr": "write",
}


def new_state() -> OcciAIState:
    return OcciAIState()


def apply_seed(
    state: OcciAIState,
    items: Sequence[SeedItem],
    *,
    unsafe: bool = False,
    carrier: Optional[str] = None,
) -> list[str]:
    return apply_items(SLOTS, state, items, unsafe=unsafe, carrier=carrier)
