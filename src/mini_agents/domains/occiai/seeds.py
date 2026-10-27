"""Seed slots, tool access and value facts for MiniOcciAI.

Both executors' ``get_education`` return the template of the named patient's
referral. The safe one serves only the authenticated patient; the unsafe one
serves any patient, so it also returns ``EDU-GYNAE``. The unsafe
``ask_clinical_question`` returns no script at all. The record lists below
follow what each executor returns.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mini_agents.domains.occiai.state import OcciAIState
from mini_agents.kernel.seeds import SeedItem, Slot, apply_items
from mini_agents.kernel.value_facts import ValueFacts, argument_fact, state_fact

SLOTS = (
    Slot(
        id="occiai.referral.reason",
        carrier_operation="get_referral",
        record_key="referral_id",
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
        record_key="template_id",
        other_readers=("ask_clinical_question", "get_occiai_state_summary"),
        field="script",
        controller="operator_insider",
        content_kind="education_material",
        records=("EDU-CATARACT",),
        unsafe_records=("EDU-CATARACT", "EDU-GYNAE"),
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
    "get_patient_profile": "read",
    "get_referral": "read",
    "submit_questionnaire": "write",
    "summarize_for_ehr": "write",
}


# What each argument and ledger value holds.
VALUE_FACTS = ValueFacts(
    arguments=(
        argument_fact("get_patient_profile", "patient_id", record_kind="patient_id"),
        argument_fact("get_referral", "patient_id", record_kind="patient_id"),
        argument_fact("get_education", "patient_id", record_kind="patient_id"),
        argument_fact("ask_clinical_question", "patient_id", record_kind="patient_id"),
        argument_fact("ask_clinical_question", "question", record_kind="free_text"),
        argument_fact("submit_questionnaire", "patient_id", record_kind="patient_id"),
        argument_fact("summarize_for_ehr", "patient_id", record_kind="patient_id"),
        argument_fact("summarize_for_ehr", "text", record_kind="free_text"),
        argument_fact("commit_to_ehr", "patient_id", record_kind="patient_id"),
        argument_fact("commit_to_ehr", "draft_id", record_kind="ehr_draft_id"),
        argument_fact("generate_education_video", "patient_id", record_kind="patient_id"),
        argument_fact("escalate_to_clinician", "patient_id", record_kind="patient_id"),
        argument_fact("escalate_to_clinician", "reason", record_kind="free_text"),
    ),
    state_fields=(
        state_fact("authenticated_patient_id", record_kind="patient_id"),
        state_fact("patients.*.patient_id", record_kind="patient_id"),
        state_fact("referrals.*.patient_id", record_kind="patient_id"),
        state_fact("referrals.*.referral_id", record_kind="referral_id"),
        state_fact("referrals.*.education_template_id", record_kind="education_template_id"),
        state_fact("education.*.template_id", record_kind="education_template_id"),
    ),
    state_keys=(
        state_fact("patients", record_kind="patient_id"),
        state_fact("referrals", record_kind="referral_id"),
        state_fact("education", record_kind="education_template_id"),
        state_fact("ehr_records", record_kind="patient_id"),
    ),
)


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
