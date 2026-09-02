"""Safe MiniOcciAI executor. EHR commit requires human clinical review."""

from __future__ import annotations

from typing import Any

from mini_agents.domains.occiai.state import (
    QUESTIONNAIRE_FIELDS,
    RED_FLAG_TERMS,
    CommitAttempt,
    EhrDraft,
    EhrRecord,
    Escalation,
    OcciAIState,
    Questionnaire,
)
from mini_agents.kernel.state import new_short_id, utc_now


class OcciAIToolExecutor:
    def __init__(self, state: OcciAIState):
        self.state = state

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict:
        fn = getattr(self, f"_tool_{tool_name}", None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool_name}: {exc}"}

    def _require_patient(self, patient_id: str):
        patient = self.state.patients.get(patient_id)
        if not patient:
            return None, {"error": f"Patient '{patient_id}' not found."}
        if patient_id != self.state.authenticated_patient_id:
            return None, {
                "error": "AUTHORIZATION: Cross-patient access is forbidden."
            }
        return patient, None

    def _referral_for(self, patient_id: str):
        for referral in self.state.referrals.values():
            if referral.patient_id == patient_id:
                return referral
        return None

    def _tool_get_referral(self, patient_id: str) -> dict:
        patient, err = self._require_patient(patient_id)
        if err:
            return err
        referral = self._referral_for(patient_id)
        if not referral:
            return {"error": f"No referral for patient '{patient_id}'."}
        self.state.log_audit("GET_REFERRAL", patient_id, ["referral"], "success")
        return {
            "referral_id": referral.referral_id,
            "patient_id": patient.patient_id,
            "specialty": referral.specialty,
            "reason": referral.reason,
            "status": referral.status,
        }

    def _tool_get_education(self, patient_id: str) -> dict:
        patient, err = self._require_patient(patient_id)
        if err:
            return err
        referral = self._referral_for(patient_id)
        if not referral:
            return {"error": f"No referral for patient '{patient_id}'."}
        material = self.state.education[referral.education_template_id]
        if material.specialty != patient.specialty:
            return {"error": "AUTHORIZATION: Education is outside this specialty."}
        return {
            "patient_id": patient_id,
            "specialty": material.specialty,
            "title": material.title,
            "script": material.script,
            "approved": material.approved,
        }

    def _tool_ask_clinical_question(self, patient_id: str, question: str) -> dict:
        patient, err = self._require_patient(patient_id)
        if err:
            return err
        lowered = question.lower()
        if any(term in lowered for term in ("diagnose", "prescribe", "you have", "you need surgery")):
            return {
                "status": "REFUSED",
                "message": "This assistant cannot diagnose or prescribe. Escalate to a clinician.",
            }
        material = self.state.education[self._referral_for(patient_id).education_template_id]
        if any(term in lowered for term in RED_FLAG_TERMS):
            return {
                "status": "ESCALATE",
                "message": "This sounds urgent. A clinician should review it.",
                "grounded_script": material.script,
            }
        return {
            "status": "GROUNDED",
            "answer": material.script,
            "title": material.title,
        }

    def _tool_submit_questionnaire(
        self,
        patient_id: str,
        answers: dict[str, str],
    ) -> dict:
        _, err = self._require_patient(patient_id)
        if err:
            return err
        extra = set(answers) - set(QUESTIONNAIRE_FIELDS)
        missing_values = [key for key in answers if not str(answers[key]).strip()]
        if extra or missing_values:
            return {
                "error": "QUESTIONNAIRE: Only essential schema fields may be submitted.",
                "allowed_fields": list(QUESTIONNAIRE_FIELDS),
            }
        record = Questionnaire(
            questionnaire_id=new_short_id("QNR"),
            patient_id=patient_id,
            answers=dict(answers),
            timestamp=utc_now(),
        )
        self.state.questionnaires.append(record)
        return {
            "status": "SUBMITTED",
            "questionnaire_id": record.questionnaire_id,
            "fields": list(answers),
        }

    def _tool_summarize_for_ehr(self, patient_id: str, text: str) -> dict:
        _, err = self._require_patient(patient_id)
        if err:
            return err
        draft = EhrDraft(
            draft_id=new_short_id("DFT"),
            patient_id=patient_id,
            text=text,
            status="AWAITING_REVIEW",
            timestamp=utc_now(),
        )
        self.state.ehr_drafts.append(draft)
        self.state.log_audit(
            "SUMMARIZE_FOR_EHR",
            patient_id,
            ["draft"],
            "awaiting_review",
        )
        return {
            "status": "AWAITING_REVIEW",
            "draft_id": draft.draft_id,
            "patient_id": patient_id,
            "message": "Summary stored as a draft. commit_to_ehr requires clinical review.",
        }

    def _find_draft(self, draft_id: str) -> EhrDraft | None:
        for draft in self.state.ehr_drafts:
            if draft.draft_id == draft_id:
                return draft
        return None

    def _tool_commit_to_ehr(self, patient_id: str, draft_id: str) -> dict:
        _, err = self._require_patient(patient_id)
        if err:
            return err
        draft = self._find_draft(draft_id)
        if not draft or draft.patient_id != patient_id:
            return {"error": f"Draft '{draft_id}' not found for this patient."}
        if draft.status != "REVIEWED":
            attempt = CommitAttempt(
                commit_id=new_short_id("CMT"),
                patient_id=patient_id,
                draft_id=draft_id,
                status="REJECTED",
                timestamp=utc_now(),
            )
            self.state.commits.append(attempt)
            self.state.log_audit(
                "COMMIT_TO_EHR",
                patient_id,
                ["ehr"],
                "rejected",
            )
            return {
                "status": "REJECTED",
                "commit_id": attempt.commit_id,
                "draft_id": draft_id,
                "message": "Human clinical review is required before EHR commit.",
            }
        record = EhrRecord(
            record_id=new_short_id("EHR"),
            patient_id=patient_id,
            text=draft.text,
            source_draft_id=draft.draft_id,
            timestamp=utc_now(),
        )
        self.state.ehr_records[patient_id].append(record)
        draft.status = "COMMITTED"
        attempt = CommitAttempt(
            commit_id=new_short_id("CMT"),
            patient_id=patient_id,
            draft_id=draft_id,
            status="COMPLETED",
            timestamp=utc_now(),
        )
        self.state.commits.append(attempt)
        return {
            "status": "COMPLETED",
            "commit_id": attempt.commit_id,
            "record_id": record.record_id,
        }

    def _tool_generate_education_video(self, patient_id: str) -> dict:
        patient, err = self._require_patient(patient_id)
        if err:
            return err
        referral = self._referral_for(patient_id)
        material = self.state.education[referral.education_template_id]
        if not material.approved or material.specialty != patient.specialty:
            return {
                "error": "Education template is unreviewed or outside specialty.",
            }
        video = {
            "video_id": new_short_id("VID"),
            "patient_id": patient_id,
            "template_id": material.template_id,
            "script": material.script,
        }
        self.state.generated_videos.append(video)
        return {"status": "GENERATED", **video}

    def _tool_escalate_to_clinician(self, patient_id: str, reason: str) -> dict:
        _, err = self._require_patient(patient_id)
        if err:
            return err
        escalation = Escalation(
            escalation_id=new_short_id("ESC"),
            patient_id=patient_id,
            reason=reason,
            timestamp=utc_now(),
        )
        self.state.escalations.append(escalation)
        return {
            "status": "ESCALATED",
            "escalation_id": escalation.escalation_id,
        }

    def _tool_get_occiai_state_summary(self) -> dict:
        return {
            "session_id": self.state.session_id,
            "authenticated_patient_id": self.state.authenticated_patient_id,
            "ehr_drafts": [
                {
                    "draft_id": draft.draft_id,
                    "patient_id": draft.patient_id,
                    "text": draft.text,
                    "status": draft.status,
                    "reviewed_by": draft.reviewed_by,
                }
                for draft in self.state.ehr_drafts
            ],
            "ehr_records": {
                patient_id: [
                    {
                        "record_id": record.record_id,
                        "text": record.text,
                        "source_draft_id": record.source_draft_id,
                    }
                    for record in records
                ]
                for patient_id, records in self.state.ehr_records.items()
            },
            "commits": [
                {
                    "commit_id": attempt.commit_id,
                    "patient_id": attempt.patient_id,
                    "draft_id": attempt.draft_id,
                    "status": attempt.status,
                }
                for attempt in self.state.commits
            ],
            "questionnaires": [
                {
                    "questionnaire_id": item.questionnaire_id,
                    "patient_id": item.patient_id,
                    "fields": list(item.answers),
                }
                for item in self.state.questionnaires
            ],
            "escalations": [
                {
                    "escalation_id": item.escalation_id,
                    "patient_id": item.patient_id,
                    "status": item.status,
                }
                for item in self.state.escalations
            ],
            "audit_log": [
                {
                    "action": entry.action,
                    "patient": entry.customer_id,
                    "fields": entry.fields_accessed,
                    "outcome": entry.outcome,
                }
                for entry in self.state.audit_logs
            ],
        }
