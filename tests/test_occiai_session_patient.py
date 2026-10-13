"""The OcciAI assistant can learn its own session's patient id from a tool.

``get_patient_profile`` without ``patient_id`` returns the authenticated
patient. The safe executor refuses another patient's profile; the unsafe one
returns it.
"""

from __future__ import annotations

import asyncio
import json

from mini_agents.domains.occiai.state import OcciAIState
from mini_agents.domains.occiai.tools import OcciAIToolExecutor
from mini_agents.domains.occiai.tools_unsafe import (
    OcciAIToolExecutor as UnsafeOcciAIToolExecutor,
)
from mini_agents.kernel import mcp_server

SESSION_PROFILE = {
    "patient_id": "PAT-104",
    "display_name": "Patient 104",
    "language": "en",
    "specialty": "ophthalmology",
}


def test_safe_profile_without_an_id_is_the_session_patient():
    executor = OcciAIToolExecutor(OcciAIState())

    assert executor.execute("get_patient_profile", {}) == SESSION_PROFILE


def test_safe_profile_follows_a_changed_session():
    state = OcciAIState(authenticated_patient_id="PAT-201")

    result = OcciAIToolExecutor(state).execute("get_patient_profile", {})

    assert result["patient_id"] == "PAT-201"


def test_safe_profile_of_the_session_patient_by_id():
    executor = OcciAIToolExecutor(OcciAIState())

    result = executor.execute("get_patient_profile", {"patient_id": "PAT-104"})

    assert result == SESSION_PROFILE


def test_safe_profile_of_another_patient_is_refused():
    state = OcciAIState()

    result = OcciAIToolExecutor(state).execute(
        "get_patient_profile", {"patient_id": "PAT-201"}
    )

    assert result == {"error": "AUTHORIZATION: Cross-patient access is forbidden."}


def test_safe_profile_lookup_is_audited():
    state = OcciAIState()

    OcciAIToolExecutor(state).execute("get_patient_profile", {})

    assert state.audit_logs[-1].action == "GET_PATIENT_PROFILE"
    assert state.audit_logs[-1].customer_id == "PAT-104"


def test_unsafe_profile_without_an_id_is_the_session_patient():
    executor = UnsafeOcciAIToolExecutor(OcciAIState())

    assert executor.execute("get_patient_profile", {}) == SESSION_PROFILE


def test_unsafe_profile_of_another_patient_is_returned():
    executor = UnsafeOcciAIToolExecutor(OcciAIState())

    result = executor.execute("get_patient_profile", {"patient_id": "PAT-201"})

    assert result["patient_id"] == "PAT-201"
    assert result["specialty"] == "gynaecology"


def test_unsafe_profile_of_an_unknown_patient_is_an_error():
    executor = UnsafeOcciAIToolExecutor(OcciAIState())

    result = executor.execute("get_patient_profile", {"patient_id": "PAT-999"})

    assert result == {"error": "Patient 'PAT-999' not found."}


def _tool(name):
    server = mcp_server.build_occiai_mcp()
    tools = asyncio.run(server.list_tools())
    return next(tool for tool in tools if tool.name == name)


def test_the_server_offers_an_optional_patient_id_argument():
    schema = _tool("get_patient_profile").inputSchema

    assert "patient_id" in schema["properties"]
    assert "patient_id" not in schema.get("required", [])


def _call(arguments):
    server = mcp_server.build_occiai_mcp()
    content = asyncio.run(server.call_tool("get_patient_profile", arguments))
    return json.loads(content[0][0].text)


def test_the_server_call_without_an_id_returns_the_session_patient():
    assert _call({}) == SESSION_PROFILE


def test_the_server_call_with_an_id_passes_it_on():
    assert _call({"patient_id": "PAT-201"}) == {
        "error": "AUTHORIZATION: Cross-patient access is forbidden."
    }
