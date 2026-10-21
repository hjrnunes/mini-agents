"""Named reasons a turn fails with. Each one goes to stderr and the summary."""

from __future__ import annotations

INPUT_MISSING = "input-missing"
INPUT_INVALID = "input-invalid"
STATE_MISSING = "state-missing"
TURN_OUT_OF_ORDER = "turn-out-of-order"
START_FAILED = "process-start-failed"
GATEWAY_REQUEST_FAILED = "gateway-request-failed"
GATEWAY_RESPONSE_INVALID = "gateway-response-invalid"
GATEWAY_RESPONSE_FAILED = "gateway-response-failed"
STATE_WRITE_FAILED = "state-write-failed"
EVIDENCE_WRITE_FAILED = "evidence-write-failed"

# Per process ("mcp-server", "ogx"): see ``not_listening`` and friends.
MCP_SERVER = "mcp-server"
OGX = "ogx"


def not_listening(name: str) -> str:
    """The process is running but never listened within the timeout."""
    return f"{name}-not-listening"


def exited(name: str) -> str:
    """The process exited during the first turn, before it listened."""
    return f"{name}-exited"


def gone(name: str) -> str:
    """A later turn found the process from the first turn no longer running,
    or its PID now held by another process."""
    return f"{name}-gone"


class AgentFailure(Exception):
    """A turn failed for a named ``reason``.

    ``gateway_requests`` counts the Responses requests the turn sent before it
    failed, for the run summary.
    """

    def __init__(self, reason: str, detail: str, *, gateway_requests: int = 0):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.gateway_requests = gateway_requests
