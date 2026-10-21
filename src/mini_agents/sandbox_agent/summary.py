"""Evidence in ``<workdir>/.asago``, which MiDojo carries out of the sandbox.

- ``mcp-calls.jsonl``: the server's ``--midojo-log``, written there directly
  from the first call on;
- ``ogx.log``: OGX's output, copied there after every turn;
- ``run-summary.json``: rewritten after every run, failed and rejected ones
  included.

The summary is self-reported by code inside the sandbox; the logs let the
consumer's parsers cross-check it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mini_agents.sandbox_agent.state import AgentState, Turn, write_json_atomic

EVIDENCE_DIR = ".asago"
MCP_CALLS = "mcp-calls.jsonl"
OGX_LOG = "ogx.log"
RUN_SUMMARY = "run-summary.json"
SCHEMA = "mini-agents-run-summary-v1"


def evidence_dir(workdir: Path) -> Path:
    return Path(workdir) / EVIDENCE_DIR


def run_summary(state: AgentState) -> dict[str, Any]:
    """The run so far. ``final`` once the last turn of ``turn_count`` ran.

    ``served_model`` is the model the last answering response named.
    ``gateway_request_count`` counts the Responses requests sent to OGX.
    ``rejected`` lists the runs refused before they became a turn; they are
    not in ``turns`` or ``exit_reasons``.
    """

    turns = [asdict(turn) for turn in state.turns]
    served = [turn.served_model for turn in state.turns if turn.served_model]
    final = state.turn_count is not None and any(
        turn.index == state.turn_count - 1 for turn in state.turns
    )
    return {
        "schema": SCHEMA,
        "domain": state.domain,
        "turn_count": state.turn_count,
        "turns_run": len(turns),
        "final": final,
        "served_model": served[-1] if served else None,
        "gateway_request_count": sum(turn.gateway_requests for turn in state.turns),
        "exit_reasons": [turn.exit_reason for turn in state.turns],
        "turns": turns,
        "rejected": [asdict(turn) for turn in state.rejected],
    }


def summary_path(workdir: Path) -> Path:
    return evidence_dir(workdir) / RUN_SUMMARY


def write_run_summary(state: AgentState, workdir: Path) -> Path:
    path = summary_path(workdir)
    write_json_atomic(path, run_summary(state))
    return path


def record_rejection(
    workdir: Path, *, domain: str, turn_count: int | None, rejected: Turn
) -> Path:
    """Add a rejected run to the summary when there is no state to add it to.

    An earlier summary is kept and extended; without a readable one, the
    summary holds only this rejection.
    """

    path = summary_path(workdir)
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = None
    if not isinstance(summary, dict) or not isinstance(summary.get("rejected"), list):
        summary = run_summary(AgentState(domain=domain, turn_count=turn_count))
    summary["rejected"].append(asdict(rejected))
    write_json_atomic(path, summary)
    return path


def copy_ogx_log(source: Path, workdir: Path) -> Path | None:
    """Copy OGX's log into the evidence directory, when OGX wrote one."""

    if not source.is_file():
        return None
    destination = evidence_dir(workdir) / OGX_LOG
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
