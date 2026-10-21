"""Turn state in ``<state dir>/state.json``: the processes and each turn.

The first turn writes it; every later turn reads it, appends its turn, and
writes it back. It never lives in the workdir: MiDojo observes every new
workdir file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"
VERSION = 1


@dataclass
class Process:
    """One detached server: its PID, its loopback port, how it was started.

    ``identity`` tells this process from a later one that reuses its PID (see
    ``processes.process_identity``); ``None`` when it could not be read.
    """

    pid: int
    port: int
    command: list[str]
    log: str
    identity: str | None = None


@dataclass
class Turn:
    """What one turn produced. ``exit_reason`` is ``ok`` or a named failure.

    ``index`` is ``None`` only for a rejected run whose ``MIDOJO_TURN_INDEX``
    is not an integer.
    """

    index: int | None
    reply: str | None
    response_id: str | None
    served_model: str | None
    gateway_requests: int
    exit_reason: str
    failure: str | None = None


@dataclass
class AgentState:
    """``turns`` holds the runs accepted as turns, in order; ``rejected`` the
    runs refused before they became one (a bad turn variable, a turn out of
    order), which never shift the next turn's expected index. ``turn_count``
    is ``None`` only in a summary of a rejected run whose count is unreadable."""

    domain: str
    turn_count: int | None
    mcp: Process | None = None
    ogx: Process | None = None
    previous_response_id: str | None = None
    turns: list[Turn] = field(default_factory=list)
    rejected: list[Turn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        if data.get("version") != VERSION:
            raise ValueError(f"unsupported state version: {data.get('version')!r}")
        return cls(
            domain=data["domain"],
            turn_count=data["turn_count"],
            mcp=Process(**data["mcp"]) if data.get("mcp") else None,
            ogx=Process(**data["ogx"]) if data.get("ogx") else None,
            previous_response_id=data.get("previous_response_id"),
            turns=[Turn(**turn) for turn in data.get("turns", [])],
            rejected=[Turn(**turn) for turn in data.get("rejected", [])],
        )


def state_path(state_dir: Path) -> Path:
    return Path(state_dir) / STATE_FILE


def load_state(state_dir: Path) -> AgentState | None:
    """The saved state, or ``None`` when no turn has saved one."""

    path = state_path(state_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return AgentState.from_dict(json.loads(text))


def save_state(state: AgentState, state_dir: Path) -> Path:
    """Write the state atomically, so a cut-off turn leaves the last good one."""

    path = state_path(state_dir)
    write_json_atomic(path, state.to_dict())
    return path


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
