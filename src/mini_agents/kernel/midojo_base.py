"""Generate ``midojo-base.json``: each domain's ledger as a MiDojo environment.

The consumer compiles a MiDojo suite from this file: ``environment`` is the
default ledger, ``seed_slots`` name the text fields a probe placeholder can be
written into, and ``tool_access`` says which tools change the ledger. Run
``scripts/gen_midojo_base.py`` after any change to a state, a slot or a tool's
access entry. ``tests/test_midojo_base.py`` fails while the committed file differs.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

from mini_agents.kernel import mcp_server

BASE_SCHEMA_VERSION = "mini-agents-midojo-base-v1"
BASE_PATH = Path(__file__).resolve().parent.parent / "midojo-base.json"
BASE_SESSION_ID = "MIDOJO00"

# MiDojo rewrites every string in the environment that matches this; ledger
# text that does would silently become empty.
PLACEHOLDER = re.compile(r"\{([A-Za-z_]\w*):([A-Za-z_]\w*)\}")


def _module(domain: str, name: str) -> Any:
    return importlib.import_module(f"mini_agents.domains.{domain}.{name}")


def _refuse_placeholders(domain: str, value: Any, path: str) -> None:
    if isinstance(value, str):
        if PLACEHOLDER.search(value):
            raise ValueError(
                f"{domain}: ledger text at {path} looks like a probe placeholder"
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _refuse_placeholders(domain, item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _refuse_placeholders(domain, item, f"{path}[{index}]")


def _target(domain: str) -> dict[str, Any]:
    seeds = _module(domain, "seeds")
    state = seeds.new_state()
    state.session_id = BASE_SESSION_ID
    environment = state.to_environment()
    _refuse_placeholders(domain, environment, "environment")
    return {
        "environment": environment,
        "seed_slots": [
            {
                "id": slot.id,
                "carrier_operation": slot.carrier_operation,
                "container": slot.container,
                "attribute": slot.attribute,
                "records": list(slot.records),
            }
            for slot in seeds.SLOTS
        ],
        "tool_access": dict(sorted(seeds.TOOL_ACCESS.items())),
    }


def build_base() -> dict[str, Any]:
    return {
        "schema_version": BASE_SCHEMA_VERSION,
        "targets": {domain: _target(domain) for domain in mcp_server.DOMAINS},
    }


def render_base() -> str:
    return json.dumps(build_base(), indent=2) + "\n"
