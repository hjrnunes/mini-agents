"""Generate ``tool-manifest.json`` from the FastMCP builders and the slot tables.

Run ``scripts/gen_manifest.py`` after you add, remove or rename a tool or a seed
slot. ``tests/test_manifest.py`` fails while the committed file differs.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

from mini_agents.kernel import mcp_server

MANIFEST_SCHEMA_VERSION = "mini-agents-tool-manifest-v1"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "tool-manifest.json"
ACCESS_VALUES = ("read", "write")


class ManifestError(Exception):
    """The builders and the domain's access table disagree."""


def _seeds(domain: str) -> Any:
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _registered_tools(domain: str) -> set[str]:
    server = mcp_server.BUILDERS[domain]()
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _check_access(domain: str, registered: set[str], access: dict[str, str]) -> None:
    for name in sorted(registered - set(access)):
        raise ManifestError(f"{domain}: tool {name} is registered but has no TOOL_ACCESS entry")
    for name in sorted(set(access) - registered):
        raise ManifestError(f"{domain}: TOOL_ACCESS lists {name}, which no builder registers")
    for name, value in sorted(access.items()):
        if value not in ACCESS_VALUES:
            raise ManifestError(f"{domain}: tool {name} has access {value!r}")


def _returned_slots(slots: tuple) -> dict[str, list[str]]:
    returned: dict[str, list[str]] = {}
    for slot in slots:
        for tool in (slot.carrier_operation, *slot.other_readers):
            returned.setdefault(tool, []).append(slot.id)
    return returned


def _target(domain: str) -> dict[str, Any]:
    seeds = _seeds(domain)
    access = seeds.TOOL_ACCESS
    _check_access(domain, _registered_tools(domain), access)
    returned = _returned_slots(seeds.SLOTS)
    return {
        "tools": [
            {
                "name": name,
                "access": access[name],
                "returns_planted_slots": returned.get(name, []),
            }
            for name in sorted(access)
        ],
        "seed_slots": [
            {
                "id": slot.id,
                "carrier_operation": slot.carrier_operation,
                "field": slot.field,
                "controller": slot.controller,
                "content_kind": slot.content_kind,
                "records": list(slot.records),
                "unsafe_records": list(slot.unsafe_records),
            }
            for slot in seeds.SLOTS
        ],
    }


def build_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "targets": {domain: _target(domain) for domain in mcp_server.DOMAINS},
    }


def render_manifest() -> str:
    return json.dumps(build_manifest(), indent=2) + "\n"
