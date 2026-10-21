"""The OpenShell policy for the sandbox image built from ``Dockerfile.sandbox``.

``sandbox-policy.json`` is a template in OpenShell's camelCase proto-JSON
(``SandboxSpec.policy``). Network is default-deny with two rules, one per
interpreter copy, because OpenShell matches the resolved executable:

- ``control_plane``: ``/opt/app-root/.venv/bin/python-mcp`` (the domain's MCP
  server) may reach ``host.openshell.internal:<control_plane_port>``;
- ``model_relay``: ``/opt/app-root/.venv/bin/python-ogx`` (OGX) may reach
  ``host.openshell.internal:<relay_port>``.

A port slot is a JSON string exactly equal to its name; the consumer replaces
it with the port as an integer. ``fill_policy`` does that and refuses a
template that lacks either slot. ``scripts/gen_midojo_base.py`` copies the
template verbatim into the ``runtime`` block of ``midojo-base.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).resolve().parent.parent / "sandbox-policy.json"
CONTROL_PLANE_PORT = "<control_plane_port>"
RELAY_PORT = "<relay_port>"
PORT_SLOTS = (CONTROL_PLANE_PORT, RELAY_PORT)


def load_template() -> dict[str, Any]:
    return json.loads(POLICY_PATH.read_text())


def _fill(value: Any, ports: dict[str, int]) -> Any:
    if isinstance(value, str) and value in ports:
        return ports[value]
    if isinstance(value, dict):
        return {key: _fill(item, ports) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item, ports) for item in value]
    return value


def _slots(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} & set(PORT_SLOTS)
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return set().union(*(_slots(item) for item in value))
    return set()


def fill_policy(
    template: dict[str, Any] | None = None,
    *,
    control_plane_port: int,
    relay_port: int,
) -> dict[str, Any]:
    """Return a copy of ``template`` (default: the committed one) with both ports set."""
    ports = {CONTROL_PLANE_PORT: control_plane_port, RELAY_PORT: relay_port}
    for slot, port in ports.items():
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError(f"{slot}: port must be an integer in 1..65535, got {port!r}")
    template = load_template() if template is None else template
    missing = [slot for slot in PORT_SLOTS if slot not in _slots(template)]
    if missing:
        raise ValueError(f"template lacks port slots: {missing}")
    return _fill(template, ports)
