"""The sandbox policy template is default-deny with one egress rule per
interpreter copy, and its two port slots fill with integers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agents.kernel import sandbox_policy
from mini_agents.kernel.sandbox_policy import (
    CONTROL_PLANE_PORT,
    POLICY_PATH,
    RELAY_PORT,
    fill_policy,
    load_template,
)

ROOT = Path(__file__).resolve().parent.parent
HOST = "host.openshell.internal"
VBIN = "/opt/app-root/.venv/bin"


def test_the_template_lives_inside_the_package():
    assert POLICY_PATH == ROOT / "src" / "mini_agents" / "sandbox-policy.json"


def test_the_template_has_the_agreed_shape():
    assert load_template() == {
        "version": 1,
        "filesystem": {
            "includeWorkdir": True,
            "readOnly": [
                "/usr",
                "/lib",
                "/proc",
                "/dev/urandom",
                "/etc",
                "/var/log",
                "/opt",
            ],
            "readWrite": ["/sandbox", "/tmp", "/dev/null", "/dev/shm"],
        },
        "landlock": {"compatibility": "best_effort"},
        "process": {"runAsUser": "1500", "runAsGroup": "1500"},
        "networkPolicies": {
            "control_plane": {
                "endpoints": [{"host": HOST, "port": "<control_plane_port>"}],
                "binaries": [{"path": f"{VBIN}/python-mcp"}],
            },
            "model_relay": {
                "endpoints": [{"host": HOST, "port": "<relay_port>"}],
                "binaries": [{"path": f"{VBIN}/python-ogx"}],
            },
        },
    }


def test_only_the_two_interpreter_copies_get_egress():
    rules = load_template()["networkPolicies"].values()

    assert sorted(b["path"] for rule in rules for b in rule["binaries"]) == [
        f"{VBIN}/python-mcp",
        f"{VBIN}/python-ogx",
    ]


def test_filling_sets_each_port_as_an_integer_and_leaves_the_template_alone():
    template = load_template()

    policy = fill_policy(template, control_plane_port=18007, relay_port=18006)

    rules = policy["networkPolicies"]
    assert rules["control_plane"]["endpoints"] == [{"host": HOST, "port": 18007}]
    assert rules["model_relay"]["endpoints"] == [{"host": HOST, "port": 18006}]
    assert template == load_template()
    assert {**policy, "networkPolicies": None} == {
        **template,
        "networkPolicies": None,
    }


def test_filling_reads_the_committed_template_by_default():
    assert fill_policy(control_plane_port=1, relay_port=2) == fill_policy(
        load_template(), control_plane_port=1, relay_port=2
    )


@pytest.mark.parametrize("port", [0, 65536, -1, "18007", 18007.0, True, None])
def test_filling_refuses_a_port_that_is_not_an_integer_in_range(port):
    with pytest.raises(ValueError, match=r"<relay_port>: port must be an integer"):
        fill_policy(control_plane_port=18007, relay_port=port)


@pytest.mark.parametrize("slot", [CONTROL_PLANE_PORT, RELAY_PORT])
def test_filling_refuses_a_template_without_a_slot(slot, tmp_path, monkeypatch):
    text = POLICY_PATH.read_text().replace(f'"{slot}"', "1")
    template = tmp_path / "sandbox-policy.json"
    template.write_text(text)
    monkeypatch.setattr(sandbox_policy, "POLICY_PATH", template)

    with pytest.raises(ValueError, match=f"template lacks port slots: \\['{slot}'\\]"):
        fill_policy(control_plane_port=1, relay_port=2)
