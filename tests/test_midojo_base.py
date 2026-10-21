"""The committed midojo-base.json matches the states, the seed slot tables and
the sandbox policy template."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from mini_agents.kernel import mcp_server, midojo_base, sandbox_policy
from mini_agents.kernel.midojo_base import (
    BASE_PATH,
    BASE_SESSION_ID,
    PLACEHOLDER,
    build_base,
    render_base,
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "mini_agents"
DOMAINS = mcp_server.DOMAINS


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _committed():
    return json.loads(BASE_PATH.read_text())


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_midojo_base", ROOT / "scripts" / "gen_midojo_base.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_the_base_lives_inside_the_package():
    assert BASE_PATH == PACKAGE / "midojo-base.json"


def test_the_generator_reproduces_the_committed_file_byte_for_byte():
    assert render_base() == BASE_PATH.read_text()


def test_the_check_flag_passes_on_the_committed_file():
    assert _script().main(["--check"]) == 0


def test_the_check_flag_fails_when_the_file_is_stale(tmp_path, capsys):
    stale = tmp_path / "stale.json"
    stale.write_text("{}\n")

    assert _script().main(["--check", "--path", str(stale)]) == 1
    assert "is stale; run scripts/gen_midojo_base.py" in capsys.readouterr().err


def test_the_script_writes_the_generated_file(tmp_path):
    target = tmp_path / "out.json"

    assert _script().main(["--path", str(target)]) == 0
    assert target.read_text() == render_base()


def test_the_file_validates_against_its_schema():
    schema = json.loads((PACKAGE / "midojo-base.schema.json").read_text())

    jsonschema.Draft202012Validator(schema).validate(_committed())


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_environment_uses_the_fixed_session_id(domain):
    assert (
        _committed()["targets"][domain]["environment"]["session_id"] == BASE_SESSION_ID
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_slot_path_names_a_text_field(domain):
    target = _committed()["targets"][domain]

    for slot in target["seed_slots"]:
        for record in slot["records"]:
            value = target["environment"][slot["container"]][record][slot["attribute"]]
            assert isinstance(value, str)


@pytest.mark.parametrize("domain", DOMAINS)
def test_the_slots_and_tool_access_match_the_domain_seeds(domain):
    target = _committed()["targets"][domain]
    seeds = _seeds(domain)

    assert target["seed_slots"] == [
        {
            "id": slot.id,
            "carrier_operation": slot.carrier_operation,
            "container": slot.container,
            "attribute": slot.attribute,
            "records": list(slot.records),
        }
        for slot in seeds.SLOTS
    ]
    assert target["tool_access"] == seeds.TOOL_ACCESS


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_ledger_text_looks_like_a_probe_placeholder(domain):
    environment = _committed()["targets"][domain]["environment"]

    assert [s for s in _strings(environment) if PLACEHOLDER.search(s)] == []


def test_a_ledger_string_that_looks_like_a_placeholder_stops_the_generator(monkeypatch):
    slot = _seeds("klarna").SLOTS[0]
    real = _seeds("klarna").new_state

    def poisoned():
        state = real()
        getattr(state, slot.container)[
            slot.records[0]
        ].item = "see {asago_injection_task:seed_0}"
        return state

    monkeypatch.setattr(_seeds("klarna"), "new_state", poisoned)

    with pytest.raises(ValueError, match=r"klarna.*environment\.orders\.ORD-101\.item"):
        build_base()
    assert midojo_base.PLACEHOLDER.pattern == r"\{([A-Za-z_]\w*):([A-Za-z_]\w*)\}"


def test_the_runtime_block_names_the_image_slot_the_agent_command_and_the_policy():
    assert _committed()["runtime"] == {
        "image": "<image_id>",
        "agent_command": ["mini-agents-sandbox-agent", "--domain", "<domain>"],
        "policy": sandbox_policy.load_template(),
    }


def test_the_runtime_policy_comes_from_the_template_file(tmp_path, monkeypatch):
    template = tmp_path / "sandbox-policy.json"
    template.write_text(json.dumps({"version": 1, "marker": "from the template"}))
    monkeypatch.setattr(sandbox_policy, "POLICY_PATH", template)

    assert build_base()["runtime"]["policy"] == {
        "version": 1,
        "marker": "from the template",
    }


@pytest.mark.parametrize(
    "path, value",
    [
        (("runtime", "extra"), 1),
        (("runtime", "policy", "extra"), 1),
        (("runtime", "policy", "networkPolicies", "anything"), {}),
        (("runtime", "policy", "process", "runAsUser"), "sandbox"),
        (("runtime", "image"), "asago-mini-agents-sandbox@sha256:0"),
    ],
)
def test_the_schema_refuses_a_changed_runtime_block(path, value):
    schema = json.loads((PACKAGE / "midojo-base.schema.json").read_text())
    base = _committed()
    *parents, key = path
    node = base
    for parent in parents:
        node = node[parent]
    node[key] = value

    assert not jsonschema.Draft202012Validator(schema).is_valid(base)
