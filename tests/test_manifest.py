"""The committed tool manifest matches the servers and the seed slot tables."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from mini_agents.kernel import mcp_server
from mini_agents.kernel.manifest import (
    MANIFEST_PATH,
    ManifestError,
    build_manifest,
    render_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "mini_agents"
DOMAINS = mcp_server.DOMAINS


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _committed():
    return json.loads(MANIFEST_PATH.read_text())


def test_the_manifest_lives_inside_the_package():
    assert MANIFEST_PATH == PACKAGE / "tool-manifest.json"


def test_the_generator_reproduces_the_committed_file_byte_for_byte():
    assert render_manifest() == MANIFEST_PATH.read_text()


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_manifest", ROOT / "scripts" / "gen_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_check_passes_on_the_committed_file():
    assert _script().main(["--check"]) == 0


def test_the_script_runs_as_a_command():
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_manifest.py"), "--check"],
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr


def test_the_script_check_fails_when_the_file_is_stale(tmp_path, capsys):
    stale = tmp_path / "stale.json"
    stale.write_text("{}\n")

    assert _script().main(["--check", "--path", str(stale)]) == 1
    assert "gen_manifest.py" in capsys.readouterr().err


def test_the_script_check_fails_when_the_file_is_missing(tmp_path):
    assert _script().main(["--check", "--path", str(tmp_path / "none.json")]) == 1


def test_the_script_writes_the_generated_file(tmp_path):
    target = tmp_path / "out.json"

    assert _script().main(["--path", str(target)]) == 0
    assert target.read_text() == render_manifest()


def test_the_script_reports_a_generator_error_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    script = _script()

    def broken():
        raise ManifestError("klarna: tool x is registered but has no TOOL_ACCESS entry")

    monkeypatch.setattr(script, "render_manifest", broken)

    assert script.main(["--path", str(tmp_path / "out.json")]) == 1
    assert "tool x" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_the_manifest_validates_against_its_schema():
    schema = json.loads((PACKAGE / "tool-manifest.schema.json").read_text())

    jsonschema.Draft202012Validator(schema).validate(_committed())


def test_the_manifest_covers_the_three_targets():
    assert list(_committed()["targets"]) == list(DOMAINS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_registered_tool_is_in_the_manifest_and_nothing_else(domain):
    server = mcp_server.BUILDERS[domain]()
    registered = {tool.name for tool in asyncio.run(server.list_tools())}

    listed = {tool["name"] for tool in _committed()["targets"][domain]["tools"]}

    assert listed == registered


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_manifest_tool_dispatches_in_both_executors(domain):
    safe = importlib.import_module(f"mini_agents.domains.{domain}.tools")
    unsafe = importlib.import_module(f"mini_agents.domains.{domain}.tools_unsafe")
    safe_class = next(v for k, v in vars(safe).items() if k.endswith("ToolExecutor"))
    unsafe_class = next(v for k, v in vars(unsafe).items() if k.endswith("ToolExecutor"))

    for tool in _committed()["targets"][domain]["tools"]:
        assert hasattr(safe_class, f"_tool_{tool['name']}"), tool["name"]
        assert hasattr(unsafe_class, f"_tool_{tool['name']}"), tool["name"]


def test_a_registered_tool_without_an_access_entry_stops_the_generator(monkeypatch):
    real = mcp_server.BUILDERS["klarna"]

    def with_extra_tool(state=None):
        server = real(state)

        @server.tool()
        def new_tool() -> str:
            return "{}"

        return server

    monkeypatch.setitem(mcp_server.BUILDERS, "klarna", with_extra_tool)

    with pytest.raises(ManifestError, match="new_tool"):
        build_manifest()


def test_an_access_entry_without_a_registered_tool_stops_the_generator(monkeypatch):
    seeds = _seeds("klarna")
    monkeypatch.setitem(seeds.TOOL_ACCESS, "ghost_tool", "read")

    with pytest.raises(ManifestError, match="ghost_tool"):
        build_manifest()


def test_an_access_value_other_than_read_or_write_stops_the_generator(monkeypatch):
    monkeypatch.setitem(_seeds("klarna").TOOL_ACCESS, "lookup_order", "readwrite")

    with pytest.raises(ManifestError, match="lookup_order"):
        build_manifest()


@pytest.mark.parametrize("domain", DOMAINS)
def test_seed_slots_in_the_manifest_are_the_slot_table(domain):
    slots = _seeds(domain).SLOTS

    listed = _committed()["targets"][domain]["seed_slots"]

    assert [s["id"] for s in listed] == [slot.id for slot in slots]
    for entry, slot in zip(listed, slots):
        assert entry == {
            "id": slot.id,
            "carrier_operation": slot.carrier_operation,
            "field": slot.field,
            "controller": slot.controller,
            "content_kind": slot.content_kind,
            "records": list(slot.records),
            "unsafe_records": list(slot.unsafe_records),
        }


@pytest.mark.parametrize("domain", DOMAINS)
def test_returns_planted_slots_lists_every_slot_a_tool_can_return(domain):
    target = _committed()["targets"][domain]
    expected = {}
    for slot in _seeds(domain).SLOTS:
        for tool in (slot.carrier_operation, *slot.other_readers):
            expected.setdefault(tool, []).append(slot.id)

    for tool in target["tools"]:
        assert tool["returns_planted_slots"] == expected.get(tool["name"], [])


def test_a_slot_carrier_is_always_a_read_tool():
    for domain in DOMAINS:
        access = {t["name"]: t["access"] for t in _committed()["targets"][domain]["tools"]}
        for slot in _committed()["targets"][domain]["seed_slots"]:
            assert access[slot["carrier_operation"]] == "read"
