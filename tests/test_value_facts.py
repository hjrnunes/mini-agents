"""Each variant ships value-facts.json, the domain's VALUE_FACTS as a scanner reads them.

Every fact must name what the variant exposes: an argument of a tool in its
``tools-list.json``, or a path of the ledger the state summary returns. A
value the domain cannot type stays undeclared: a free-text category such as
``topic``, or ``authenticated_actor_id``, which names a guest or a host.
"""

from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
import json
import re
from datetime import date
from pathlib import Path

import pytest

from mini_agents.kernel import declarations, mcp_server
from mini_agents.kernel.value_facts import ArgumentFact, StateFact

DOMAINS = mcp_server.DOMAINS
VARIANTS = declarations.VARIANTS
CASES = [(domain, variant) for domain in DOMAINS for variant in VARIANTS]
IDS = [f"{domain}-{variant}" for domain, variant in CASES]
ROOT = Path(__file__).resolve().parent.parent

# What a scanner's value-facts reader accepts: the document keys, each fact's
# keys, the record-kind pattern and the formats.
DOCUMENT_KEYS = {"arguments", "state_fields", "state_keys"}
DECLARED_KEYS = {"record_kind", "format"}
ARGUMENT_KEYS = {"operation_id", "argument"}
STATE_KEYS = {"path"}
RECORD_KIND = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
FORMATS = {"date", "date-time"}

UNDECLARED_ARGUMENTS = {"topic"}
UNDECLARED_STATE = {"authenticated_actor_id"}


def _seeds(domain):
    return importlib.import_module(f"mini_agents.domains.{domain}.seeds")


def _shipped_facts(domain, variant):
    path = declarations.declaration_dir(domain, variant) / declarations.VALUE_FACTS_FILE
    return json.loads(path.read_text())


def _shipped_tools(domain, variant):
    path = declarations.declaration_dir(domain, variant) / declarations.TOOLS_LIST_FILE
    return json.loads(path.read_text())


def _resolve(value, segments):
    """Every value at a dotted path, a ``*`` segment standing for any key."""
    if not segments:
        return [value]
    head, rest = segments[0], segments[1:]
    assert isinstance(value, dict), f"{head} is not inside a mapping"
    if head == "*":
        return [found for item in value.values() for found in _resolve(item, rest)]
    assert head in value, f"no {head}"
    return _resolve(value[head], rest)


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_file_ships_inside_the_package(domain, variant):
    shipped = importlib.resources.files("mini_agents") / "declarations" / domain / variant

    assert (shipped / "value-facts.json").is_file()


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_file_is_a_value_facts_document(domain, variant):
    facts = _shipped_facts(domain, variant)

    assert set(facts) == DOCUMENT_KEYS
    for name, keys in (
        ("arguments", ARGUMENT_KEYS),
        ("state_fields", STATE_KEYS),
        ("state_keys", STATE_KEYS),
    ):
        for fact in facts[name]:
            assert keys < set(fact) <= keys | DECLARED_KEYS, fact
            if "record_kind" in fact:
                assert RECORD_KIND.match(fact["record_kind"]), fact
            if "format" in fact:
                assert fact["format"] in FORMATS, fact
    pairs = [(fact["operation_id"], fact["argument"]) for fact in facts["arguments"]]
    assert len(pairs) == len(set(pairs))
    for name in ("state_fields", "state_keys"):
        paths = [fact["path"] for fact in facts[name]]
        assert len(paths) == len(set(paths)), name


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_the_file_is_the_domain_value_facts(domain, variant):
    assert _shipped_facts(domain, variant) == _seeds(domain).VALUE_FACTS.to_json()


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_argument_fact_names_an_argument_of_a_listed_tool(domain, variant):
    tools = {tool["name"]: tool for tool in _shipped_tools(domain, variant)["tools"]}

    for fact in _shipped_facts(domain, variant)["arguments"]:
        assert fact["operation_id"] in tools, fact
        properties = tools[fact["operation_id"]]["inputSchema"]["properties"]
        assert fact["argument"] in properties, fact


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_each_state_fact_names_a_ledger_path(domain, variant):
    ledger = _seeds(domain).new_state().to_environment()
    facts = _shipped_facts(domain, variant)

    for fact in facts["state_fields"]:
        values = _resolve(ledger, fact["path"].split("."))
        assert values, fact
        if fact.get("format") == "date":
            for value in values:
                date.fromisoformat(value)
    for fact in facts["state_keys"]:
        (mapping,) = _resolve(ledger, fact["path"].split("."))
        assert isinstance(mapping, dict) and mapping, fact


@pytest.mark.parametrize("domain, variant", CASES, ids=IDS)
def test_categories_and_mixed_ids_stay_undeclared(domain, variant):
    facts = _shipped_facts(domain, variant)

    assert not {fact["argument"] for fact in facts["arguments"]} & UNDECLARED_ARGUMENTS
    for name in ("state_fields", "state_keys"):
        names = {segment for fact in facts[name] for segment in fact["path"].split(".")}
        assert not names & UNDECLARED_STATE, name


def test_the_undeclared_values_exist():
    """Leaving them out means something only while the domains expose them."""
    arguments = {
        name
        for domain, variant in CASES
        for tool in _shipped_tools(domain, variant)["tools"]
        for name in tool["inputSchema"]["properties"]
    }
    ledger = _seeds("airbnb").new_state().to_environment()

    assert UNDECLARED_ARGUMENTS <= arguments
    assert UNDECLARED_STATE <= set(ledger)


def test_a_fact_declares_a_record_kind_or_a_format():
    with pytest.raises(ValueError):
        ArgumentFact(operation_id="lookup_order", argument="order_id")
    with pytest.raises(ValueError):
        StateFact(path="orders", format="time")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "gen_declarations", ROOT / "scripts" / "gen_declarations.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generator_check_fails_on_a_stale_value_facts_file(tmp_path, monkeypatch):
    generator = _generator()
    path = tmp_path / "klarna" / "safe" / declarations.VALUE_FACTS_FILE
    text = declarations.render_value_facts("klarna")
    monkeypatch.setattr(generator, "rendered", lambda: {path: text})
    path.parent.mkdir(parents=True)
    stale = json.loads(text)
    stale["arguments"].pop()
    path.write_text(json.dumps(stale, indent=2) + "\n")

    assert generator.main(["--check"]) == 1
    assert generator.main([]) == 0
    assert path.read_text() == text
    assert generator.main(["--check"]) == 0
