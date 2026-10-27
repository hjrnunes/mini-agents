"""Value facts: the record kind or date format a domain declares for a value.

A consumer reads them from ``value-facts.json`` beside each variant's
``tools-list.json``: ``arguments`` declares what one argument of one tool
takes, ``state_fields`` the value at a state path, ``state_keys`` the keys of
the mapping at one. A state path is dotted, relative to the ledger the state
summary returns, and a ``*`` segment stands for any one key.

A record kind is a lower_snake label, and two different kinds never share a
value. A format is ``date`` or ``date-time`` (RFC 3339). A value with no
declaration has no fact: free-text categories such as a ``topic`` and ids that
name more than one kind of record stay undeclared.

Each domain keeps its facts as ``VALUE_FACTS`` in ``domains/<name>/seeds.py``;
both variants share them, since they share one tool schema and one state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

FORMATS = ("date", "date-time")


@dataclass(frozen=True)
class Declared:
    """A record kind, a format, or both."""

    record_kind: Optional[str] = None
    format: Optional[str] = None

    def __post_init__(self) -> None:
        if self.record_kind is None and self.format is None:
            raise ValueError("a value fact declares a record_kind, a format, or both")
        if self.format is not None and self.format not in FORMATS:
            raise ValueError(f"unknown value format: {self.format}")

    def declared(self) -> dict[str, str]:
        declared = {"record_kind": self.record_kind, "format": self.format}
        return {key: value for key, value in declared.items() if value is not None}


@dataclass(frozen=True)
class ArgumentFact(Declared):
    """The values one argument of one tool takes."""

    operation_id: str = ""
    argument: str = ""

    def to_json(self) -> dict[str, str]:
        return {
            "operation_id": self.operation_id,
            "argument": self.argument,
            **self.declared(),
        }


@dataclass(frozen=True)
class StateFact(Declared):
    """The values at a state path, or the keys of the mapping at it."""

    path: str = ""

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, **self.declared()}


def argument_fact(operation_id: str, name: str, **declared: str) -> ArgumentFact:
    return ArgumentFact(operation_id=operation_id, argument=name, **declared)


def state_fact(path: str, **declared: str) -> StateFact:
    return StateFact(path=path, **declared)


@dataclass(frozen=True)
class ValueFacts:
    arguments: tuple[ArgumentFact, ...] = ()
    state_fields: tuple[StateFact, ...] = ()
    state_keys: tuple[StateFact, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "arguments": [fact.to_json() for fact in self.arguments],
            "state_fields": [fact.to_json() for fact in self.state_fields],
            "state_keys": [fact.to_json() for fact in self.state_keys],
        }
