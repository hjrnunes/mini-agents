from __future__ import annotations

from typing import Any, Protocol


class ToolExecutor(Protocol):
    """Domain tool runner. Rules live here, not in the system prompt."""

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict: ...


class ToolDispatch:
    """Base of the domain executors: run ``_tool_<name>`` and return its dict.

    The method is resolved at call time, so a name with no ``_tool_`` method
    returns an error dict instead of failing at import. Any ``TypeError`` the
    call raises becomes a "Bad arguments" error dict, including one raised
    inside the tool body.
    """

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict:
        fn = getattr(self, f"_tool_{tool_name}", None)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool_name}: {exc}"}
