from __future__ import annotations

from typing import Any, Protocol


class ToolExecutor(Protocol):
    """Domain tool runner. Rules live here, not in the system prompt."""

    def execute(self, tool_name: str, args: dict[str, Any]) -> dict: ...
