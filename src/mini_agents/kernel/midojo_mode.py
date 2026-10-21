"""MiDojo mode: the evaluation's ledger lives in MiDojo, not in this process.

A server started with ``--midojo-url`` binds the evaluation whose session
token a call carries (the ``X-Midojo-Session`` header, else
``MIDOJO_SESSION_TOKEN``). On that session's first call it loads the
evaluation's environment as its ledger; a new token loads again, so a second
evaluation never continues the first one's ledger. Before the tool runs it
starts the call in MiDojo under a fresh ``call_id``; a failed start fails the
call and the tool does not run. After every call, reads included (lookups
append audit-log entries), it pushes the whole ledger back and then reports the
call with the same ``call_id``, which finishes the started record. A call
cancelled or interrupted between its start and its report is not finished: it
stays unfinished in MiDojo.

The tool's own error is FastMCP's ``ToolError``: ``Tool.run`` wraps whatever
argument validation or the tool function raises in it, and an unknown tool
name raises it too. Such a call is reported with its error. Anything else that
escapes, a cancellation or a failure of the framework itself, leaves the call
unfinished and propagates.

The hook sits at the MCP ``call_tool`` layer. FastMCP registers its low-level
handler with ``validate_input=False`` and validates arguments inside
``Tool.run``, so a call rejected for a missing argument passes through here
and is reported with its error text.

With ``--midojo-log``, each call appends one JSON line: ``sequence``,
``call_id``, ``function``, and the outcome of ``load``, ``start``, ``push``
and ``report`` (``null`` when not attempted, or for ``load`` when the session
was already loaded; ``report`` is ``unfinished`` for a started call that never
reached its report), plus ``failure``. The request carries only the session
token.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from mcp import types
from mcp.server.fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from mini_agents.kernel.mcp_server import _TargetMCP

SESSION_HEADER = "X-Midojo-Session"
TOKEN_ENV = "MIDOJO_SESSION_TOKEN"

ENVIRONMENT_PATH = "/agent/environment"
FUNCTION_CALLS_PATH = "/agent/function-calls"
START_PATH = "/agent/function-calls/start"
CALL_ID_PREFIX = "mini-agents:"
TIMEOUT_SECONDS = 30.0
TOKEN_MISSING = "MiDojo session token missing"
ENVIRONMENT_UNAVAILABLE = "MiDojo environment unavailable"
START_FAILED = "MiDojo call start failed"
_BODY_EXCERPT = 200


class _Refused(Exception):
    """The control plane answered with a non-2xx status."""


class MidojoBinding:
    """Binds one server to the MiDojo evaluation of each call's session token."""

    def __init__(
        self,
        control_url: str,
        log_path: Path | None,
        *,
        http: httpx.AsyncClient | None = None,
    ):
        self.control_url = control_url.rstrip("/")
        self.log_path = log_path
        self._http = http or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
        self._token: str | None = None
        self._sequence = 0
        # Calls arrive concurrently; one at a time keeps load, push and report
        # in call order and stops a reload from overwriting a call's writes.
        self._lock = asyncio.Lock()

    async def call(
        self,
        server: _TargetMCP,
        name: str,
        arguments: dict[str, Any],
        invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        async with self._lock:
            self._sequence += 1
            line: dict[str, Any] = {
                "sequence": self._sequence,
                "call_id": None,
                "function": name,
                "load": None,
                "start": None,
                "push": None,
                "report": None,
                "failure": None,
            }
            try:
                return await self._call(server, name, arguments, invoke, line)
            finally:
                self._append(line)

    async def _call(
        self,
        server: _TargetMCP,
        name: str,
        arguments: dict[str, Any],
        invoke: Callable[[str, dict[str, Any]], Awaitable[Any]],
        line: dict[str, Any],
    ) -> Any:
        token = _session_token(server)
        if token is None:
            line["load"] = "no_token"
            line["failure"] = TOKEN_MISSING
            raise ToolError(TOKEN_MISSING)
        if token != self._token:
            try:
                await self._load(server, token)
            except Exception as exc:
                line["load"] = "failed"
                line["failure"] = _failure("GET", ENVIRONMENT_PATH, exc)
                raise ToolError(ENVIRONMENT_UNAVAILABLE) from exc
            self._token = token
            line["load"] = "ok"
        call_id = f"{CALL_ID_PREFIX}{uuid.uuid4()}"
        line["call_id"] = call_id
        start = {"call_id": call_id, "function": name, "args": arguments}
        try:
            await self._request("POST", START_PATH, token, start)
        except Exception as exc:
            line["start"] = "failed"
            line["failure"] = _failure("POST", START_PATH, exc)
            raise ToolError(START_FAILED) from exc
        line["start"] = "ok"
        # Until the report overwrites it: a cancellation, or any exception that
        # is not the tool's own error (a ToolError), leaves the call unfinished.
        line["report"] = "unfinished"
        try:
            result = await invoke(name, arguments)
        except ToolError as exc:
            error = str(exc) or type(exc).__name__
            await self._push_and_report(
                server, token, call_id, name, arguments, "", error, line
            )
            raise
        text = _first_text(result)
        await self._push_and_report(
            server, token, call_id, name, arguments, text, None, line
        )
        return result

    async def _load(self, server: _TargetMCP, token: str) -> None:
        response = await self._request("GET", ENVIRONMENT_PATH, token)
        state_class = type(server.executor.state)
        server.executor.state = state_class.from_environment(response.json())

    async def _push_and_report(
        self,
        server: _TargetMCP,
        token: str,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: str,
        error: str | None,
        line: dict[str, Any],
    ) -> None:
        failures: list[str] = []
        line["push"] = await self._send(
            "PUT",
            ENVIRONMENT_PATH,
            token,
            server.executor.state.to_environment,
            failures,
        )
        record = {
            "call_id": call_id,
            "function": name,
            "args": arguments,
            "result": result,
            "error": error,
            "blocked": False,
        }
        line["report"] = await self._send(
            "POST", FUNCTION_CALLS_PATH, token, lambda: record, failures
        )
        line["failure"] = "; ".join(failures) or None

    async def _send(
        self,
        method: str,
        path: str,
        token: str,
        body: Callable[[], Any],
        failures: list[str],
    ) -> str:
        try:
            await self._request(method, path, token, body())
        except Exception as exc:
            failures.append(_failure(method, path, exc))
            return "failed"
        return "ok"

    async def _request(
        self, method: str, path: str, token: str, body: Any = None
    ) -> httpx.Response:
        response = await self._http.request(
            method,
            self.control_url + path,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        if not response.is_success:
            excerpt = response.text[:_BODY_EXCERPT]
            raise _Refused(f"HTTP {response.status_code} {excerpt}".rstrip())
        return response

    def _append(self, line: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")


def _session_token(server: _TargetMCP) -> str | None:
    try:
        request = server._mcp_server.request_context.request
    except LookupError:
        request = None
    headers = getattr(request, "headers", None) or {}
    token = (headers.get(SESSION_HEADER) or "").strip()
    return token or os.environ.get(TOKEN_ENV, "").strip() or None


def _first_text(result: Any) -> str:
    """The first text block of what FastMCP's ``call_tool`` returned."""
    if isinstance(result, types.CallToolResult):
        result = result.content
    elif isinstance(result, tuple):  # (unstructured, structured)
        result = result[0]
    for block in result:
        if isinstance(block, types.TextContent):
            return block.text
    return ""


def _failure(method: str, path: str, exc: Exception) -> str:
    if isinstance(exc, _Refused):
        return f"{method} {path}: {exc}"
    return f"{method} {path}: {type(exc).__name__}: {exc}"
