"""No domain exposes a tool that executes commands, touches files or fetches URLs.

In the sandbox the MCP server's interpreter is the only process allowed to
reach MiDojo's control plane, which holds the answer key. The model reaches
that process only through the domain tools, so none of them may run a command,
open a shell, read or write a file, or make a request to a URL. The tests read
each server's registered tool list (names, descriptions, argument names), in
both the safe and the unsafe build.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from mini_agents.kernel import mcp_server

DOMAINS = mcp_server.DOMAINS

# Whole words of a tool name (split on "_") or an argument name.
NAME_WORDS = {
    "exec", "execute", "run", "shell", "bash", "sh", "zsh", "cmd", "command",
    "subprocess", "spawn", "eval", "script", "terminal",
    "file", "files", "filename", "filepath", "fs", "filesystem", "path",
    "dir", "directory", "folder", "read", "write", "open",
    "fetch", "url", "uri", "href", "http", "https", "request", "download",
    "upload", "curl", "wget", "browse", "browser", "web", "website", "host",
}
ARGUMENT_NAMES = NAME_WORDS | {"code", "argv", "endpoint"}
DESCRIPTION = re.compile(
    r"\b("
    r"shell|bash|terminal|subprocess|command[- ]line"
    r"|(execute|run)s?\s+(an?\s+|the\s+|any\s+|arbitrary\s+)?"
    r"(shell\s+|system\s+)?(commands?|scripts?|code|programs?|binar(y|ies))"
    r"|files?|file\s?system|director(y|ies)|paths?"
    r"|urls?|https?|fetch\w*|web\s?pages?|websites?|internet|curl|wget"
    r"|(http|network|web)\s+requests?|downloads?|uploads?"
    r")\b",
    re.IGNORECASE,
)


def _reasons(tool) -> list[str]:
    reasons = [
        f"name word {word!r}"
        for word in tool.name.lower().split("_")
        if word in NAME_WORDS
    ]
    for argument in (tool.inputSchema or {}).get("properties") or {}:
        if argument.lower() in ARGUMENT_NAMES or set(
            argument.lower().split("_")
        ) & NAME_WORDS:
            reasons.append(f"argument {argument!r}")
    reasons += [
        f"description says {match.group(0)!r}"
        for match in DESCRIPTION.finditer(tool.description or "")
    ]
    return reasons


def _tools(server):
    return asyncio.run(server.list_tools())


def _flagged(server) -> dict[str, list[str]]:
    return {tool.name: r for tool in _tools(server) if (r := _reasons(tool))}


@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_no_domain_tool_executes_commands_touches_files_or_fetches_urls(
    domain, unsafe, monkeypatch
):
    monkeypatch.setattr(mcp_server, "UNSAFE_MODE", unsafe)
    server = mcp_server.BUILDERS[domain]()

    assert len(_tools(server)) > 1
    assert _flagged(server) == {}


def _run_shell_command(command: str) -> str:
    """Run a shell command and return its output."""
    return ""


def _read_file(path: str) -> str:
    """Return the contents of a file."""
    return ""


def _save_note(name: str, body: str) -> str:
    """Write the note to a local file in the working directory."""
    return ""


def _lookup_page(page: str) -> str:
    """Fetch a page from the internet and return its text."""
    return ""


def _check_link(url: str) -> str:
    """Return whether the link resolves."""
    return ""


def _execute(code: str) -> str:
    """Evaluate the expression."""
    return ""


@pytest.mark.parametrize(
    "fn, expected",
    [
        (_run_shell_command, "name word 'run'"),
        (_read_file, "name word 'file'"),
        (_save_note, "description says 'file'"),
        (_lookup_page, "description says 'Fetch'"),
        (_check_link, "argument 'url'"),
        (_execute, "argument 'code'"),
    ],
)
def test_a_tool_that_could_reach_the_host_is_flagged(fn, expected):
    server = mcp_server.BUILDERS["klarna"]()
    name = fn.__name__.lstrip("_")
    server.tool(name=name)(fn)

    assert expected in _flagged(server)[name]
