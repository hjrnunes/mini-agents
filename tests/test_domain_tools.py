"""No domain exposes a tool that executes commands, touches files or fetches URLs.

In the sandbox the MCP server's interpreter is the only process allowed to
reach MiDojo's control plane, which holds the answer key. The model reaches
that process only through the domain tools, so none of them may run a command,
open a shell, read or write a file, or make a request to a URL.

The primary guard is ``REVIEWED_TOOLS``: each domain's registered tool list,
safe and unsafe build, must equal it exactly. A new tool fails the test until
someone reviews what it can reach and adds it here. The word checks on names,
descriptions and argument names are a second guard on top.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from mini_agents.kernel import mcp_server

DOMAINS = mcp_server.DOMAINS

# Every tool each domain registers, reviewed: none runs a command, opens a
# shell, touches a file or fetches a URL. Add a tool here only after checking
# that its executor (tools.py and tools_unsafe.py) works on the ledger alone.
REVIEWED_TOOLS = {
    "klarna": {
        "lookup_order",
        "get_account_details",
        "retrieve_policy",
        "schedule_payment",
        "process_refund",
        "escalate_to_human",
        "get_klarna_state_summary",
    },
    "airbnb": {
        "get_reservation",
        "get_listing",
        "check_availability",
        "lookup_policy",
        "modify_booking",
        "update_listing",
        "moderate_message",
        "escalate_trust_safety",
        "get_airbnb_state_summary",
    },
    "occiai": {
        "get_patient_profile",
        "get_referral",
        "get_education",
        "ask_clinical_question",
        "submit_questionnaire",
        "summarize_for_ehr",
        "commit_to_ehr",
        "generate_education_video",
        "escalate_to_clinician",
        "get_occiai_state_summary",
    },
}

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


def _server(domain, unsafe, monkeypatch):
    monkeypatch.setattr(mcp_server, "UNSAFE_MODE", unsafe)
    return mcp_server.BUILDERS[domain]()


def test_every_domain_has_a_reviewed_tool_list():
    assert set(REVIEWED_TOOLS) == set(DOMAINS)
    assert all(REVIEWED_TOOLS[domain] for domain in DOMAINS)


@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_each_domain_registers_exactly_its_reviewed_tools(domain, unsafe, monkeypatch):
    registered = {tool.name for tool in _tools(_server(domain, unsafe, monkeypatch))}

    assert registered, f"{domain} registers no tools"
    assert registered == REVIEWED_TOOLS[domain], (
        f"{domain} tools changed: added {sorted(registered - REVIEWED_TOOLS[domain])}, "
        f"removed {sorted(REVIEWED_TOOLS[domain] - registered)}. Check that no new "
        "tool runs a command, touches a file or fetches a URL, then update "
        "REVIEWED_TOOLS."
    )


@pytest.mark.parametrize("unsafe", [False, True], ids=["safe", "unsafe"])
@pytest.mark.parametrize("domain", DOMAINS)
def test_no_domain_tool_executes_commands_touches_files_or_fetches_urls(
    domain, unsafe, monkeypatch
):
    server = _server(domain, unsafe, monkeypatch)

    assert _tools(server)
    assert _flagged(server) == {}


def test_a_neutrally_named_new_tool_fails_the_reviewed_list():
    server = mcp_server.BUILDERS["klarna"]()

    @server.tool(name="sync_ledger")
    def sync_ledger(target: str) -> str:
        """Bring the ledger up to date."""
        return ""

    assert _flagged(server) == {}
    assert {tool.name for tool in _tools(server)} - REVIEWED_TOOLS["klarna"] == {
        "sync_ledger"
    }


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
