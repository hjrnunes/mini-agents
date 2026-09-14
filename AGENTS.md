# AGENTS.md

MiniBank-style MCP evaluation targets: in-memory worlds whose tools mutate a
ledger an agent can inspect. Chat text is not the score; the ledger is.
Seeded attack scenarios live in `README.md`.

## Architecture

One kernel, three domains, one rule per domain:

- `kernel/` — `mcp_server.py` (FastMCP servers + OGX wiring) and `stack.py`
  (one process tree: six SSE servers, Responses API on `:8321`).
- `domains/<name>/` — `state.py` (in-memory world + seed data), `tools.py`
  (safe executor), `tools_unsafe.py` (red-team counterpart).

## Invariants

**Rules live in the executor, not in the prompt.** Every business rule,
authorization check, and workflow guardrail belongs in `tools.py`. The system
prompt stays policy-free; the tool result carries the outcome.

**Safe and unsafe share one tool schema.** `tools_unsafe.py` is a parallel
executor with the rules stripped, nothing else: same tool names, same
arguments, same state. Any tool added to a safe executor needs its unsafe
counterpart and its `mcp_server.py` wrapper in the same change.

**Dispatch is by name.** The executor resolves `getattr(self, f"_tool_{name}")`
at call time. A tool registered in `mcp_server.py` with no `_tool_` method
returns `{"error": "Unknown tool"}` when called, not at import. Run the server
to catch a wiring gap; imports alone pass.

**Every domain exposes a state summary.** `get_<domain>_state_summary`
serializes the full ledger. Tests and evaluation grade the ledger through this
one tool.

## Adding a domain

Touch, in order: `domains/<name>/` (`state.py`, `tools.py`, `tools_unsafe.py`),
`mcp_server.py` (executor fn, builder, `BUILDERS`), `kernel/stack.py`
(`MCP_SERVERS`), `ogx-config.yaml` and `docker-compose.yml` (the two ports,
safe then unsafe). A domain missing from `stack.py` starts nowhere and fails
silently — check both files before declaring done.

## Seeded state

Each `state.py` seeds its own fixture and its own `authenticated_*` id.
`ORD-104`, `RES-104`, `PAT-104` are the attacks; cross-domain fixture reuse is
a bug. New attacks mean a new fixture in `state.py` and a test pair (safe
rejects, unsafe lands).

## Testing

Tests instantiate the safe and unsafe executors directly and assert on the
state summary — never on chat text, never through a transport. Same attack
args in, two ledger assertions out.

## Runtime

Supports Python 3.11 or newer, the common floor of the Asago pipeline
repositories. Nothing here needs newer syntax or standard library APIs. Do not
raise the floor without a matching change in those repositories.

## Check

`uv sync && uv run pytest` — green is done.
