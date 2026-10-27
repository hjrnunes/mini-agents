# AGENTS.md

MiniBank-style MCP evaluation targets: in-memory worlds whose tools mutate a
ledger an agent can inspect. Chat text is not the score; the ledger is.
Seeded attack scenarios live in `README.md`.

## Architecture

One kernel, three domains, one rule per domain:

- `kernel/` — `mcp_server.py` (FastMCP servers + OGX wiring) and `stack.py`
  (one process tree: six SSE servers, Responses API on `:8321`).
- `domains/<name>/` — `state.py` (in-memory world + seed data), `tools.py`
  (safe executor), `tools_unsafe.py` (red-team counterpart), `seeds.py` (slots
  that `--seed-file` can plant, and each tool's read or write access).

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
one tool. It is an observer, not an assistant tool: it stays registered and any
client can call it by name, but `tools/list` leaves it out
(`kernel/mcp_server.py`, `_TargetMCP`). `--state-summary` or
`MINI_AGENTS_STATE_SUMMARY=1` lists it again, except to a client that sends
`X-Mini-Agents-View: assistant`. No server instruction names it. Pass each new
builder its observer's name; code that needs every registered tool sets
`list_observer = True` on the server first, as `kernel/manifest.py` does.

## Adding a domain

Touch, in order: `domains/<name>/` (`state.py`, `tools.py`, `tools_unsafe.py`,
`seeds.py`), `mcp_server.py` (executor fn, builder, `BUILDERS`),
`kernel/stack.py` (`MCP_SERVERS`), `ogx-config.yaml` and `docker-compose.yml`
(the two ports, safe then unsafe), then regenerate the manifest with
`uv run python scripts/gen_manifest.py` and
`uv run python scripts/gen_declarations.py`. A domain missing from `stack.py`
starts nowhere and fails silently — check both files before declaring done.

**Adding a tool** also needs an entry in the domain's `seeds.py` `TOOL_ACCESS`
(`write` when it creates or changes a ledger record, otherwise `read`), a
regenerated manifest and regenerated declaration files. The manifest generator
stops on a registered tool with no entry, and `tests/test_tool_access.py`
needs the tool's arguments.

## Seeded state

Each `state.py` seeds its own fixture and its own `authenticated_*` id.
`ORD-104`, `RES-104`, `PAT-104` are the attacks; cross-domain fixture reuse is
a bug. New attacks mean a new fixture in `state.py` and a test pair (safe
rejects, unsafe lands).

**Seed slots.** A free-text field that a read tool returns can be planted at
startup (`--seed-file`). Declare it as a `Slot` in `domains/<name>/seeds.py`
with the records each executor returns, add the call to the tables in
`tests/test_seed_slots.py`, and regenerate `tool-manifest.json`. A slot needs
its carrier tool to return the field in both executors, or the record lists must
differ the way `occiai.referral.reason` does. The twin's list holds every safe
record, so a package seeded for the safe target also runs on the twin. Do not edit
`tool-manifest.json` by hand. A slot's `max_chars` (2000 unless set) bounds an
`artifact-seed-v1` item's content and must stay what `seed.schema.json` states
for `mini-agents-seed-v1`.

**Declaration files.** `src/mini_agents/declarations/<domain>/{safe,unsafe}/`
holds `tools-list.json` (the default server's `tools/list` result) and
`runtime-spec.yaml` (`declared-runtime-v1`: seed slots with `max_chars` and
each variant's records, and every listed tool's operation class from
`TOOL_ACCESS`). Regenerate them with `uv run python scripts/gen_declarations.py`
after a tool, its description or arguments, a slot or a `TOOL_ACCESS` entry
changes; `tests/test_declarations.py` checks them against the live servers.

## Testing

Tests instantiate the safe and unsafe executors directly and assert on the
state summary — never on chat text, never through a transport. Same attack
args in, two ledger assertions out.

## Runtime

Supports Python 3.11 or newer, the common floor of the Asago pipeline
repositories. Nothing here needs newer syntax or standard library APIs. Do not
raise the floor without a matching change in those repositories.

## Check

`uv sync --extra gateway && uv run pytest` — green is done. The `gateway` extra
installs OGX from `uv.lock` on Python 3.12 or newer; a bare `uv sync` removes it.

`Dockerfile.sandbox` is the MiDojo sandbox image; after changing it or the lock,
build it and run `scripts/smoke_sandbox_image.sh` (README "Sandbox image").
