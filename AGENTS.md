# AGENTS.md

MiniBank-style MCP evaluation targets: in-memory worlds whose tools mutate a
ledger an agent can inspect. Chat text is not the score; the ledger is.
Seeded attack scenarios live in `README.md`.

## Key concepts

- **System under test** — one MiniBank world: the kernel, its six tool servers
  and its business ledger.
- **Agent tool** — an operation an assistant can call; the safe and unsafe
  executors expose the same names.
- **Testing tool** — the client that drives the world in a run (Garak, MiDojo,
  orch); this repository never runs one.
- **Ledger** — the in-memory state that tools mutate; the score reads the
  ledger, not the chat text.
- **Slot** (`Slot`) — a free-text field a read tool returns and the seed API
  can plant.
- **Planted item** — the payload content planted in one slot for one attack.
- **Seed document** (`mini-agents-seed-v1`, `artifact-seed-v1`) — the package
  input that names a slot, one of its records, a mode and the content;
  `parse_seed` reads it and the domain `apply_seed` plants it into a state.
- **Declaration file** (`declared-runtime-v1`) — a generated `tools-list.json`,
  `runtime-spec.yaml` or `value-facts.json`, checked against the live servers.
- **Target config** (`target-v1`): a generated `target.yaml` beside them. It
  holds the commands that start a variant and its gateway, and no secret.
- **Gateway**: OGX for one target, behind a loopback model shim that holds
  the model key and rewrites the requests of a model that needs it.

## Architecture

One kernel, three domains, one rule per domain:

- `kernel/` — `mcp_server.py` (FastMCP servers + OGX wiring) and `stack.py`
  (one process tree: six SSE servers, Responses API on `:8321`); `gateway.py`
  and `model_shim.py` (one target's OGX and its model shim); `target_config.py`
  (the `target.yaml` generator).
- `domains/<name>/` — `state.py` (in-memory world + seed data), `tools.py`
  (safe executor), `tools_unsafe.py` (red-team counterpart), `seeds.py` (slots
  the seed API can plant, and each tool's read or write access).

## Invariants

**Rules live in the executor, not in the prompt.** Every business rule,
authorization check, and workflow guardrail belongs in `tools.py`. The system
prompt stays policy-free; the tool result carries the outcome.

**Safe and unsafe share one tool schema.** `tools_unsafe.py` is a parallel
executor with the rules stripped, nothing else: same tool names, same
arguments, same state. Any tool added to a safe executor needs its unsafe
counterpart and its `mcp_server.py` wrapper in the same change.

**Dispatch is by name.** The six executors inherit `ToolDispatch`
(`kernel/executor.py`), which resolves `getattr(self, f"_tool_{name}")` at call
time. A tool registered in `mcp_server.py` with no `_tool_` method
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
`list_observer = True` on the server first, as `tests/test_tool_access.py`
does.

## Adding a domain

Touch, in order: `domains/<name>/` (`state.py`, `tools.py`, `tools_unsafe.py`,
`seeds.py`), `mcp_server.py` (executor fn, builder, `BUILDERS`),
`kernel/stack.py` (`MCP_SERVERS`), `ogx-config.yaml` and `docker-compose.yml`
(the two ports, safe then unsafe), then regenerate the declaration files with
`uv run python scripts/gen_declarations.py`. A domain missing from `stack.py`
starts nowhere and fails silently — check both files before declaring done.

**Adding a tool** also needs an entry in the domain's `seeds.py` `TOOL_ACCESS`
(`write` when it creates or changes a ledger record, otherwise `read`), a
`VALUE_FACTS` entry for each argument that holds one record kind or a date,
and regenerated declaration files. `tests/test_tool_access.py` fails on a
registered tool with no entry, and needs the tool's arguments. Type a date
argument `DateArg` in `mcp_server.py`, so its input schema states the format
its value fact declares (`tests/test_value_facts.py`).

## Seeded state

Each `state.py` seeds its own fixture and its own `authenticated_*` id.
`ORD-104`, `RES-104`, `PAT-104` are the attacks; cross-domain fixture reuse is
a bug. New attacks mean a new fixture in `state.py` and a test pair (safe
rejects, unsafe lands).

**Seed slots.** A free-text field that a read tool returns can carry a planted
item. Declare it as a `Slot` in `domains/<name>/seeds.py`
with the records each executor returns, add the call to the tables in
`tests/test_seed_slots.py`, and regenerate the declaration files. A slot needs
its carrier tool to return the field in both executors, or the record lists must
differ the way `occiai.referral.reason` does. The twin's list holds every safe
record, so a package seeded for the safe target also runs on the twin. A
slot's `max_chars` (2000 unless set) bounds an `artifact-seed-v1` item's
content; `runtime-spec.yaml` states it, and `mini-agents-seed-v1` caps every
item at the same 2000. The testing tool plants the item at run time; the target
starts plain and takes no seed argument.

**A fresh start is the reset.** The ledger lives in memory only and a start
reads nothing back from disk, so a restart with the same arguments gives the
same initial ledger (`tests/test_reset.py`). Keep it that way: no state on
disk that a start reads back, and a seed planted into a running state dies
with the process. README "Reset (a fresh start)".

**Declaration files.** `src/mini_agents/declarations/<domain>/{safe,unsafe}/`
holds `tools-list.json` (the default server's `tools/list` result) and
`runtime-spec.yaml` (`declared-runtime-v1`: seed slots with `max_chars` and
each variant's records, and every listed tool's operation class from
`TOOL_ACCESS`) and `value-facts.json` (the record kind or date format of tool
arguments and ledger values, from the domain's `seeds.py` `VALUE_FACTS`).
Regenerate them with `uv run python scripts/gen_declarations.py` after a tool,
its description or arguments, a slot, a `TOOL_ACCESS` entry or a value fact
changes; `tests/test_declarations.py` checks them against the live servers,
and `tests/test_value_facts.py` that every fact names a listed argument or a
ledger path. Leave a value undeclared when it holds no single record kind.

**Target config.** `target.yaml` (`target_config.py`, written with the
declaration files) is how a testing tool starts, resets and stops a variant:
`endpoint.start`, a `gateway` block, placeholders for the port, the checkout's
interpreter and the run's model profile (`{model}`, `{model_url}`,
`{model_key}`). It never holds a model URL, key or id, and a test fails on one.
The ports come from `stack.MCP_SERVERS`; a new domain gets its config from the
same regeneration. The gateway (`kernel/gateway.py`) registers `MODEL_ID` as an
explicit `provider_model_id`, never `auto`, and keeps the key in its shim
(`kernel/model_shim.py`). A model whose provider needs its requests changed
gets a `MODEL_REWRITES` entry there (today `gpt-6-sol`); the rewrite rules and
their tests live with the shim (`tests/test_model_shim.py`). Never record a
request body, header, key or any value a request chose in a log: the rewrite
log holds the ops only, and the shim looks a request's rules up by its `model`.
`tests/test_target_lifecycle.py` and `tests/test_gateway.py` run the configs
through a stand-in for the testing tool; real OGX there needs the `gateway`
extra.

## Testing

Tests instantiate the safe and unsafe executors directly and assert on the
state summary — never on chat text, never through a transport. Same attack
args in, two ledger assertions out.

**Runtime-contract cases.** `tests/test_declarations.py` validates every
declared seed slot against the artifact generator's runtime contract. Set
`ARTIFACT_GENERATOR_RUNTIME_CONTRACT` to the artifact generator's
`runtime-contract` folder, the one that holds `artifact-runtime-contract-v1/`
(in its checkout: `packages/asago-package/src/asago_package/contracts/runtime-contract`).
The six `test_each_runtime_spec_slot_validates_against_runtime_contract` cases
skip only when the variable is unset or empty. They fail when it names a folder
without `artifact-runtime-contract-v1/schema.json` and
`artifact-runtime-contract-v1/cases/valid/slot-with-record-key.json`, so a
wrong path never passes as a skip. Read the `-rs` summary for `0 skipped`:

```sh
ARTIFACT_GENERATOR_RUNTIME_CONTRACT=<runtime-contract folder> uv run --frozen pytest -q -rs
```

## Runtime

Supports Python 3.11 or newer, the common floor of the Asago pipeline
repositories. Nothing here needs newer syntax or standard library APIs. Do not
raise the floor without a matching change in those repositories.

## Check

`uv sync --frozen --extra gateway`, then
`.venv/bin/python scripts/patch_ogx_streaming.py`, then `uv run --frozen pytest`
— green is done. The `gateway` extra installs OGX from `uv.lock` on Python 3.12
or newer; a plain `uv sync` removes it, and `uv run` without `--frozen` can
re-sync it away. The gateway tests start real OGX and skip without it.
`scripts/patch_ogx_streaming.py --check` exits 1 unless every replacement
landed. Without `ARTIFACT_GENERATOR_RUNTIME_CONTRACT`, a green run
still skips six runtime-contract cases (see Testing).
