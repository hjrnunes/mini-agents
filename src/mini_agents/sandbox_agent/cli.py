"""``mini-agents-sandbox-agent --domain <domain> <message>``: one run per turn.

The first turn (``MIDOJO_TURN_INDEX`` unset or 0) starts the domain's safe
server and OGX; a later turn checks both still run and fails with a named
reason when one is gone, never restarting it. Every turn sends one Responses
request, records it in the state file and the run summary, copies OGX's log
into the workdir, and prints only the reply on stdout. Diagnostics go to
stderr as ``mini-agents-sandbox-agent: <reason>: <detail>``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mini_agents.sandbox_agent import failures
from mini_agents.sandbox_agent.environment import ogx_environment, server_environment
from mini_agents.sandbox_agent.failures import AgentFailure
from mini_agents.sandbox_agent.ogx_config import (
    DOMAINS,
    LOOPBACK_HOST,
    mcp_url,
    render_ogx_config,
    write_ogx_config,
)
from mini_agents.sandbox_agent.processes import (
    Liveness,
    free_ports,
    mcp_command,
    ogx_command,
    pid_alive,
    start_detached,
    stop,
    wait_listening,
)
from mini_agents.sandbox_agent.responses import create_response, reply_text, request_body
from mini_agents.sandbox_agent.state import (
    AgentState,
    Process,
    Turn,
    load_state,
    save_state,
)
from mini_agents.sandbox_agent.summary import (
    MCP_CALLS,
    copy_ogx_log,
    evidence_dir,
    write_run_summary,
)

PROG = "mini-agents-sandbox-agent"
STATE_DIR = Path("/tmp/agent-state")
WORKDIR = Path("/sandbox/workdir")
PYTHON_MCP = Path("/opt/app-root/.venv/bin/python-mcp")
PYTHON_OGX = Path("/opt/app-root/.venv/bin/python-ogx")
TIKTOKEN_DIR = Path("/opt/app-root/tiktoken")
LISTEN_TIMEOUT = 30.0
REQUEST_TIMEOUT = 600.0

TURN_INDEX = "MIDOJO_TURN_INDEX"
TURN_COUNT = "MIDOJO_TURN_COUNT"
REQUIRED = (
    "MIDOJO_URL",
    "MIDOJO_SESSION_TOKEN",
    "AGENT_MODEL",
    "AGENT_MODEL_BASE_URL",
    "AGENT_MODEL_API_KEY",
)
# Set, but may be empty: an empty system message is a choice, a missing one a
# bundle that forgot it.
INSTRUCTIONS = "AGENT_INSTRUCTIONS"

MCP_LOG = "mcp-server.log"
OGX_LOG = "ogx.log"
OGX_CONFIG = "ogx-config.yaml"


@dataclass(frozen=True)
class Inputs:
    midojo_url: str
    model: str
    model_base_url: str
    model_api_key: str
    instructions: str


@dataclass(frozen=True)
class Settings:
    domain: str
    state_dir: Path = STATE_DIR
    workdir: Path = WORKDIR
    python_mcp: Path = PYTHON_MCP
    python_ogx: Path = PYTHON_OGX
    tiktoken_dir: Path = TIKTOKEN_DIR
    listen_timeout: float = LISTEN_TIMEOUT
    request_timeout: float = REQUEST_TIMEOUT


def read_turn(environ: Mapping[str, str]) -> tuple[int, int]:
    """``(index, count)``; an unset index is the first turn, an unset count 1."""

    values = {}
    for name, default in ((TURN_INDEX, 0), (TURN_COUNT, 1)):
        raw = environ.get(name, "").strip()
        try:
            values[name] = int(raw) if raw else default
        except ValueError:
            raise AgentFailure(
                failures.INPUT_INVALID, f"{name} is not an integer: {raw!r}"
            ) from None
    index, count = values[TURN_INDEX], values[TURN_COUNT]
    if count < 1 or not 0 <= index < count:
        raise AgentFailure(
            failures.INPUT_INVALID,
            f"{TURN_INDEX}={index} is not a turn of {TURN_COUNT}={count}",
        )
    return index, count


def read_inputs(environ: Mapping[str, str]) -> Inputs:
    missing = [name for name in REQUIRED if not environ.get(name, "").strip()]
    if INSTRUCTIONS not in environ:
        missing.append(INSTRUCTIONS)
    if missing:
        raise AgentFailure(failures.INPUT_MISSING, "not set: " + ", ".join(missing))
    return Inputs(
        midojo_url=environ["MIDOJO_URL"].strip(),
        model=environ["AGENT_MODEL"].strip(),
        model_base_url=environ["AGENT_MODEL_BASE_URL"].strip(),
        model_api_key=environ["AGENT_MODEL_API_KEY"],
        instructions=environ[INSTRUCTIONS],
    )


class Agent:
    """Runs one turn. ``alive`` checks a PID from an earlier turn."""

    def __init__(self, settings: Settings, *, alive: Liveness = pid_alive):
        self.settings = settings
        self.alive = alive

    @property
    def evidence(self) -> Path:
        return evidence_dir(self.settings.workdir)

    def run(self, message: str, environ: Mapping[str, str]) -> str:
        index, count = read_turn(environ)
        state = self._state_for(index, count)
        try:
            inputs = read_inputs(environ)
            if index == 0:
                self._start(state, inputs, environ)
            else:
                self._check_running(state)
            turn = self._send(state, inputs, message, index)
        except AgentFailure as failure:
            self._record(
                state,
                Turn(
                    index=index,
                    reply=None,
                    response_id=None,
                    served_model=None,
                    gateway_requests=failure.gateway_requests,
                    exit_reason=failure.reason,
                    failure=failure.detail,
                ),
            )
            raise
        state.previous_response_id = turn.response_id
        self._record(state, turn)
        return turn.reply or ""

    def _state_for(self, index: int, count: int) -> AgentState:
        """This turn's state; a turn that does not follow the last fails here
        and records nothing."""

        try:
            state = load_state(self.settings.state_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AgentFailure(
                failures.STATE_MISSING, f"unreadable state file: {exc}"
            ) from exc
        if state is None:
            if index == 0:
                return AgentState(domain=self.settings.domain, turn_count=count)
            raise AgentFailure(
                failures.STATE_MISSING,
                f"turn {index} found no state in {self.settings.state_dir}; "
                "the first turn writes it",
            )
        if len(state.turns) != index:
            raise AgentFailure(
                failures.TURN_OUT_OF_ORDER,
                f"turn {index} follows {len(state.turns)} recorded turn(s)",
            )
        if state.domain != self.settings.domain or state.turn_count != count:
            raise AgentFailure(
                failures.TURN_OUT_OF_ORDER,
                f"turn {index} is for {self.settings.domain} of {count} turns; "
                f"the state is for {state.domain} of {state.turn_count}",
            )
        return state

    def _start(self, state: AgentState, inputs: Inputs, environ: Mapping[str, str]) -> None:
        settings = self.settings
        state_dir = settings.state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        venv_bin = settings.python_mcp.parent
        mcp_port, ogx_port = free_ports(2)
        try:
            config = render_ogx_config(
                domain=settings.domain,
                mcp_port=mcp_port,
                ogx_port=ogx_port,
                model=inputs.model,
                model_base_url=inputs.model_base_url,
                state_dir=state_dir,
            )
        except ValueError as exc:
            raise AgentFailure(failures.INPUT_INVALID, str(exc)) from exc
        config_path = write_ogx_config(config, state_dir / OGX_CONFIG)

        started = []
        try:
            for name, command, port, env, log in (
                (
                    failures.MCP_SERVER,
                    mcp_command(
                        settings.python_mcp,
                        domain=settings.domain,
                        port=mcp_port,
                        midojo_url=inputs.midojo_url,
                        midojo_log=self.evidence / MCP_CALLS,
                    ),
                    mcp_port,
                    server_environment(
                        environ, venv_bin=venv_bin, tiktoken_dir=settings.tiktoken_dir
                    ),
                    state_dir / MCP_LOG,
                ),
                (
                    failures.OGX,
                    ogx_command(settings.python_ogx, config_path),
                    ogx_port,
                    ogx_environment(
                        environ,
                        venv_bin=venv_bin,
                        tiktoken_dir=settings.tiktoken_dir,
                        api_key=inputs.model_api_key,
                    ),
                    state_dir / OGX_LOG,
                ),
            ):
                # cwd is the state directory: nothing a server writes by a
                # relative path may land in the observed workdir.
                process = start_detached(command, env=env, log=log, cwd=state_dir, name=name)
                started.append((name, process, port, log))
                record = Process(pid=process.pid, port=port, command=command, log=str(log))
                if name == failures.MCP_SERVER:
                    state.mcp = record
                else:
                    state.ogx = record
                save_state(state, state_dir)
            deadline = time.monotonic() + settings.listen_timeout
            for name, process, port, log in started:
                wait_listening(process, port=port, name=name, deadline=deadline, log=log)
        except AgentFailure:
            for _, process, _, _ in started:
                stop(process)
            raise

    def _check_running(self, state: AgentState) -> None:
        for name, process in ((failures.MCP_SERVER, state.mcp), (failures.OGX, state.ogx)):
            if process is None:
                raise AgentFailure(
                    failures.gone(name), "the first turn did not start it"
                )
            if not self.alive(process.pid):
                raise AgentFailure(
                    failures.gone(name),
                    f"pid {process.pid} from the first turn is no longer running; "
                    f"see {process.log}",
                )

    def _send(self, state: AgentState, inputs: Inputs, message: str, index: int) -> Turn:
        assert state.mcp is not None and state.ogx is not None
        body = request_body(
            model=inputs.model,
            message=message,
            instructions=inputs.instructions,
            mcp_url=mcp_url(state.mcp.port),
            previous_response_id=state.previous_response_id,
        )
        response = create_response(
            f"http://{LOOPBACK_HOST}:{state.ogx.port}",
            body,
            timeout=self.settings.request_timeout,
        )
        model = response.get("model")
        return Turn(
            index=index,
            reply=reply_text(response),
            response_id=response["id"],
            served_model=model if isinstance(model, str) else None,
            gateway_requests=1,
            exit_reason="ok",
        )

    def _record(self, state: AgentState, turn: Turn) -> None:
        """Save the turn, rewrite the summary, copy OGX's log. A failure to
        write evidence is reported but never replaces the turn's outcome."""

        state.turns.append(turn)
        try:
            save_state(state, self.settings.state_dir)
            write_run_summary(state, self.settings.workdir)
            copy_ogx_log(self.settings.state_dir / OGX_LOG, self.settings.workdir)
        except OSError as exc:
            print(f"{PROG}: evidence not written: {exc}", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Run one turn of a MiDojo openshell evaluation: start or reuse the "
            "domain's safe MCP server and OGX, send the message, print the reply."
        ),
    )
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--python-mcp", type=Path, default=PYTHON_MCP)
    parser.add_argument("--python-ogx", type=Path, default=PYTHON_OGX)
    parser.add_argument("--tiktoken-dir", type=Path, default=TIKTOKEN_DIR)
    parser.add_argument("--listen-timeout", type=float, default=LISTEN_TIMEOUT)
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("message", help="this turn's message (the last argument)")
    return parser


def _message_last(argv: Sequence[str]) -> list[str]:
    """Take the last argument as the message even when it starts with '-'."""

    argv = list(argv)
    if argv and "--" not in argv and argv[-1] not in ("-h", "--help"):
        argv.insert(len(argv) - 1, "--")
    return argv


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    alive: Liveness = pid_alive,
) -> int:
    args = _parser().parse_args(_message_last(sys.argv[1:] if argv is None else argv))
    settings = Settings(
        domain=args.domain,
        state_dir=args.state_dir,
        workdir=args.workdir,
        python_mcp=args.python_mcp,
        python_ogx=args.python_ogx,
        tiktoken_dir=args.tiktoken_dir,
        listen_timeout=args.listen_timeout,
        request_timeout=args.request_timeout,
    )
    try:
        reply = Agent(settings, alive=alive).run(
            args.message, os.environ if environ is None else environ
        )
    except AgentFailure as failure:
        print(f"{PROG}: {failure.reason}: {failure.detail}", file=sys.stderr)
        return 1
    sys.stdout.write(reply + "\n")
    sys.stdout.flush()
    return 0
