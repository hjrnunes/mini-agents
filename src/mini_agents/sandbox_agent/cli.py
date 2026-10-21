"""``mini-agents-sandbox-agent --domain <domain> <message>``: one run per turn.

The first turn (``MIDOJO_TURN_INDEX`` unset or 0) starts the domain's safe
server and OGX; a later turn checks both still run and fails with a named
reason when one is gone, never restarting it. Every turn sends one Responses
request, records it in the state file and the run summary, copies OGX's log
into the workdir, and prints only the reply on stdout. A turn whose state or
evidence cannot be written fails, and prints no reply. Diagnostics go to
stderr as ``mini-agents-sandbox-agent: <reason>: <detail>``.

The message is always the last argument, taken literally: ``--help``, ``-h``,
``--`` or ``-x`` there is a message, not an option.
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
    Identity,
    Liveness,
    free_ports,
    mcp_command,
    ogx_command,
    pid_alive,
    process_identity,
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
    state_path,
)
from mini_agents.sandbox_agent.summary import (
    MCP_CALLS,
    copy_ogx_log,
    evidence_dir,
    record_rejection,
    summary_path,
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
# Optional: the comma-separated tools the model may call. Unset exposes them all.
ALLOWED_TOOLS = "AGENT_ALLOWED_TOOLS"

MCP_LOG = "mcp-server.log"
OGX_LOG = "ogx.log"
OGX_CONFIG = "ogx-config.yaml"
HELP_FLAGS = ("-h", "--help")


@dataclass(frozen=True)
class Inputs:
    midojo_url: str
    model: str
    model_base_url: str
    model_api_key: str
    instructions: str
    allowed_tools: tuple[str, ...] | None = None


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


def _turn_variable(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        raise AgentFailure(
            failures.INPUT_INVALID, f"{name} is not an integer: {raw!r}"
        ) from None


def _turn_variable_or_none(
    environ: Mapping[str, str], name: str, default: int
) -> int | None:
    try:
        return _turn_variable(environ, name, default)
    except AgentFailure:
        return None


def read_turn(environ: Mapping[str, str]) -> tuple[int, int]:
    """``(index, count)``; an unset index is the first turn, an unset count 1."""

    index = _turn_variable(environ, TURN_INDEX, 0)
    count = _turn_variable(environ, TURN_COUNT, 1)
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
        allowed_tools=_allowed_tools(environ),
    )


def _allowed_tools(environ: Mapping[str, str]) -> tuple[str, ...] | None:
    if ALLOWED_TOOLS not in environ:
        return None
    tools = tuple(name.strip() for name in environ[ALLOWED_TOOLS].split(","))
    if not all(tools):
        raise AgentFailure(
            failures.INPUT_INVALID, f"{ALLOWED_TOOLS} is not a comma-separated list of tool names"
        )
    return tools


def _warn(failure: AgentFailure) -> None:
    print(f"{PROG}: {failure.reason}: {failure.detail}", file=sys.stderr)


class Agent:
    """Runs one turn. ``alive`` and ``identity`` check a PID from an earlier
    turn: it must run and still be the process the first turn started."""

    def __init__(
        self,
        settings: Settings,
        *,
        alive: Liveness = pid_alive,
        identity: Identity = process_identity,
    ):
        self.settings = settings
        self.alive = alive
        self.identity = identity
        self._loaded: AgentState | None = None

    @property
    def evidence(self) -> Path:
        return evidence_dir(self.settings.workdir)

    def run(self, message: str, environ: Mapping[str, str]) -> str:
        try:
            state, index = self._accept(environ)
        except AgentFailure as failure:
            self._reject(failure, environ)
            raise
        try:
            inputs = read_inputs(environ)
            if index == 0:
                self._start(state, inputs, environ)
            else:
                self._check_running(state)
            turn = self._send(state, inputs, message, index)
        except AgentFailure as failure:
            failed = Turn(
                index=index,
                reply=None,
                response_id=None,
                served_model=None,
                gateway_requests=failure.gateway_requests,
                exit_reason=failure.reason,
                failure=failure.detail,
            )
            try:
                self._record(state, failed)
            except AgentFailure as unwritten:
                _warn(unwritten)
            raise
        state.previous_response_id = turn.response_id
        self._record(state, turn)
        return turn.reply or ""

    def _accept(self, environ: Mapping[str, str]) -> tuple[AgentState, int]:
        """This run's turn and its state. A run that is not the next turn
        fails here and is recorded as rejected, not as a turn."""

        path = state_path(self.settings.state_dir)
        try:
            self._loaded = load_state(self.settings.state_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AgentFailure(
                failures.STATE_MISSING, f"unreadable state file {path}: {exc}"
            ) from exc
        index, count = read_turn(environ)
        state = self._loaded
        if state is None:
            if index == 0:
                return AgentState(domain=self.settings.domain, turn_count=count), 0
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
        return state, index

    def _reject(self, failure: AgentFailure, environ: Mapping[str, str]) -> None:
        """Record a rejected run: in the state when it could be read, else in
        the summary alone, leaving an unreadable state file untouched.

        Its index is ``MIDOJO_TURN_INDEX`` when that is an integer, else
        ``None``.
        """

        rejected = Turn(
            index=_turn_variable_or_none(environ, TURN_INDEX, 0),
            reply=None,
            response_id=None,
            served_model=None,
            gateway_requests=0,
            exit_reason=failure.reason,
            failure=failure.detail,
        )
        if self._loaded is not None:
            self._loaded.rejected.append(rejected)
            unwritten = self._persist(self._loaded)
        else:
            try:
                record_rejection(
                    self.settings.workdir,
                    domain=self.settings.domain,
                    turn_count=_turn_variable_or_none(environ, TURN_COUNT, 1),
                    rejected=rejected,
                )
                unwritten = None
            except OSError as exc:
                unwritten = AgentFailure(
                    failures.EVIDENCE_WRITE_FAILED,
                    f"{summary_path(self.settings.workdir)}: {exc}",
                )
        if unwritten is not None:
            _warn(unwritten)

    def _start(self, state: AgentState, inputs: Inputs, environ: Mapping[str, str]) -> None:
        settings = self.settings
        state_dir = settings.state_dir
        for reason, directory in (
            (failures.STATE_WRITE_FAILED, state_dir),
            (failures.EVIDENCE_WRITE_FAILED, self.evidence),
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AgentFailure(reason, f"{directory}: {exc}") from exc
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
        try:
            config_path = write_ogx_config(config, state_dir / OGX_CONFIG)
        except OSError as exc:
            raise AgentFailure(
                failures.STATE_WRITE_FAILED, f"{state_dir / OGX_CONFIG}: {exc}"
            ) from exc

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
                identity = self.identity(process.pid)
                if identity is None and process.poll() is None:
                    raise AgentFailure(
                        failures.START_FAILED,
                        f"{name}: cannot read the identity of pid {process.pid}",
                    )
                record = Process(
                    pid=process.pid,
                    port=port,
                    command=command,
                    log=str(log),
                    identity=identity,
                )
                if name == failures.MCP_SERVER:
                    state.mcp = record
                else:
                    state.ogx = record
                try:
                    save_state(state, state_dir)
                except OSError as exc:
                    raise AgentFailure(
                        failures.STATE_WRITE_FAILED, f"{state_path(state_dir)}: {exc}"
                    ) from exc
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
            current = self.identity(process.pid)
            if process.identity is None or current != process.identity:
                raise AgentFailure(
                    failures.gone(name),
                    f"pid {process.pid} is no longer the process the first turn "
                    f"started (identity {current!r}, recorded {process.identity!r}); "
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
            allowed_tools=inputs.allowed_tools,
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

    def _persist(self, state: AgentState) -> AgentFailure | None:
        """Save the state, rewrite the summary, copy OGX's log. Each is tried
        even when an earlier one fails; the result names the first failure."""

        settings = self.settings
        problems: list[tuple[str, str]] = []
        for reason, path, write in (
            (
                failures.STATE_WRITE_FAILED,
                state_path(settings.state_dir),
                lambda: save_state(state, settings.state_dir),
            ),
            (
                failures.EVIDENCE_WRITE_FAILED,
                summary_path(settings.workdir),
                lambda: write_run_summary(state, settings.workdir),
            ),
            (
                failures.EVIDENCE_WRITE_FAILED,
                self.evidence / OGX_LOG,
                lambda: copy_ogx_log(settings.state_dir / OGX_LOG, settings.workdir),
            ),
        ):
            try:
                write()
            except OSError as exc:
                problems.append((reason, f"{path}: {exc}"))
        if not problems:
            return None
        return AgentFailure(problems[0][0], "; ".join(detail for _, detail in problems))

    def _record(self, state: AgentState, turn: Turn) -> None:
        """Append the turn and persist it. When a write fails, the turn takes
        that failure (an ``ok`` turn becomes the write's reason) and whatever
        can still be written says so; then the failure is raised."""

        state.turns.append(turn)
        unwritten = self._persist(state)
        if unwritten is None:
            return
        if turn.exit_reason == "ok":
            turn.exit_reason = unwritten.reason
            turn.failure = unwritten.detail
        else:
            turn.failure = f"{turn.failure}; {unwritten.reason}: {unwritten.detail}"
        self._persist(state)
        raise unwritten


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        usage="%(prog)s --domain DOMAIN [options] MESSAGE",
        description=(
            "Run one turn of a MiDojo openshell evaluation: start or reuse the "
            "domain's safe MCP server and OGX, send the message, print the reply."
        ),
        epilog=(
            "MESSAGE is the last argument, taken literally even when it looks "
            "like an option ('--help', '-x', '--')."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--python-mcp", type=Path, default=PYTHON_MCP)
    parser.add_argument("--python-ogx", type=Path, default=PYTHON_OGX)
    parser.add_argument("--tiktoken-dir", type=Path, default=TIKTOKEN_DIR)
    parser.add_argument("--listen-timeout", type=float, default=LISTEN_TIMEOUT)
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT)
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Options from every argument but the last; the last is the message.

    Only a lone ``-h`` or ``--help`` asks for help. One ``--`` just before the
    message is accepted and dropped.
    """

    parser = _parser()
    argv = list(argv)
    if len(argv) == 1 and argv[0] in HELP_FLAGS:
        parser.parse_args(argv)  # prints the help and exits 0
    if not argv:
        parser.error("the message (the last argument) is missing")
    options, message = argv[:-1], argv[-1]
    if options and options[-1] == "--":
        options = options[:-1]
    args = parser.parse_args(options)
    args.message = message
    return args


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    alive: Liveness = pid_alive,
    identity: Identity = process_identity,
) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
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
        reply = Agent(settings, alive=alive, identity=identity).run(
            args.message, os.environ if environ is None else environ
        )
    except AgentFailure as failure:
        _warn(failure)
        return 1
    sys.stdout.write(reply + "\n")
    sys.stdout.flush()
    return 0
