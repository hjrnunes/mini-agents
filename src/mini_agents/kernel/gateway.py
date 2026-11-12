"""Start the gateway for one target: the model shim, then OGX.

    python -m mini_agents.kernel.gateway --connector ID --mcp-url URL --port PORT

The gateway answers the Responses API on loopback ``PORT`` and knows one tool
server, ``URL``, under ``ID``. A target config's ``gateway.start`` runs this
command (``kernel/target_config.py``); ``kernel/stack.py`` stays the
one-command stack for Docker.

The model endpoint, key and model id reach the process through the variables
the target config's ``gateway.env`` names:

- ``OPENAI_BASE_URL`` is the provider's base URL.
- ``OPENAI_API_KEY`` is its key.
- ``MODEL_ID`` is the model, which OGX registers with that exact id as its
  ``provider_model_id``. Never ``auto``: OGX resolves ``auto`` to the first
  model the provider lists, so the request would name whatever the provider
  lists first.

OGX talks to the model shim (``kernel/model_shim.py``) on a free loopback port,
with a placeholder key. The shim holds the real key and rewrites the requests
of a model that needs it (``MODEL_REWRITES``). ``MINI_AGENTS_REWRITE_LOG`` names
the file the shim appends its rewrite rows to.

The process runs until OGX exits or it gets SIGTERM or SIGINT, then stops OGX
and the shim and removes the scratch directory. OGX's stores live in that
directory, so each start is a fresh gateway. ``--state-dir`` keeps the
directory: it holds the rendered OGX config (a placeholder key, never the real
one) and OGX's stores.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mini_agents.kernel.model_shim import (
    LOOPBACK_HOST,
    REWRITE_LOG_ENV,
    ModelShim,
    rewrites_for,
)
from mini_agents.kernel.stack import _ogx_module_available, ogx_config_path

BASE_URL_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "OPENAI_API_KEY"
MODEL_ID_ENV = "MODEL_ID"
# What OGX sends upstream in place of the key; the shim replaces it.
PLACEHOLDER_KEY = "mini-agents-shim-placeholder"
CONFIG_FILE = "ogx-config.yaml"


def render_ogx_config(
    source: str,
    *,
    connector_id: str,
    mcp_url: str,
    port: int,
    model_id: str,
    shim_url: str,
) -> str:
    """Return the OGX config of one target's gateway, rendered from ``source``.

    The rendered config listens on loopback ``port``, names ``connector_id``
    as its only connector so the gateway knows no other tool server, sends its
    model requests to ``shim_url`` with a placeholder key, and registers
    ``model_id`` as both its id and its ``provider_model_id``.
    """

    import yaml

    loaded = yaml.safe_load(source)
    if not isinstance(loaded, dict):
        raise ValueError("the OGX config is not a mapping")
    config = copy.deepcopy(loaded)
    provider = _single(config, "providers", "inference", "inference provider")
    settings = provider.setdefault("config", {})
    settings["base_url"] = shim_url
    settings["api_token"] = PLACEHOLDER_KEY
    model = _single(config, "registered_resources", "models", "registered model")
    model["model_id"] = model_id
    model["provider_model_id"] = model_id
    config["server"] = {
        **(config.get("server") or {}),
        "port": port,
        "host": LOOPBACK_HOST,
    }
    config["connectors"] = [{"connector_id": connector_id, "url": mcp_url}]
    return yaml.safe_dump(config, sort_keys=False)


def _single(
    config: dict[str, Any], section: str, key: str, what: str
) -> dict[str, Any]:
    parent = config.get(section)
    entries = parent.get(key) if isinstance(parent, dict) else None
    if (
        not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(entries[0], dict)
    ):
        raise ValueError(f"the OGX config {section}.{key} must name exactly one {what}")
    return entries[0]


def ogx_command(config: Path) -> list[str]:
    """Return the command that runs OGX on ``config`` under this interpreter.

    The interpreter that runs the gateway runs OGX, so a venv's own OGX is the
    one that starts whatever ``PATH`` holds.
    """

    if not _ogx_module_available():
        raise SystemExit(
            "ogx is not installed, so the gateway cannot start. Run "
            "`uv sync --frozen --extra gateway`, then "
            "`.venv/bin/python scripts/patch_ogx_streaming.py`."
        )
    return [sys.executable, "-m", "ogx.cli.ogx", "run", str(config), "--insecure"]


def child_environment(
    environ: Mapping[str, str], *, shim_url: str, state_dir: Path
) -> dict[str, str]:
    """Return the environment OGX runs with: the shim and no real key.

    OGX's stores go under ``state_dir`` unless the caller names them.
    """

    environment = dict(environ)
    environment[BASE_URL_ENV] = shim_url
    environment[API_KEY_ENV] = PLACEHOLDER_KEY
    environment.setdefault("SQLITE_STORE_DIR", str(state_dir / "store"))
    environment.setdefault("FILES_STORAGE_DIR", str(state_dir / "files"))
    return environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start one target's gateway: the model shim, then OGX."
    )
    parser.add_argument("--connector", required=True, help="the tool server's id")
    parser.add_argument("--mcp-url", required=True, help="the tool server's URL")
    parser.add_argument(
        "--port", required=True, type=int, help="the Responses API port"
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="keep the rendered config and OGX's stores here, not in a "
        "temporary directory removed on exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    upstream = os.environ.get(BASE_URL_ENV)
    model_id = os.environ.get(MODEL_ID_ENV)
    missing = [
        name
        for name, value in ((BASE_URL_ENV, upstream), (MODEL_ID_ENV, model_id))
        if not value
    ]
    if missing:
        raise SystemExit(
            f"the gateway needs {' and '.join(missing)} in its environment"
        )
    assert upstream and model_id
    if urlsplit(args.mcp_url).port == args.port:
        raise SystemExit("the gateway and the tool server must use different ports")
    log = os.environ.get(REWRITE_LOG_ENV)

    kept = args.state_dir is not None
    state_dir = (
        args.state_dir
        if args.state_dir is not None
        else Path(tempfile.mkdtemp(prefix="mini-agents-gateway-"))
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    shim = ModelShim(
        upstream,
        os.environ.get(API_KEY_ENV) or None,
        rewrites_for(model_id),
        log_path=Path(log) if log else None,
    )
    ogx: subprocess.Popen[bytes] | None = None
    stopping = False

    def stop(_signum: int | None = None, _frame: object = None) -> None:
        nonlocal stopping
        stopping = True
        if ogx is not None and ogx.poll() is None:
            ogx.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        shim.start()
        config = state_dir / CONFIG_FILE
        config.write_text(
            render_ogx_config(
                ogx_config_path().read_text(),
                connector_id=args.connector,
                mcp_url=args.mcp_url,
                port=args.port,
                model_id=model_id,
                shim_url=shim.url,
            )
        )
        if not stopping:
            ogx = subprocess.Popen(
                ogx_command(config),
                env=child_environment(
                    os.environ, shim_url=shim.url, state_dir=state_dir
                ),
            )
            # A signal between the spawn and this line saw no process to stop.
            if stopping:
                ogx.terminate()
        code = ogx.wait() if ogx is not None else 0
        if stopping:
            return 0
        return code if code >= 0 else 128 - code
    finally:
        if ogx is not None and ogx.poll() is None:
            ogx.terminate()
            try:
                ogx.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ogx.kill()
                ogx.wait()
        shim.stop()
        if not kept:
            shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
