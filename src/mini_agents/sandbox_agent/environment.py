"""The servers' environments.

Exec ignores the image's ``ENV``, so the command sets ``PATH`` and
``TIKTOKEN_CACHE_DIR`` for its children itself. A runtime may still hand the
command MiDojo's ``MIDOJO_URL`` and ``MIDOJO_SESSION_TOKEN``; neither server
needs them, so neither gets them. OGX gets the model key as
``OPENAI_API_KEY``, which its config references.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")
SCRUBBED = ("MIDOJO_SESSION_TOKEN", "MIDOJO_URL")


def child_path(venv_bin: Path) -> str:
    return ":".join((str(venv_bin), *SYSTEM_PATH))


def server_environment(
    environ: Mapping[str, str], *, venv_bin: Path, tiktoken_dir: Path
) -> dict[str, str]:
    """The MCP server's environment: the command's, without ``SCRUBBED``,
    plus PATH and tiktoken."""

    env = {name: value for name, value in environ.items() if name not in SCRUBBED}
    env["PATH"] = child_path(venv_bin)
    env["TIKTOKEN_CACHE_DIR"] = str(tiktoken_dir)
    return env


def ogx_environment(
    environ: Mapping[str, str],
    *,
    venv_bin: Path,
    tiktoken_dir: Path,
    api_key: str,
) -> dict[str, str]:
    """OGX's environment: the server's, plus the model key."""

    env = server_environment(environ, venv_bin=venv_bin, tiktoken_dir=tiktoken_dir)
    env["OPENAI_API_KEY"] = api_key
    return env
