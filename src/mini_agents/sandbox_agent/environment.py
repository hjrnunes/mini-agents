"""The servers' environments.

Exec ignores the image's ``ENV``, so the command sets ``PATH`` and
``TIKTOKEN_CACHE_DIR`` for its children itself. The MCP server keeps
``MIDOJO_URL`` and ``MIDOJO_SESSION_TOKEN``: it reads the token from its
environment. OGX gets neither, and gets the model key as ``OPENAI_API_KEY``,
which its config references.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

SYSTEM_PATH = ("/usr/local/bin", "/usr/bin", "/bin")
SCRUBBED_FROM_OGX = ("MIDOJO_SESSION_TOKEN", "MIDOJO_URL")


def child_path(venv_bin: Path) -> str:
    return ":".join((str(venv_bin), *SYSTEM_PATH))


def server_environment(
    environ: Mapping[str, str], *, venv_bin: Path, tiktoken_dir: Path
) -> dict[str, str]:
    """The MCP server's environment: the command's, plus PATH and tiktoken."""

    return {
        **environ,
        "PATH": child_path(venv_bin),
        "TIKTOKEN_CACHE_DIR": str(tiktoken_dir),
    }


def ogx_environment(
    environ: Mapping[str, str],
    *,
    venv_bin: Path,
    tiktoken_dir: Path,
    api_key: str,
) -> dict[str, str]:
    """OGX's environment: the server's, without MiDojo's URL and token."""

    env = server_environment(environ, venv_bin=venv_bin, tiktoken_dir=tiktoken_dir)
    for name in SCRUBBED_FROM_OGX:
        env.pop(name, None)
    env["OPENAI_API_KEY"] = api_key
    return env
