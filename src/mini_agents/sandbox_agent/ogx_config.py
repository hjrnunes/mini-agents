"""OGX's config inside the sandbox: one safe connector, loopback, the relay.

Moved from orch's ``safe_lifecycle.render_safe_gateway_config``: the server
binds ``127.0.0.1`` (OGX binds every interface when ``server.host`` is unset),
the only connector is the domain's safe server on its loopback ``/sse`` URL,
and the API key stays out of the file (``${env.OPENAI_API_KEY}``; OGX gets the
value through its environment). Changed for the sandbox:

- the config is built here rather than read from ``ogx-config.yaml``, because
  PyYAML is installed only with the ``gateway`` extra; the sections follow
  that file;
- ``file_processors`` is left out of ``apis`` and ``providers``: its provider
  loads onnxruntime, whose telemetry the sandbox blocks, and stage 3 processes
  no files, so the expected-blocked egress list stays empty;
- every store sits under the state directory, never in the workdir;
- inference points at the model relay's sandbox URL;
- behind MiDojo's host interceptor there is no server in the sandbox, so no
  connector: each request names the interceptor as its tool's ``server_url``.

The file is written as JSON, which OGX's YAML loader reads unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mini_agents.kernel.stack import MCP_SERVERS

LOOPBACK_HOST = "127.0.0.1"
DOMAINS = tuple(dict.fromkeys(domain for domain, _, _ in MCP_SERVERS))
API_KEY_REFERENCE = "${env.OPENAI_API_KEY}"
APIS = (
    "inference",
    "interactions",
    "messages",
    "responses",
    "tool_groups",
    "tool_runtime",
    "vector_io",
    "files",
)


def connector_id(domain: str) -> str:
    return f"{domain}-safe"


def mcp_url(port: int) -> str:
    """The safe server's SSE URL on one loopback port."""
    return f"http://{LOOPBACK_HOST}:{port}/sse"


def render_ogx_config(
    *,
    domain: str,
    mcp_port: int | None,
    ogx_port: int,
    model: str,
    model_base_url: str,
    state_dir: Path,
) -> dict[str, Any]:
    """The OGX config for one domain's safe server, as a mapping; with
    ``mcp_port`` None, for no server in the sandbox."""

    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain!r}")
    ports = (("OGX port", ogx_port),)
    if mcp_port is not None:
        ports = (("MCP port", mcp_port), *ports)
    for label, port in ports:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{label} must be an integer between 1 and 65535")
    if mcp_port == ogx_port:
        raise ValueError("the MCP server and OGX must use different ports")
    relay = urlparse(model_base_url)
    if relay.scheme not in ("http", "https") or not relay.hostname:
        raise ValueError(f"model base URL is not an http(s) URL: {model_base_url!r}")
    if not model:
        raise ValueError("model is empty")

    store = Path(state_dir) / "ogx"
    sql = "sql_default"
    kv = "kv_default"
    return {
        "version": 2,
        "distro_name": "starter",
        "apis": list(APIS),
        "providers": {
            "inference": [
                {
                    "provider_id": "openai",
                    "provider_type": "remote::vllm",
                    "config": {
                        "api_token": API_KEY_REFERENCE,
                        "base_url": model_base_url,
                    },
                }
            ],
            "interactions": [
                {
                    "provider_id": "builtin",
                    "provider_type": "inline::builtin",
                    "config": {"store": {"table_name": "interactions", "backend": sql}},
                }
            ],
            "messages": [
                {
                    "provider_id": "builtin",
                    "provider_type": "inline::builtin",
                    "config": {
                        "kvstore": {"backend": kv, "namespace": "message_batches"}
                    },
                }
            ],
            "responses": [
                {
                    "provider_id": "builtin",
                    "provider_type": "inline::builtin",
                    "config": {
                        "persistence": {
                            "responses": {
                                "table_name": "responses",
                                "backend": sql,
                                "max_write_queue_size": 10000,
                                "num_writers": 4,
                            }
                        }
                    },
                }
            ],
            "vector_io": [
                {
                    "provider_id": "sqlite-vec",
                    "provider_type": "inline::sqlite-vec",
                    "config": {
                        "db_path": str(store / "sqlite_vec.db"),
                        "persistence": {
                            "namespace": "vector_io::sqlite_vec",
                            "backend": kv,
                        },
                    },
                }
            ],
            "files": [
                {
                    "provider_id": "builtin-files",
                    "provider_type": "inline::localfs",
                    "config": {
                        "storage_dir": str(store / "files"),
                        "metadata_store": {
                            "table_name": "files_metadata",
                            "backend": sql,
                        },
                    },
                }
            ],
            "tool_runtime": [
                {
                    "provider_id": "model-context-protocol",
                    "provider_type": "remote::model-context-protocol",
                }
            ],
            "batches": [
                {
                    "provider_id": "reference",
                    "provider_type": "inline::reference",
                    "config": {"sqlstore": {"table_name": "batches", "backend": sql}},
                }
            ],
        },
        "storage": {
            "backends": {
                kv: {"type": "kv_sqlite", "db_path": str(store / "kvstore.db")},
                sql: {"type": "sql_sqlite", "db_path": str(store / "sql_store.db")},
            },
            "stores": {
                "metadata": {"namespace": "registry", "backend": kv},
                "inference": {
                    "table_name": "inference_store",
                    "backend": sql,
                    "max_write_queue_size": 10000,
                    "num_writers": 4,
                },
                "conversations": {"table_name": "openai_conversations", "backend": sql},
                "prompts": {"table_name": "prompts", "backend": sql},
                "connectors": {"table_name": "connectors", "backend": sql},
            },
        },
        "registered_resources": {
            "models": [
                {
                    "metadata": {},
                    "model_id": model,
                    "provider_id": "all",
                    "provider_model_id": "auto",
                    "model_type": "llm",
                }
            ],
            "vector_dbs": [],
        },
        "server": {"port": ogx_port, "host": LOOPBACK_HOST},
        "connectors": (
            []
            if mcp_port is None
            else [{"connector_id": connector_id(domain), "url": mcp_url(mcp_port)}]
        ),
    }


def write_ogx_config(config: dict[str, Any], destination: Path) -> Path:
    """Write the config atomically and return its path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
