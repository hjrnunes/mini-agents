"""The gateway a target config starts: OGX behind the model shim.

The render and environment tests need only PyYAML. The run tests start real OGX
from the airbnb target's config, with ``tests/model_stub.py`` as the model
provider, so they need the ``gateway`` extra and skip without it. No test calls
a real model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import pytest
from model_stub import ANSWER, StubModel
from target_lifecycle import (
    agent_url,
    listening,
    read_config,
    start_gateway,
    start_target,
)

from mini_agents.kernel import declarations, gateway
from mini_agents.kernel.model_shim import REWRITE_LOG_ENV
from mini_agents.kernel.stack import _ogx_module_available, ogx_config_path
from mini_agents.kernel.target_config import TARGET_FILE

KEY = "sk-gateway-test-key-0123456789abcdef"
CALL = ("get_listing", {"listing_id": "LST-101"})
SOURCE = ogx_config_path().read_text()

needs_ogx = pytest.mark.skipif(
    not _ogx_module_available(),
    reason="OGX is not installed; run `uv sync --frozen --extra gateway`",
)


def _render(**overrides):
    pytest.importorskip("yaml")
    arguments = {
        "connector_id": "airbnb",
        "mcp_url": "http://127.0.0.1:8890/sse",
        "port": 8321,
        "model_id": "gpt-6-sol",
        "shim_url": "http://127.0.0.1:5555/v1",
    }
    return gateway.render_ogx_config(SOURCE, **{**arguments, **overrides})


def test_the_rendered_config_registers_the_model_by_its_exact_id():
    import yaml

    config = yaml.safe_load(_render(model_id="some-model"))

    (model,) = config["registered_resources"]["models"]
    assert model["model_id"] == "some-model"
    assert model["provider_model_id"] == "some-model"
    assert "auto" not in json.dumps(config["registered_resources"])


def test_the_rendered_config_has_one_connector_on_loopback():
    import yaml

    config = yaml.safe_load(_render(connector_id="airbnb-unsafe", port=4321))

    assert config["connectors"] == [
        {"connector_id": "airbnb-unsafe", "url": "http://127.0.0.1:8890/sse"}
    ]
    assert config["server"]["port"] == 4321
    assert config["server"]["host"] == "127.0.0.1"


def test_the_rendered_config_sends_model_requests_to_the_shim_with_a_placeholder():
    import yaml

    rendered = _render(shim_url="http://127.0.0.1:5555/v1")
    config = yaml.safe_load(rendered)

    (provider,) = config["providers"]["inference"]
    assert provider["config"]["base_url"] == "http://127.0.0.1:5555/v1"
    assert provider["config"]["api_token"] == gateway.PLACEHOLDER_KEY
    assert "${env." not in json.dumps(provider)
    assert gateway.API_KEY_ENV not in rendered


def test_the_rendered_config_keeps_everything_else_of_the_source():
    import yaml

    source = yaml.safe_load(SOURCE)
    rendered = yaml.safe_load(_render())

    for section in source:
        if section not in {"connectors", "server", "providers", "registered_resources"}:
            assert rendered[section] == source[section]


@pytest.mark.parametrize("section", ["providers", "registered_resources"])
def test_a_source_with_two_providers_or_models_is_refused(section):
    import yaml

    source = yaml.safe_load(SOURCE)
    key = "inference" if section == "providers" else "models"
    source[section][key].append(dict(source[section][key][0]))

    with pytest.raises(ValueError, match="exactly one"):
        gateway.render_ogx_config(
            yaml.safe_dump(source),
            connector_id="airbnb",
            mcp_url="http://127.0.0.1:1/sse",
            port=1,
            model_id="m",
            shim_url="http://127.0.0.1:2",
        )


def test_the_child_environment_holds_the_shim_and_no_real_key(tmp_path):
    environment = gateway.child_environment(
        {
            gateway.API_KEY_ENV: KEY,
            gateway.BASE_URL_ENV: "https://upstream/v1",
            "X": "y",
        },
        shim_url="http://127.0.0.1:5555/v1",
        state_dir=tmp_path,
    )

    assert environment[gateway.API_KEY_ENV] == gateway.PLACEHOLDER_KEY
    assert environment[gateway.BASE_URL_ENV] == "http://127.0.0.1:5555/v1"
    assert KEY not in json.dumps(environment)
    assert environment["X"] == "y"
    assert environment["SQLITE_STORE_DIR"].startswith(str(tmp_path))
    assert environment["FILES_STORAGE_DIR"].startswith(str(tmp_path))


def test_the_child_environment_keeps_stores_the_caller_names(tmp_path):
    environment = gateway.child_environment(
        {"SQLITE_STORE_DIR": "/elsewhere"},
        shim_url="http://127.0.0.1:5555/v1",
        state_dir=tmp_path,
    )

    assert environment["SQLITE_STORE_DIR"] == "/elsewhere"


@pytest.mark.parametrize("missing", [gateway.BASE_URL_ENV, gateway.MODEL_ID_ENV])
def test_the_gateway_refuses_to_start_without_a_model_endpoint_or_id(
    monkeypatch, missing
):
    for name, value in {
        gateway.BASE_URL_ENV: "http://127.0.0.1:1/v1",
        gateway.MODEL_ID_ENV: "m",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)

    with pytest.raises(SystemExit, match=missing):
        gateway.main(
            ["--connector", "airbnb", "--mcp-url", "http://h:1/sse", "--port", "2"]
        )


def test_the_gateway_refuses_the_tool_servers_own_port(monkeypatch):
    monkeypatch.setenv(gateway.BASE_URL_ENV, "http://127.0.0.1:1/v1")
    monkeypatch.setenv(gateway.MODEL_ID_ENV, "m")

    with pytest.raises(SystemExit, match="different ports"):
        gateway.main(
            [
                "--connector",
                "airbnb",
                "--mcp-url",
                "http://127.0.0.1:8890/sse",
                "--port",
                "8890",
            ]
        )


def _ask(agent_url, mcp_port, model):
    request = {
        "model": model,
        "input": "Show me listing LST-101.",
        "tools": [
            {
                "type": "mcp",
                "server_label": "target",
                "server_url": f"http://127.0.0.1:{mcp_port}/sse",
                "require_approval": "never",
                "allowed_tools": [CALL[0]],
            }
        ],
        "max_output_tokens": 512,
    }
    reply = httpx.post(f"{agent_url}/responses", json=request, timeout=120)
    reply.raise_for_status()
    return reply.json()


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _everything_under(root):
    return b"".join(
        path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    )


@needs_ogx
@pytest.mark.parametrize(
    "model, rewrites",
    [
        (
            "gpt-6-sol",
            [
                {"field": "reasoning_effort", "op": "inject", "value": "none"},
                {"field": "max_tokens", "op": "drop"},
            ],
        ),
        ("another-model", []),
    ],
)
def test_a_gateway_started_from_the_config_runs_a_turn_through_the_shim(
    tmp_path, model, rewrites
):
    config = read_config(declarations.declaration_dir("airbnb", "safe") / TARGET_FILE)
    log = tmp_path / "rewrites.jsonl"
    state = tmp_path / "state"
    # The provider lists another model first, as one with many does.
    with StubModel(model, tool_call=CALL, also_listed=("listed-first",)) as stub:
        target, mcp_port = start_target(config, log=tmp_path / "target.log")
        try:
            process, port = start_gateway(
                config,
                mcp_url=f"http://127.0.0.1:{mcp_port}/sse",
                model=model,
                model_url=stub.url,
                model_key=KEY,
                log=tmp_path / "gateway.log",
                extra_env={REWRITE_LOG_ENV: str(log)},
                extra_args=["--state-dir", str(state)],
            )
            try:
                answer = _ask(agent_url(config, port), mcp_port, model)
            finally:
                code = process.stop()
        finally:
            target.stop()

    assert code == 0
    assert not listening(port)
    assert process.group_is_gone()
    output = answer["output"]
    assert [item["type"] for item in output][:2] == ["mcp_list_tools", "mcp_call"]
    assert json.loads(output[1]["output"])["listing_id"] == "LST-101"
    assert ANSWER in json.dumps(output[-1])

    completions = stub.completions()
    assert len(completions) == 2
    assert [row["rewrites"] for row in _rows(log)] == [rewrites, rewrites]
    for row in _rows(log):
        assert set(row) == {"sequence", "route", "status", "rewrites", "error"}
        assert row["route"] == "/chat/completions"
        assert row["status"] == 200
        assert row["error"] is None
    for received in completions:
        body = received.json()
        assert body["model"] == model
        assert received.headers["authorization"] == f"Bearer {KEY}"
        if rewrites:
            assert body["reasoning_effort"] == "none"
            assert "max_tokens" not in body
            assert body["max_completion_tokens"] > 0
        else:
            assert "reasoning_effort" not in body
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in stub.requests)

    assert KEY.encode() not in _everything_under(tmp_path)
    assert gateway.PLACEHOLDER_KEY in (state / gateway.CONFIG_FILE).read_text()


@needs_ogx
def test_a_gateway_with_nothing_to_stop_exits_on_sigterm_and_cleans_up(tmp_path):
    config = read_config(declarations.declaration_dir("airbnb", "safe") / TARGET_FILE)
    scratch = Path(tempfile.gettempdir())
    before = set(scratch.glob("mini-agents-gateway-*"))
    with StubModel("gpt-6-sol") as stub:
        process, port = start_gateway(
            config,
            mcp_url="http://127.0.0.1:9/sse",
            model="gpt-6-sol",
            model_url=stub.url,
            model_key=KEY,
            log=tmp_path / "gateway.log",
        )
        code = process.stop()

    assert code == 0
    assert not listening(port)
    assert process.group_is_gone()
    after = set(scratch.glob("mini-agents-gateway-*"))
    assert after <= before
    assert KEY.encode() not in _everything_under(tmp_path)
