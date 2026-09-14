"""OGX connectors, MCP ports, and compose publish list must stay aligned."""

from pathlib import Path

import pytest

from mini_agents.kernel.stack import (
    MCP_SERVERS,
    _ogx_command,
    _ogx_module_available,
    ogx_config_path,
)


def test_ogx_connectors_match_mcp_listen_ports():
    text = ogx_config_path().read_text()

    for domain, port, unsafe in MCP_SERVERS:
        suffix = "unsafe" if unsafe else "safe"
        assert f"connector_id: {domain}-{suffix}" in text
        assert f"url: http://localhost:{port}/sse" in text


def test_each_domain_has_safe_and_unsafe_ports():
    by_domain: dict[str, dict[str, int]] = {}
    for domain, port, unsafe in MCP_SERVERS:
        by_domain.setdefault(domain, {})["unsafe" if unsafe else "safe"] = port

    assert set(by_domain) == {"klarna", "airbnb", "occiai"}
    for ports in by_domain.values():
        assert ports["safe"] != ports["unsafe"]


def test_compose_publishes_ogx_and_every_mcp_port():
    compose = Path(ogx_config_path()).with_name("docker-compose.yml").read_text()
    assert '"8321:8321"' in compose
    for _, port, _ in MCP_SERVERS:
        assert f'"{port}:{port}"' in compose


def test_ogx_launcher_prefers_the_console_script(monkeypatch, tmp_path):
    """With ogx on PATH, launch the console script."""

    monkeypatch.setattr(
        "mini_agents.kernel.stack.shutil.which", lambda _name: "/fake/bin/ogx"
    )
    command = _ogx_command(tmp_path / "ogx-config.yaml")
    assert command[:3] == ["/fake/bin/ogx", "run", str(tmp_path / "ogx-config.yaml")]
    assert command[-1] == "--insecure"


def test_ogx_launcher_falls_back_to_the_submodule(monkeypatch, tmp_path):
    """Without a console script, name the submodule, never bare `ogx`."""

    monkeypatch.setattr("mini_agents.kernel.stack.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "mini_agents.kernel.stack._ogx_module_available", lambda: True
    )
    command = _ogx_command(tmp_path / "ogx-config.yaml")
    assert command[1:4] == ["-m", "ogx.cli.ogx", "run"]
    # `python -m ogx` can never work; ogx ships no __main__.py.
    assert command[1:3] != ["-m", "ogx"]


def test_ogx_launcher_reports_a_missing_install(monkeypatch, tmp_path):
    """A missing ogx must explain itself instead of raising ModuleNotFoundError."""

    monkeypatch.setattr("mini_agents.kernel.stack.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "mini_agents.kernel.stack._ogx_module_available", lambda: False
    )
    with pytest.raises(SystemExit, match="ogx is not installed"):
        _ogx_command(tmp_path / "ogx-config.yaml")


def test_missing_ogx_module_is_detected_not_raised():
    """find_spec raises when a parent package is absent; that is 'unavailable'."""

    assert _ogx_module_available() in {True, False}
