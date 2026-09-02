"""OGX connectors, MCP ports, and compose publish list must stay aligned."""

from pathlib import Path

from mini_agents.kernel.stack import MCP_SERVERS, ogx_config_path


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
