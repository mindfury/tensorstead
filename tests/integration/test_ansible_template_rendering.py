"""Render the service templates the way Ansible renders them.

Reading a Jinja template is not the same as rendering one. A conditional
written as a ``{% if %}`` block swallowed the newline that followed it, because
Ansible templates with ``trim_blocks=True``, and welded the next systemd
directive onto the end of a file path:

    ExecStart=... --tls-key /etc/tensorstead/agent.keyRestart=on-failure

The unit installed cleanly, the deploy reported success, and the coordinator
then failed at startup with ``FileNotFoundError`` on a path that had a systemd
directive glued to it. Every check that read the template source passed.

These tests render with Ansible's settings and assert the *output* is a
well-formed unit file, which is the only form in which that class of fault is
visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]

# Ansible's template module defaults.
# autoescape is deliberately off: these render systemd units and env files,
# not HTML, and escaping would corrupt them.
_ENV = Environment(  # noqa: S701
    trim_blocks=True, lstrip_blocks=False, keep_trailing_newline=True
)
_ENV.filters["bool"] = bool

_VARS = {
    "tensorstead_service_user": "tensorstead",
    "tensorstead_config_dir": "/etc/tensorstead",
    "tensorstead_venv": "/opt/tensorstead/venv",
    "tensorstead_coordinator_bind_host": "0.0.0.0",
    "tensorstead_coordinator_port": 8080,
    "tensorstead_coordinator_store": "/var/lib/tensorstead/coordinator/tensorstead.db",
    "tensorstead_agent_tls_cert": "/etc/tensorstead/agent.crt",
    "tensorstead_agent_tls_key": "/etc/tensorstead/agent.key",
    "tensorstead_agent_tls_server_name": "spark-alpha.internal",
    "tensorstead_agent_ca_bundle": "/etc/tensorstead/ca.pem",
    "tensorstead_management_token": "test-token",
    "tensorstead_replication_token": "test-replication-token",
    "tensorstead_agent_port": 8081,
    "tensorstead_model_store": "/var/lib/tensorstead/models",
    "tensorstead_image_store": "/var/lib/tensorstead/images",
    "tensorstead_state_dir": "/var/lib/tensorstead/state",
    "tensorstead_mcp_bind_host": "0.0.0.0",
    "tensorstead_mcp_port": 8090,
    "tensorstead_mcp_path": "/mcp",
    # The agent env file reads a controller-side fact through hostvars.
    "hostvars": {"localhost": {"tensorstead_inference_api_key": "test-inference-key"}},
    "inventory_hostname": "spark-alpha.internal",
}

_UNITS = sorted(
    str(p.relative_to(_ROOT)) for p in _ROOT.glob("ansible/roles/*/templates/*.service.j2")
)
_ENV_FILES = sorted(
    str(p.relative_to(_ROOT)) for p in _ROOT.glob("ansible/roles/*/templates/*.env.j2")
)


def _render(relative: str, **overrides: object) -> str:
    template = (_ROOT / relative).read_text()
    return _ENV.from_string(template).render(**{**_VARS, **overrides})


@pytest.mark.parametrize("unit", _UNITS)
@pytest.mark.parametrize("managed_tls", [True, False])
def test_rendered_unit_files_have_one_directive_per_line(unit: str, managed_tls: bool) -> None:
    """Every non-blank, non-section line must be exactly one ``Key=Value``."""
    rendered = _render(unit, tensorstead_managed_tls=managed_tls)

    for number, line in enumerate(rendered.splitlines(), start=1):
        if not line.strip() or line.startswith("[") or line.lstrip().startswith("#"):
            continue
        key, separator, _ = line.partition("=")
        assert separator == "=", f"{unit}:{number} is not a directive: {line!r}"
        assert key.strip() and " " not in key.strip(), (
            f"{unit}:{number} has a malformed directive key, which is what a "
            f"swallowed newline looks like: {line!r}"
        )


@pytest.mark.parametrize("unit", _UNITS)
def test_rendered_units_keep_their_restart_directive(unit: str) -> None:
    """The swallowed newline consumed `Restart=`; nothing else noticed."""
    rendered = _render(unit, tensorstead_managed_tls=True)
    assert any(line.startswith("Restart=") for line in rendered.splitlines()), (
        f"{unit} lost its Restart= directive when rendered with TLS enabled"
    )


def test_coordinator_tls_paths_are_not_glued_to_the_next_directive() -> None:
    rendered = _render(
        "ansible/roles/coordinator/templates/tensorstead-coordinator.service.j2",
        tensorstead_managed_tls=True,
    )
    exec_line = next(x for x in rendered.splitlines() if x.startswith("ExecStart="))

    assert exec_line.endswith("--tls-key /etc/tensorstead/agent.key")
    assert "Restart" not in exec_line


def test_coordinator_without_managed_tls_passes_no_tls_flags() -> None:
    rendered = _render(
        "ansible/roles/coordinator/templates/tensorstead-coordinator.service.j2",
        tensorstead_managed_tls=False,
    )
    exec_line = next(x for x in rendered.splitlines() if x.startswith("ExecStart="))

    assert "--tls-cert" not in exec_line
    assert "--tls-key" not in exec_line


@pytest.mark.parametrize("env_file", _ENV_FILES)
@pytest.mark.parametrize("managed_tls", [True, False])
def test_rendered_env_files_have_one_assignment_per_line(env_file: str, managed_tls: bool) -> None:
    rendered = _render(env_file, tensorstead_managed_tls=managed_tls)

    for number, line in enumerate(rendered.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, _ = line.partition("=")
        assert separator == "=", f"{env_file}:{number} is not an assignment: {line!r}"
        assert key.strip() and " " not in key.strip(), (
            f"{env_file}:{number} has a malformed variable name: {line!r}"
        )


def test_the_agent_carries_a_node_inference_key_only_when_one_was_generated() -> None:
    """The node-wide key is opt-in; without it the line must be absent, not empty.

    An empty ``TENSORSTEAD_INFERENCE_API_KEY=`` would read as "no key" to the
    agent today, but an absent line says so without depending on that.
    """
    agent_env = "ansible/roles/agent/templates/agent.env.j2"
    with_key = _render(agent_env, tensorstead_managed_tls=False)
    without_key = _render(agent_env, tensorstead_managed_tls=False, hostvars={"localhost": {}})

    assert "TENSORSTEAD_INFERENCE_API_KEY=test-inference-key" in with_key.splitlines()
    assert "TENSORSTEAD_INFERENCE_API_KEY" not in without_key
