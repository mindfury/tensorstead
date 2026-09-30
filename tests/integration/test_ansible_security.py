"""Static safeguards against secret exposure and non-idempotent service setup."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def test_secret_templates_are_not_examples_and_are_owner_only() -> None:
    profile = (_ROOT / "ansible/group_vars/all.yml.example").read_text()
    coordinator_tasks = (_ROOT / "ansible/roles/coordinator/tasks/main.yml").read_text()
    agent_tasks = (_ROOT / "ansible/roles/agent/tasks/main.yml").read_text()

    assert not any(
        line.startswith("tensorstead_management_token:") for line in profile.splitlines()
    )
    assert not any(
        line.startswith("tensorstead_replication_token:") for line in profile.splitlines()
    )
    assert "mode: '0600'" in coordinator_tasks
    assert "mode: '0600'" in agent_tasks
    assert "no_log: true" in coordinator_tasks
    assert "no_log: true" in agent_tasks


def test_roles_use_named_idempotent_systemd_units() -> None:
    """Units are named, and their boot arrangement is declared rather than left.

    This asserted ``enabled: true`` literally, which conflated two things: that
    the roles *declare* a boot arrangement for a *named* unit -- the idempotence
    property worth pinning -- and that the arrangement is always "on", which was
    only ever the default. The estate's is now off by operator decision, so the
    literal check would have forced the roles back to a
    behaviour nobody wants in order to keep a test green.

    What matters is that enablement is declared and governed by the documented
    setting, so a deploy converges in both directions instead of only ever
    enabling -- which is how a disabled unit quietly comes back.
    """
    coordinator_tasks = (_ROOT / "ansible/roles/coordinator/tasks/main.yml").read_text()
    agent_tasks = (_ROOT / "ansible/roles/agent/tasks/main.yml").read_text()

    assert "tensorstead-coordinator" in coordinator_tasks
    assert "tensorstead-agent" in agent_tasks
    for tasks in (coordinator_tasks, agent_tasks):
        assert "enabled:" in tasks, "the role no longer declares a boot arrangement at all"
        assert "tensorstead_services_enabled_at_boot" in tasks, (
            "boot enablement is hardcoded again rather than governed by the setting"
        )


def test_operator_tls_and_ssh_prerequisites_are_documented() -> None:
    guide = (_ROOT / "ansible/README.md").read_text()
    private_ca_guide = (_ROOT / "ansible/tls-private-ca.md").read_text()

    for expected in (
        "TLS is automatic in the normal setup",
        "network ID card",
        "SSH prerequisite",
        "without an interactive password",
        "Safe local validation versus deployment",
    ):
        assert expected in guide
    assert "spark-alpha.internal" in private_ca_guide
    assert "subjectAltName" in private_ca_guide


def test_normal_installation_manages_tls_without_a_user_supplied_key() -> None:
    site = (_ROOT / "ansible/playbooks/site.yml").read_text()
    profile = (_ROOT / "ansible/group_vars/all.yml.example").read_text()

    assert "Create the managed private CA once" in site
    assert "Generate agent private key and certificate request" in site
    assert "Install the managed public CA bundle" in site
    assert "tensorstead_managed_tls: true" in profile


def test_new_agent_requests_are_resigned_after_a_host_reimage() -> None:
    site = (_ROOT / "ansible/playbooks/site.yml").read_text()
    signing_section = site.split("- name: Sign each new agent certificate", maxsplit=1)[1]
    signing_section = signing_section.split("- name: Install managed certificates", maxsplit=1)[0]

    assert "hostvars[item].agent_tls_csr is defined" in signing_section
    assert "creates:" not in signing_section
