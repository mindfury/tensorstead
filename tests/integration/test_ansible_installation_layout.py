"""Static integration checks for the Ansible installation surface."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def test_installation_playbook_supports_split_and_colocated_roles() -> None:
    site = (_ROOT / "ansible/playbooks/site.yml").read_text()
    inventory = (_ROOT / "ansible/inventories/example/hosts.yml").read_text()

    assert "tensorstead_coordinators:tensorstead_agents" in site

    # The example must actually *demonstrate* colocation -- one host in both
    # groups -- since that is the arrangement this estate runs and the one the
    # site playbook's group union exists for.
    #
    # Parsed rather than string-matched. This asserted the literal
    # `spark-01: {}`, which broke the moment the example host gained a variable;
    # the empty-mapping spelling was never the property, it was just how the
    # file happened to be written.
    import yaml

    groups = yaml.safe_load(inventory)["all"]["children"]
    coordinators = set(groups["tensorstead_coordinators"]["hosts"])
    agents = set(groups["tensorstead_agents"]["hosts"])

    assert coordinators & agents, (
        "the example inventory shows no colocated host, so the one topology this "
        "estate actually runs is undemonstrated"
    )
    assert agents - coordinators, (
        "the example inventory shows no agent-only host, so the split topology is undemonstrated"
    )


def test_service_units_start_independent_components() -> None:
    coordinator_template = (
        _ROOT / "ansible/roles/coordinator/templates/tensorstead-coordinator.service.j2"
    )
    coordinator = coordinator_template.read_text()
    agent = (_ROOT / "ansible/roles/agent/templates/tensorstead-agent.service.j2").read_text()

    assert "stead coordinator serve" in coordinator
    assert "User={{ tensorstead_service_user }}" in coordinator
    assert "build_agent_app_from_env" in agent
    assert "Requires=docker.service" in agent


def test_wheel_installation_keeps_the_versioned_filename() -> None:
    install_tasks = (_ROOT / "ansible/roles/common/tasks/install.yml").read_text()

    assert "tensorstead_package_artifact | basename" in install_tasks
    assert "tensorstead.whl" not in install_tasks


def test_installing_a_wheel_restarts_every_long_running_service() -> None:
    """A service missing from the restart handler keeps running old code.

    `tensorstead-mcp` was absent from this loop, so from the moment the hosted MCP
    service was introduced, every release left it serving whatever it had
    loaded at its last restart — while `make deploy` and `make readiness` both
    reported success. It was found by the smoke suite's live parity
    check, which noticed the deployed endpoint published five fewer tools than
    the installed wheel contained.

    Derived from the unit templates rather than hard-coded, so a service added
    later cannot be forgotten here too.
    """
    import yaml

    handlers = yaml.safe_load(
        (_ROOT / "ansible/roles/common/handlers/main.yml").read_text(encoding="utf-8")
    )
    # Parsed, not grepped. A substring check against the file text passed even
    # with the service removed, because the explanatory comment named it --
    # the third time today a text-matching guardrail was fooled by prose.
    restarted = {
        item
        for handler in handlers
        if "systemd_service" in str(handler)
        for item in handler.get("loop", [])
    }
    assert restarted, "no restart loop found; this test would pass vacuously"

    units = {
        path.name.removesuffix(".service.j2")
        for path in _ROOT.glob("ansible/roles/*/templates/*.service.j2")
    }
    assert units, "no service templates found; this test would pass vacuously"

    missing = units - restarted
    assert missing == set(), (
        f"these services are installed but never restarted when a new wheel "
        f"lands, so they keep running old code: {sorted(missing)}"
    )


def test_the_restart_handler_does_not_swallow_failures() -> None:
    """A restart that failed must not look identical to one that succeeded."""
    handler = (_ROOT / "ansible/roles/common/handlers/main.yml").read_text(encoding="utf-8")

    # Directives only, never comments: an earlier guardrail matched "raft"
    # inside "draft" and taught the same lesson -- a check that fires on prose
    # is a check people learn to override.
    directives = [
        line.strip()
        for line in handler.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.startswith("failed_when: false") for line in directives), (
        "swallowing the restart result makes a broken deploy report success"
    )
    assert "ansible_facts.services" in handler, (
        "absent units must be skipped by presence, not by ignoring errors"
    )


def test_node_preparation_never_writes_to_netplan() -> None:
    """`/etc/netplan` belongs to NVIDIA Sync, and Sync enforces the claim.

    Cluster Assistant scans that directory, renames competing files to
    `.sync-disabled-<timestamp>`, and journals what it disabled. An earlier
    version of this role wrote an MTU-only netplan file believing a single
    non-overlapping key was separation enough. It was not: the file was
    disabled, the running MTU survived only as residual state from the last
    `netplan apply`, and the role's own assertion reported success for a value
    that would revert at the next reboot.

    The fabric MTU now lives in `/etc/systemd/network/*.link` — a device
    property set through udev, a subsystem Sync does not police.

    Asserted here rather than cleaned up on the node. The transitional tasks
    that removed those files have been deleted, since they could only ever fire
    against a machine an older version of this role had touched. What has to
    survive is the *rule*, and a rule is better kept in a test than in migration
    code nobody can tell is still needed.
    """
    role = _ROOT / "ansible/roles/node_prepare"

    for path in sorted(role.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # Reading Sync's own file is how the role checks whether Cluster
        # Assistant has run; that is a precondition, not a write.
        for line in text.splitlines():
            if "/etc/netplan" not in line or "99-nvidia-sync-cluster.yaml" in line:
                continue
            assert not any(verb in line for verb in ("dest:", "path:", "src:")), (
                f"{path.relative_to(_ROOT)} names a path under /etc/netplan in a "
                f"task that writes or removes files: {line.strip()!r}. That "
                f"directory is NVIDIA Sync's; the fabric MTU belongs in "
                f"/etc/systemd/network/*.link"
            )
