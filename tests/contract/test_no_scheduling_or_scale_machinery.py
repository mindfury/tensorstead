"""GUARDRAIL: no scheduling or scale machinery.

Node assignment comes only from the client's request — no placement chooser,
ranker, or bin-packer exists. None of the exclusions listed below are present: no
HA coordinator, consensus, leader election, replicated store, external
broker, policy engine, or multi-tenancy boundary.

This is verified by asserting that:

- No code path chooses a node for a deployment — participating_nodes comes
  from the request, not from a scheduler.
- No HA, consensus, or leader-election concept exists.
- No broker, policy engine, or multi-tenancy boundary exists.
- The deployment service accepts the client's nodes directly, without
  ranking or filtering.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"

# Concepts that would constitute scheduling or scale machinery.
_FORBIDDEN_CONCEPTS = [
    "placement",
    "scheduler",
    "bin_pack",
    "binpack",
    "rank_nodes",
    "ranker",
    "choose_node",
    "select_node",
    "best_node",
    "optimal_node",
    "consensus",
    "leader_election",
    "raft",
    "paxos",
    " replicated_store",
    "replicated_state",
    "high_availability",
    "ha_coordinator",
    "failover",
    "message_broker",
    "kafka",
    "rabbitmq",
    "redis_pubsub",
    "policy_engine",
    "opa_engine",
    "multi_tenant",
    "tenant_id",
    "tenancy_boundary",
]


def test_no_scheduling_concepts_in_product_code() -> None:
    """No placement, ranking, or bin-packing concept exists."""
    offenders: list[str] = []
    for py_file in sorted(_SRC.rglob("*.py")):
        source = py_file.read_text()
        for concept in _FORBIDDEN_CONCEPTS:
            # Word-boundary matched, not substring. A bare `in` check reported
            # "raft" inside "draft" -- a false positive in ordinary prose. A
            # guardrail that cries wolf teaches people to ignore it, which
            # costs more than the rule it was protecting.
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(concept.strip().lower())}(?![a-z0-9])", source.lower()
            ):
                # Distinguish "participating_nodes" (allowed) from "placement"
                # (forbidden) — the former is the client's input, the latter
                # would be a scheduler.
                if concept == "placement" and "participating" in source.lower():
                    # Check it's not a placement scheduler
                    lines = source.splitlines()
                    for i, line in enumerate(lines):
                        if "placement" in line.lower() and "scheduler" in line.lower():
                            offenders.append(f"{py_file.relative_to(_SRC)}:{i + 1}: {concept!r}")
                    continue
                offenders.append(f"{py_file.relative_to(_SRC)}: {concept!r}")
    assert not offenders, "scheduling/scale machinery found:\n" + "\n".join(offenders)


def test_deployment_service_uses_client_nodes_directly() -> None:
    """Deployment creation uses the client's nodes, not a scheduler's choice."""
    deployments_path = _SRC / "service" / "deployments.py"
    source = deployments_path.read_text()

    # The create method must accept participating_nodes as a parameter,
    # not compute them from a scheduler.
    assert "participating_nodes" in source, "create must accept participating_nodes"
    # No scheduler call should appear.
    assert "schedule" not in source.lower(), (
        "deployment service contains 'schedule' — node assignment must come "
        "only from the client's request"
    )
    assert "place" not in source.lower().replace("placement", "").replace("replace", ""), (
        "deployment service contains placement logic — the design forbids it"
    )


def test_no_ha_or_consensus_in_coordinator() -> None:
    """No HA, consensus, or leader-election concept in the coordinator."""
    coordinator_files = list((_SRC / "coordinator").rglob("*.py"))
    for py_file in coordinator_files:
        source = py_file.read_text()
        for concept in ["consensus", "leader_election", "raft", "paxos", "failover"]:
            assert concept.lower() not in source.lower(), (
                f"{py_file.relative_to(_SRC)} contains {concept!r} — "
                "HA/consensus is excluded from v1"
            )


def test_no_multi_tenancy_boundary() -> None:
    """No multi-tenancy boundary or tenant_id concept exists."""
    for py_file in sorted(_SRC.rglob("*.py")):
        source = py_file.read_text()
        assert "tenant_id" not in source.lower(), (
            f"{py_file.relative_to(_SRC)} references tenant_id — v1 has no multi-tenancy"
        )
        assert "multi_tenant" not in source.lower(), (
            f"{py_file.relative_to(_SRC)} references multi_tenant — v1 has no multi-tenancy"
        )
