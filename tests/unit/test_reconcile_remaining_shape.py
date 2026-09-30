"""``reconcile`` raised 500 exactly when it had something to report.

The summary it returns flattens each node's leftover divergences into one list,
tagged by node. It was written as::

    {"node_id": nid, **info.get("remaining", [])}

and ``info["remaining"]`` is a *list*. ``**`` over a list raises
``TypeError: 'list' object is not a mapping``.

The comprehension was guarded by ``if info.get("remaining")``, so it evaluated
only for nodes that still had a divergence: reconcile returned cleanly whenever
there was nothing to say and raised whenever there was. It is the command an
operator reaches for *because* something has diverged, so the only path that
mattered was the broken one.

Found by reconciling a real deployment whose container had gone missing.
These drive the real ``ObservationService.reconcile`` rather than mirroring its
comprehension, so a reintroduction fails here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tensorstead.domain.identity import new_ulid
from tensorstead.service.observation import ObservationService

pytestmark = pytest.mark.unit

_DEPLOYMENT = new_ulid()
_PRESENT = new_ulid()
_ABSENT = new_ulid()


class _Repo:
    """Enough repository to reach the summary, and no more."""

    def __init__(self, nodes: dict[str, Any]) -> None:
        self._nodes = nodes

    def get_deployment(self, deployment_id: str) -> Any:
        return type("D", (), {"id": _DEPLOYMENT, "name": "d", "current_revision": 1})()

    def get_revision(self, deployment_id: str, revision: int) -> Any:
        return type("R", (), {"participating_nodes": tuple(self._nodes)})()

    def get_node(self, node_id: str) -> Any:
        return self._nodes.get(node_id)


class _Client:
    """A node client whose reconcile leaves divergences behind."""

    def __init__(self, remaining: list[dict[str, Any]]) -> None:
        self._remaining = remaining

    def reconcile_deployment(self, node: Any, deployment_id: str) -> dict[str, Any]:
        return {"changed": [], "remaining": self._remaining}


def _service(nodes: dict[str, Any], remaining: list[dict[str, Any]]) -> ObservationService:
    """One place to cross the port boundary these stubs stand in for."""
    return ObservationService(_Repo(nodes), _Client(remaining))  # type: ignore[arg-type]


def _node() -> Any:
    return type("N", (), {"id": _PRESENT, "name": "n", "registered_at": datetime.now()})()


def test_a_node_the_inventory_lost_is_reported_not_raised() -> None:
    """The simplest path to a non-empty ``remaining``: no node client involved."""
    service = _service({_ABSENT: None}, [])
    out = service.reconcile(_DEPLOYMENT)
    assert out["status"] == "partial"
    assert out["remaining"] == [{"node_id": _ABSENT, "kind": "node_not_found"}]


def test_divergences_the_agent_could_not_clear_are_reported() -> None:
    service = _service(
        {_PRESENT: _node()}, [{"kind": "declared_running_but_absent", "detail": None}]
    )
    out = service.reconcile(_DEPLOYMENT)
    assert out["status"] == "partial"
    assert out["remaining"] == [
        {"node_id": _PRESENT, "kind": "declared_running_but_absent", "detail": None}
    ]


def test_several_on_one_node_are_reported_separately() -> None:
    """Flattening, not merging: two problems must not collapse into one entry."""
    service = _service(
        {_PRESENT: _node()}, [{"kind": "declared_running_but_absent"}, {"kind": "image_mismatch"}]
    )
    out = service.reconcile(_DEPLOYMENT)
    assert len(out["remaining"]) == 2
    assert {e["kind"] for e in out["remaining"]} == {
        "declared_running_but_absent",
        "image_mismatch",
    }


def test_full_convergence_still_reports_succeeded() -> None:
    """The path that always worked must keep working."""
    service = _service({_PRESENT: _node()}, [])
    out = service.reconcile(_DEPLOYMENT)
    assert out["status"] == "succeeded"
    assert out["remaining"] == []
