"""The optimistic revision check.

The per-deployment lock serialises operations, so two concurrent modifications
do not interleave. It does not help a caller that read revision N, lost a race,
and then acted: that caller succeeded against a definition which had changed
underneath it, and the contract requires the loser to receive a *reported
conflict outcome*.

The check was implemented and called from nowhere. It was implemented *twice*,
in fact -- and the copy these tests exercise is the one that eventually got
wired, while the other was deleted.

Note what these tests could not catch: they call the service method directly,
so they passed throughout the period when the PATCH route silently dropped the
field. Route-level coverage is in ``tests/contract/test_api_expected_revision``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.domain.errors import ConcurrentModificationError, NotFoundError
from tensorstead.service.deployments import DeploymentService

pytestmark = pytest.mark.unit


class _Deployment:
    def __init__(self, revision: int) -> None:
        self.id = "d1"
        self.name = "qwen36-27b"
        self.current_revision = revision


class _Repo:
    def __init__(self, revision: int | None = 3) -> None:
        self._deployment = _Deployment(revision) if revision is not None else None

    def get_deployment(self, deployment_id: str) -> Any:
        return self._deployment


def _service(revision: int | None = 3) -> DeploymentService:
    return DeploymentService(_Repo(revision), {})  # type: ignore[arg-type]


def test_a_matching_revision_passes_silently() -> None:
    _service(3)._check_expected_revision("d1", 3)


def test_a_stale_revision_is_refused_with_both_numbers() -> None:
    """The caller must be able to see what it expected and what is true."""
    with pytest.raises(ConcurrentModificationError) as caught:
        _service(4)._check_expected_revision("d1", 3)

    message = str(caught.value)
    assert "qwen36-27b" in message
    assert "3" in message and "4" in message
    assert caught.value.detail == {"expected": 3, "actual": 4}


def test_the_conflict_carries_a_reported_code() -> None:
    """The requirement is a reported outcome, not an opaque failure."""
    with pytest.raises(ConcurrentModificationError) as caught:
        _service(9)._check_expected_revision("d1", 1)

    assert caught.value.code == "concurrent_modification"


def test_an_absent_deployment_is_a_not_found_not_a_conflict() -> None:
    with pytest.raises(NotFoundError):
        _service(None)._check_expected_revision("d1", 1)


def test_the_check_is_opt_in() -> None:
    """A caller with no opinion must be unaffected (backward compatible)."""
    import inspect

    signature = inspect.signature(DeploymentService.modify)
    parameter = signature.parameters["expected_revision"]

    assert parameter.default is None, "absent must mean 'no opinion'"


def test_every_surface_can_supply_it() -> None:
    """The check is useless if only one client can ask for it."""
    from tensorstead.contracts.api import DeploymentModifyRequest

    assert "expected_revision" in DeploymentModifyRequest.model_fields

    cli = __import__("pathlib").Path("src/tensorstead/cli/commands.py").read_text(encoding="utf-8")
    assert "--expect-revision" in cli

    mcp = __import__("pathlib").Path("src/tensorstead/mcp/tools.py").read_text(encoding="utf-8")
    assert '"expected_revision": expected_revision' in mcp


# ----------------------------------------- what a key-level merge quietly drops
#
# A modify naming `host_config` to change `ipc_mode` replaces the whole map. That
# is the documented contract -- the merge is at the key level and a nested map
# is one key -- and it still cost a revision on the appliance by dropping a
# retained diagnostic setting while the operator believed they were changing
# shared memory alone.


def test_a_replaced_nested_map_reports_what_it_dropped() -> None:
    from tensorstead.service.deployments import _nested_keys_dropped

    warnings = _nested_keys_dropped(
        {"host_config": {"shm_size": 4, "environment": {"X": "1"}, "ipc_mode": "host"}},
        {"host_config": {"shm_size": 68719476736}},
    )

    assert len(warnings) == 1
    assert "environment" in warnings[0] and "ipc_mode" in warnings[0]
    assert "host_config" in warnings[0]


def test_changing_a_nested_value_is_not_a_drop() -> None:
    """Changing values is what a modify is for; only removals are surprising."""
    from tensorstead.service.deployments import _nested_keys_dropped

    assert (
        _nested_keys_dropped(
            {"host_config": {"shm_size": 4}}, {"host_config": {"shm_size": 68719476736}}
        )
        == []
    )


def test_an_untouched_nested_map_is_silent() -> None:
    from tensorstead.service.deployments import _nested_keys_dropped

    before = {"host_config": {"shm_size": 4, "ipc_mode": "host"}}
    assert _nested_keys_dropped(before, dict(before)) == []


def test_a_flat_value_replacement_is_not_reported() -> None:
    """Only nested maps. A scalar changing is the ordinary case."""
    from tensorstead.service.deployments import _nested_keys_dropped

    assert _nested_keys_dropped({"tensor_parallel_size": 1}, {"tensor_parallel_size": 2}) == []
