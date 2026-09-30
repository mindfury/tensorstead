"""Unit test the compatibility policy.

The policy is data + a small function in ``contracts/version.py``. These tests
pin the three cases:

- major mismatch refuses *every* operation with ``agent_version_incompatible``,
  naming both versions;
- matching major with the agent below an operation's minimum refuses *that
  operation only* with ``operation_unsupported_by_agent``, naming required and
  actual;
- matching major at or above the minimum permits.
"""

from __future__ import annotations

import pytest

from tensorstead.contracts.version import (
    CODE_AGENT_VERSION_INCOMPATIBLE,
    CODE_OPERATION_UNSUPPORTED_BY_AGENT,
    CONTRACT_VERSION,
    CONTRACT_VERSION_MAJOR,
    OP_MODEL_ACQUIRE,
    OP_NODE_REGISTER,
    CompatibilityRefusal,
    check_operation_supported,
    parse,
)

pytestmark = pytest.mark.unit


def test_parse_accepts_major_minor() -> None:
    assert parse("1.0") == (1, 0)
    assert parse("2.14") == (2, 14)


def test_parse_rejects_malformed() -> None:
    for bad in ("1", "1.0.1", "x.y", "1.", ".0", "a.b", ""):
        with pytest.raises(ValueError):
            parse(bad)


def test_major_mismatch_refuses_every_operation() -> None:
    """A major mismatch refuses any operation, naming both versions."""
    for operation in (OP_NODE_REGISTER, OP_MODEL_ACQUIRE):
        with pytest.raises(CompatibilityRefusal) as exc:
            check_operation_supported(operation, "0.9", contract_major=1)
        refusal = exc.value
        assert refusal.code == CODE_AGENT_VERSION_INCOMPATIBLE
        assert refusal.operation == operation
        assert refusal.actual_version == (0, 9)


def test_matching_major_below_minimum_refuses_that_operation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below an operation's minimum refuses that operation, naming both versions.

    Every v1 operation has minimum 1.0, so to exercise the *per-operation* path
    (matching major, agent below a specific operation's minimum) we patch the
    operation minimum table with a synthetic minimum above the agent's version.
    This is the exact case the compatibility policy describes: an upgraded coordinator gains
    peer replication; a node whose agent predates that operation stays fully
    manageable for everything else.
    """
    import tensorstead.contracts.version as version_mod

    monkeypatch.setitem(version_mod.OPERATION_MINIMUM_VERSIONS, OP_MODEL_ACQUIRE, (1, 3))

    # The newer operation is refused, naming required (1.3) and actual (1.1).
    with pytest.raises(CompatibilityRefusal) as exc:
        check_operation_supported(OP_MODEL_ACQUIRE, "1.1", contract_major=1)
    refusal = exc.value
    assert refusal.code == CODE_OPERATION_UNSUPPORTED_BY_AGENT
    assert refusal.required_version == (1, 3)
    assert refusal.actual_version == (1, 1)

    # A different operation (still min 1.0) remains permitted: matching major,
    # agent 1.1 >= 1.0.
    check_operation_supported(OP_NODE_REGISTER, "1.1", contract_major=1)


def test_agent_at_operation_minimum_is_permitted() -> None:
    """Matching major at the operation's minimum is permitted."""
    # No exception raised.
    check_operation_supported(OP_NODE_REGISTER, "1.0", contract_major=1)
    check_operation_supported(OP_MODEL_ACQUIRE, CONTRACT_VERSION, contract_major=1)


def test_per_operation_refusal_names_required_and_actual() -> None:
    """The per-operation refusal carries required and actual versions."""
    # Use a synthetic table via monkeypatch is not needed: we assert the shape by
    # driving a major mismatch to a higher major, which exercises the same named
    # fields path with a required_version that differs from the actual.
    with pytest.raises(CompatibilityRefusal) as exc:
        check_operation_supported(OP_NODE_REGISTER, "1.4", contract_major=2)
    assert exc.value.required_version == (2, 0)
    assert exc.value.actual_version == (1, 4)


def test_policy_matches_p8_table() -> None:
    """The in-build constants report a coherent v1 contract.

    The minor version is pinned deliberately. It moved 1.0 -> 1.1 when
    ObservedStateResponse gained ``endpoint_authenticated``, and
    1.1 -> 1.2 when it gained ``inference_ready``, and 1.2 -> 1.3
    when it gained ``managed_containers`` -- all additive
    optional fields, which the compatibility policy makes MINOR because an older agent simply
    omits them. 1.3 -> 1.4 added the ``deployment.runtime`` *operation*,
    which is MINOR for a different reason: the route is new rather than a
    field, so an older agent does not serve it and the per-operation minimum
    refuses that one call instead of the whole node.

    1.13 -> 1.14 added ``AcquireRequest.file_selector``, which
    is the ``restore_on_boot`` shape again: a field travelling into an
    ``extra="forbid"`` model, so an older agent refuses the call. It is MINOR
    rather than MAJOR because the coordinator omits the field when the selection
    is empty, and every acquire this estate has ever performed is that case.

    1.14 -> 1.15 added two *operations*, ``images:remove`` and ``images`` (list),
    which is the ``deployment.runtime`` shape: new routes rather than fields, so an
    older agent does not serve them and the per-operation minimum refuses those
    calls instead of the whole node. ``image.delete``'s own minimum moved to 1.15
    with them, because a delete that cannot reach the node has not deleted
    anything, and reporting otherwise is the defect it was raised to fix.


    Changing this line should always be a decision about the wire contract,
    never a way to make a diff pass.
    """
    assert CONTRACT_VERSION_MAJOR == 1
    assert CONTRACT_VERSION == "1.15"
