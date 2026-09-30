"""Domain invariants for Deployment / DeploymentRevision.

Assert the properties the design calls out as tests, not incidental
consequences:

- ``DeploymentRevision`` is immutable — a frozen dataclass, and its
  fields cannot be reassigned.
- Revision numbering is 1-based, monotonic, and gapless.
- ``Deployment.id`` is stable across revisions.
- A revision must name at least one participating node.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import (
    Deployment,
    DeploymentRevision,
)

pytestmark = pytest.mark.unit

REV_KWARGS: dict[str, Any] = {
    "model_id": new_ulid(),
    "model_source_id": "huggingface",
    "source_model_id": "org/model",
    "resolved_revision": "e1f2a3b",
    "revision_pinned": True,
    "runtime_type": "vllm",
    "runtime_version": "0.6.0",
    "image_reference": "repo/vllm:tag",
    "image_digest": "sha256:abc",
    "runtime_config": {},
    "origin_platform_facts": {},
}


def make_deployment(revision: int = 1) -> Deployment:
    return Deployment(
        id=new_ulid(),
        name="llama-70b",
        desired_state="stopped",
        current_revision=revision,
    )


def make_revision(deployment_id: str, revision: int, **overrides: Any) -> DeploymentRevision:
    kwargs: dict[str, Any] = dict(REV_KWARGS)
    kwargs.update(overrides)
    return DeploymentRevision(
        deployment_id=deployment_id,
        revision=revision,
        participating_nodes=("01J00000000000000000000000",),
        endpoint="10.0.0.11:8000",
        **kwargs,
    )


def test_revision_is_frozen() -> None:
    """DeploymentRevision is immutable."""
    rev = make_revision(new_ulid(), 1)
    with pytest.raises((AttributeError, TypeError)):
        rev.revision = 2  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        rev.participating_nodes = ()  # type: ignore[misc]


def test_revision_must_be_one_based() -> None:
    """Revision numbering is 1-based."""
    with pytest.raises(ValueError):
        make_revision(new_ulid(), 0)
    with pytest.raises(ValueError):
        make_revision(new_ulid(), -1)


def test_deployment_id_stable_across_revisions() -> None:
    """Deployment.id is stable across revisions."""
    deployment = make_deployment(revision=3)
    rev1 = make_revision(deployment.id, 1)
    rev3 = make_revision(deployment.id, 3)
    assert rev1.deployment_id == deployment.id
    assert rev3.deployment_id == deployment.id


def test_revision_numbering_is_monotonic_and_gapless() -> None:
    """A sequence of accepted modifications is monotonic and gapless."""
    deployment = make_deployment(revision=3)
    revisions = [make_revision(deployment.id, n) for n in (1, 2, 3)]
    numbers = [r.revision for r in revisions]
    assert numbers == sorted(numbers)
    # gapless: exactly {1, 2, 3} with no holes
    assert numbers == list(range(1, len(numbers) + 1))


def test_revision_requires_at_least_one_node() -> None:
    """A deployment must name one or more participating nodes."""
    with pytest.raises(ValueError):
        DeploymentRevision(
            deployment_id=new_ulid(),
            revision=1,
            participating_nodes=(),
            endpoint="10.0.0.11:8000",
            **REV_KWARGS,
        )


def test_deployment_current_revision_one_based() -> None:
    with pytest.raises(ValueError):
        Deployment(id=new_ulid(), name="x", desired_state="stopped", current_revision=0)
