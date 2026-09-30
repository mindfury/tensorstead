"""Branches of the coordinator's approval check that the HTTP path cannot reach.

Two of the refusals in ``authorize_code_options`` guard states the test estate
cannot produce through the API -- the fake model source resolves every revision,
including ``main``, to a concrete sha, so no unpinned model is reachable that
way. Tested here directly rather than left unexercised: an unreachable branch in
an authorization path is a branch nobody has ever seen run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tensorstead.domain.approvals import CodeExecutionApproval
from tensorstead.domain.errors import InvalidDeploymentError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import ImageOrigin, ImageRecord, Model
from tensorstead.service.approvals import authorize_code_options, requested_code_options

pytestmark = pytest.mark.unit

_REVISION = "fab0aecb760cec45227f6656abcaafa11abca87a"
_DIGEST = "sha256:02366b8f87b8490c137b49f91beb7046904b16c70b7af2408753854d94e69170"
_IMAGE = "local/vllm:flashnext"


class _Repo:
    """Only the two reads the approval check performs."""

    def __init__(self, images: list[ImageRecord], approvals: list[CodeExecutionApproval]) -> None:
        self._images = images
        self._approvals = {a.fingerprint: a for a in approvals}

    def list_images(self) -> list[ImageRecord]:
        return list(self._images)

    def find_code_approval(self, fingerprint: str) -> CodeExecutionApproval | None:
        return self._approvals.get(fingerprint)


def _image(node_id: str, digest: str = _DIGEST, reference: str = _IMAGE) -> ImageRecord:
    return ImageRecord(
        node_id=node_id,
        reference=reference,
        digest=digest,
        pulled_at=datetime.now().astimezone(),
        origin=ImageOrigin.IMPORTED,
    )


def _model(**overrides: Any) -> Model:
    fields: dict[str, Any] = {
        "id": new_ulid(),
        "source_id": "huggingface",
        "source_model_id": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "resolved_revision": _REVISION,
        "revision_pinned": True,
    }
    fields.update(overrides)
    return Model(**fields)


def _check(repo: _Repo, model: Model, nodes: tuple[str, ...] = ("n1",)) -> Any:
    return authorize_code_options(
        repo,  # type: ignore[arg-type]
        runtime_type="vllm",
        runtime_config={"trust_remote_code": True},
        model=model,
        image_reference=_IMAGE,
        participating_nodes=nodes,
    )


def test_a_config_asking_for_nothing_approvable_costs_one_scan() -> None:
    """The ordinary case must not require an image, a pin, or an approval."""
    approved, approval = authorize_code_options(
        _Repo([], []),  # type: ignore[arg-type]
        runtime_type="vllm",
        runtime_config={"tensor_parallel_size": 2, "max_num_seqs": 8},
        model=_model(revision_pinned=False, resolved_revision=None),
        image_reference="anything",
        participating_nodes=("n1",),
    )

    assert approved == frozenset()
    assert approval is None


def test_an_unpinned_model_is_refused_before_any_lookup() -> None:
    """An approval is granted for code read at one commit."""
    with pytest.raises(InvalidDeploymentError, match="immutable"):
        _check(_Repo([_image("n1")], []), _model(revision_pinned=False))


def test_a_model_with_no_resolved_revision_is_refused() -> None:
    with pytest.raises(InvalidDeploymentError, match="immutable"):
        _check(_Repo([_image("n1")], []), _model(resolved_revision=None))


def test_an_image_absent_from_a_declared_node_is_refused() -> None:
    """Every node must hold it: the approval binds to what will actually run."""
    repo = _Repo([_image("n1")], [])

    with pytest.raises(InvalidDeploymentError, match="image"):
        _check(repo, _model(), nodes=("n1", "n2"))


def test_nodes_disagreeing_about_the_digest_are_refused() -> None:
    """A tag resolving differently per node would authorize two runtimes.

    This estate has a standing record of a reference meaning different bytes
    on two nodes actually mattering.
    """
    repo = _Repo([_image("n1"), _image("n2", digest="sha256:" + "d" * 64)], [])

    with pytest.raises(InvalidDeploymentError, match="image"):
        _check(repo, _model(), nodes=("n1", "n2"))


def test_a_matching_approval_authorizes_and_is_returned() -> None:
    approval = CodeExecutionApproval(
        id=new_ulid(),
        option="trust_remote_code",
        runtime_type="vllm",
        model_source_id="huggingface",
        source_model_id="nvidia/Qwen3.8-Flash-Next-NVFP4",
        model_revision=_REVISION,
        image_digest=_DIGEST,
        reason="vendor guidance",
        approved_by="operator",
        policy_version="2026-09-05.1",
        created_at=datetime.now().astimezone(),
    )
    repo = _Repo([_image("n1"), _image("n2")], [approval])

    approved, used = _check(repo, _model(), nodes=("n1", "n2"))

    assert approved == frozenset({"trust_remote_code"})
    assert used is not None and used.id == approval.id


def test_extra_args_is_never_treated_as_a_request_for_approval() -> None:
    """The passthrough is a separate refusal and must stay one.

    Treating an approvable option arriving through ``extra_args`` as an approval
    question would make one reviewed opening into two doors, and only one of
    them was reviewed.
    """
    assert requested_code_options({"extra_args": {"trust-remote-code": True}}) == frozenset()
    assert requested_code_options({"extra_args": {"trust_remote_code": True}}) == frozenset()


def test_a_false_value_asks_for_nothing() -> None:
    """Declaring the flag off is not a request to load code."""
    assert requested_code_options({"trust_remote_code": False}) == frozenset()
