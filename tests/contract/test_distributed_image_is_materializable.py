"""A distributed image is not distributed until a container can be made from it.

The seam this test records. Tensorstead reported ``images:distribute`` as
200 OK onto a TP=2 worker whose copy of the image was missing its config and
layer content; the failure surfaced two minutes later as a bare HTTP 500 from
the deployment route, naming neither the image nor the distribution hop::

    failed to read config content: NotFound:
    content digest sha256:d670b497...: not found

Every existing check passed on that image. It loaded, it reported the expected
id, and it resolved locally -- because the daemon's content store is addressed
by digest, so a manifest whose blobs never arrived is still recorded under the
id the manifest names. Identity was verified; integrity never was.

These tests hold the arrival path to the stronger claim, on both routes that
put an image on a node: the import route and the peer-distribution service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.image_distribution import ImageDistributionError, ImageDistributionService

pytestmark = pytest.mark.contract

_IMAGE_ID = "sha256:b89f26d7ac24968f8f6b5675f23520a5fe3471b6d03d0d89b6484da221f9de51"


class _Engine:
    """Loads under the expected id; materializability is set per test."""

    def __init__(self, *, materializable: bool, removal_fails: bool = False) -> None:
        self.materializable = materializable
        self.removal_fails = removal_fails
        self.removed: list[str] = []
        self.checked: list[str] = []

    def import_image(self, *, archive_path: str) -> str:
        return _IMAGE_ID

    def verify_image_materializable(self, *, image_id: str) -> None:
        self.checked.append(image_id)
        if not self.materializable:
            raise ImageBuildError(
                "failed to read config content: NotFound: content digest "
                "sha256:d670b497aa3cb567cc78260eacdfdaa682e5f4ef5d81c9d048200d28eb8050ec: "
                "not found",
                reference=image_id,
            )

    def remove_image(self, *, image_id: str, force: bool = False) -> None:
        if self.removal_fails:
            raise RuntimeError("daemon refused the removal")
        self.removed.append(image_id)


def _staged(tmp_path: Path) -> Path:
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"archive bytes")
    return archive


def test_an_image_that_cannot_become_a_container_is_refused(tmp_path: Path) -> None:
    """The identity check passes and the distribution must still fail.

    This is the exact shape of the production failure: the id is right, so
    every comparison the product made before this change was satisfied.
    """
    engine = _Engine(materializable=False)
    service = ImageDistributionService(engine, str(tmp_path))

    with pytest.raises(ImageDistributionError) as caught:
        service._load_and_verify(
            _staged(tmp_path), expected_image_id=_IMAGE_ID, reference="local/tp2:ray257"
        )

    assert engine.checked == [_IMAGE_ID], "the check must run on the arrival path"
    assert "not usable on this node" in str(caught.value)


def test_a_refused_image_is_removed_rather_than_left_in_the_store(tmp_path: Path) -> None:
    """Removal is load-bearing here in a way it is not for a mismatch.

    Incomplete content in the daemon's store is sticky: a later ``docker load``
    of the same id deduplicates against it, adds the tag and reports success
    without repairing anything. Left behind, this does not merely poison one
    tag -- it makes every retry reproduce the fault and look like a fresh
    failure. Re-distributing under a new tag is precisely what an operator
    tries next, and it is what silently inherited the broken state.
    """
    engine = _Engine(materializable=False)
    service = ImageDistributionService(engine, str(tmp_path))

    with pytest.raises(ImageDistributionError):
        service._load_and_verify(
            _staged(tmp_path), expected_image_id=_IMAGE_ID, reference="local/tp2:ray257"
        )

    assert engine.removed == [_IMAGE_ID]


def test_a_failed_removal_is_reported_without_masking_the_refusal(tmp_path: Path) -> None:
    """Both facts reach the operator, and the refusal is still a refusal.

    The retry warning is the actionable half: an operator who does not know the
    image is still there will re-distribute into the same broken content.
    """
    engine = _Engine(materializable=False, removal_fails=True)
    service = ImageDistributionService(engine, str(tmp_path))

    with pytest.raises(ImageDistributionError) as caught:
        service._load_and_verify(
            _staged(tmp_path), expected_image_id=_IMAGE_ID, reference="local/tp2:ray257"
        )

    message = str(caught.value)
    assert "not usable on this node" in message
    assert "could not be removed" in message
    assert "retry will inherit this state" in message


def test_a_whole_image_is_accepted_and_kept(tmp_path: Path) -> None:
    """The check must not refuse the ordinary case it was added to protect."""
    engine = _Engine(materializable=True)
    service = ImageDistributionService(engine, str(tmp_path))

    loaded = service._load_and_verify(
        _staged(tmp_path), expected_image_id=_IMAGE_ID, reference="local/tp2:ray257"
    )

    assert loaded == _IMAGE_ID
    assert engine.checked == [_IMAGE_ID]
    assert engine.removed == []


def test_the_staged_archive_is_removed_even_when_the_image_is_refused(
    tmp_path: Path,
) -> None:
    """A half-verified transfer must not leave bytes a later import could pick up."""
    engine = _Engine(materializable=False)
    service = ImageDistributionService(engine, str(tmp_path))
    staged = _staged(tmp_path)

    with pytest.raises(ImageDistributionError):
        service._load_and_verify(staged, expected_image_id=_IMAGE_ID, reference="local/tp2:ray257")

    assert not staged.exists()
