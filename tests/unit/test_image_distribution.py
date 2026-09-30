"""Produce once, distribute.

Building a spec independently on each node yields a *different* identifier for
the same recipe. That breaks the comparison and makes
`image_digest_mismatch` divergence meaningless — a failure that looks like
success, which is the class of defect this codebase has repeatedly produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.image_distribution import ImageDistributionError, ImageDistributionService
from tensorstead.domain.models import ImageBuildSpec, ImageOrigin
from tensorstead.service.models_ import ImageBuildService

pytestmark = pytest.mark.unit


class _Node:
    def __init__(self, name: str) -> None:
        self.id = name
        self.name = name
        self.agent_endpoint = f"https://{name}:8443"


class _Repo:
    def __init__(self) -> None:
        self.specs = {"s": ImageBuildSpec(name="s", base_image="b@sha256:a", steps=())}
        self.images: list[Any] = []
        self.nodes = {"n1": _Node("spark-01"), "n2": _Node("spark-02")}

    def get_build_spec(self, name: str) -> Any:
        return self.specs.get(name)

    def save_image(self, image: Any) -> None:
        self.images.append(image)

    def get_node(self, node_id: str) -> Any:
        return self.nodes.get(node_id)

    def list_images(self) -> list[Any]:
        return list(self.images)

    def list_deployments(self) -> list[Any]:
        return []


class _Client:
    def __init__(self, arrives_as: str | Exception = "sha256:built-ref") -> None:
        self.arrives_as = arrives_as
        self.builds = 0
        self.distributes: list[dict[str, Any]] = []

    def build_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.builds += 1
        return {"image_id": "sha256:built-ref"}

    def distribute_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.distributes.append(payload)
        if isinstance(self.arrives_as, Exception):
            raise self.arrives_as
        return {"image_id": self.arrives_as}


def test_the_image_is_built_once_and_copied_to_the_rest() -> None:
    """One build, one identifier, every node."""
    client = _Client()
    service = ImageBuildService(_Repo(), client)

    result = service.build_and_distribute("s", nodes=["n1", "n2"], reference="local/v:1")

    assert client.builds == 1, "building per node defeats the entire purpose"
    assert len(client.distributes) == 1
    assert result["status"] == "succeeded"
    assert result["per_node"]["n1"]["status"] == "built"
    assert result["per_node"]["n2"]["status"] == "distributed"
    assert result["per_node"]["n2"]["image_id"] == result["image_id"]


def test_the_source_endpoint_is_the_node_that_built_it() -> None:
    client = _Client()
    ImageBuildService(_Repo(), client).build_and_distribute(
        "s", nodes=["n1", "n2"], reference="local/v:1"
    )

    assert client.distributes[0]["source_endpoint"] == "https://spark-01:8443"
    assert client.distributes[0]["expected_image_id"] == "sha256:built-ref"


def test_a_node_that_cannot_receive_makes_the_whole_operation_fail() -> None:
    """Partial success is an overall failure."""
    client = _Client(arrives_as=ConnectionError("unreachable"))
    result = ImageBuildService(_Repo(), client).build_and_distribute(
        "s", nodes=["n1", "n2"], reference="local/v:1"
    )

    assert result["status"] == "failed"
    assert result["failed_nodes"] == ["spark-02"], "the failing node must be named"
    # The node that did receive it is left alone, never unwound.
    assert result["per_node"]["n1"]["status"] == "built"


def test_an_identifier_mismatch_is_a_failure_not_a_shrug() -> None:
    """Different bytes on different nodes is the exact fault this design prevents."""
    service = ImageBuildService(_Repo(), _Client(arrives_as="sha256:something-else"))

    result = service.build_and_distribute("s", nodes=["n1", "n2"], reference="local/v:1")

    assert result["status"] == "failed"
    assert result["failed_nodes"] == ["spark-02"]
    assert result["per_node"]["n2"]["detail"] == "identifier mismatch"


def test_building_with_no_nodes_is_refused() -> None:
    from tensorstead.domain.errors import NotFoundError

    with pytest.raises(NotFoundError):
        ImageBuildService(_Repo(), _Client()).build_and_distribute("s", nodes=[], reference="x")


def test_only_the_building_node_is_recorded_when_distribution_fails() -> None:
    repo = _Repo()
    ImageBuildService(repo, _Client(arrives_as=RuntimeError("no"))).build_and_distribute(
        "s", nodes=["n1", "n2"], reference="local/v:1"
    )

    assert [i.node_id for i in repo.images] == ["n1"]
    assert repo.images[0].origin is ImageOrigin.BUILT


# --------------------------------------------------------- the agent's half


class _Engine:
    def __init__(self, loads_as: str, *, materializable: bool = True) -> None:
        self.loads_as = loads_as
        self.removed: list[str] = []
        self.removal_fails = False
        # An image the daemon holds under the right id and from which no
        # container can be created.
        self.materializable = materializable
        self.materializability_checked: list[str] = []

    def import_image(self, *, archive_path: str) -> str:
        return self.loads_as

    def verify_image_materializable(self, *, image_id: str) -> None:
        self.materializability_checked.append(image_id)
        if not self.materializable:
            raise RuntimeError(
                "failed to read config content: NotFound: content digest "
                "sha256:d670b497aa3cb567cc78260eacdfdaa682e5f4ef5d81c9d048200d28eb8050ec: "
                "not found"
            )

    def remove_image(self, *, image_id: str, force: bool = False) -> None:
        if self.removal_fails:
            raise RuntimeError("daemon refused the removal")
        self.removed.append(image_id)


def test_an_archive_arriving_as_the_wrong_image_is_rejected(tmp_path: Path) -> None:
    """Verified before it is trusted, the same rule applied to images.

    Exercises ``_load_and_verify`` directly rather than through
    ``pull_from_peer``'s network fetch: the previous version of this test
    named a non-resolving host (``peer``) and its assertion accepted either
    "could not fetch" or "expected", so it always passed on the DNS failure
    and never actually reached the load/compare logic it claimed to cover.
    """
    engine = _Engine("sha256:wrong")
    service = ImageDistributionService(engine, str(tmp_path))
    staged = tmp_path / "archive.tar"
    staged.write_bytes(b"x")

    with pytest.raises(ImageDistributionError) as caught:
        service._load_and_verify(staged, expected_image_id="sha256:right", reference="local/v:1")

    assert "arrived as 'sha256:wrong'" in str(caught.value)
    assert "expected 'sha256:right'" in str(caught.value)
    assert not staged.exists(), "the staged archive must not survive a rejection"


def test_a_rejected_archive_is_removed_from_the_daemon(tmp_path: Path) -> None:
    """Rejection must not leave the image behind.

    Reporting failure while the mismatched image stays loaded and tagged in
    the daemon is a poisoned tag a later, unrelated start could pick up --
    the harm is in what lingers, not in the error message.
    """
    engine = _Engine("sha256:wrong")
    service = ImageDistributionService(engine, str(tmp_path))
    staged = tmp_path / "archive.tar"
    staged.write_bytes(b"x")

    with pytest.raises(ImageDistributionError):
        service._load_and_verify(staged, expected_image_id="sha256:right", reference="local/v:1")

    assert engine.removed == ["sha256:wrong"]


def test_a_removal_that_also_fails_still_raises_the_identity_error(tmp_path: Path) -> None:
    """Cleanup failing must not replace or hide the fact that mattered: the mismatch."""
    engine = _Engine("sha256:wrong")
    engine.removal_fails = True
    service = ImageDistributionService(engine, str(tmp_path))
    staged = tmp_path / "archive.tar"
    staged.write_bytes(b"x")

    with pytest.raises(ImageDistributionError) as caught:
        service._load_and_verify(staged, expected_image_id="sha256:right", reference="local/v:1")

    assert "expected 'sha256:right'" in str(caught.value)
    assert "could not be removed" in str(caught.value)


def test_a_matching_archive_is_not_removed(tmp_path: Path) -> None:
    """The happy path must never call remove_image at all."""
    engine = _Engine("sha256:right")
    service = ImageDistributionService(engine, str(tmp_path))
    staged = tmp_path / "archive.tar"
    staged.write_bytes(b"x")

    result = service._load_and_verify(
        staged, expected_image_id="sha256:right", reference="local/v:1"
    )

    assert result == "sha256:right"
    assert engine.removed == []
