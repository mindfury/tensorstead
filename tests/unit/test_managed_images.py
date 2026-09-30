"""Managed runtime images.

Written because the product could not repair a broken upstream image: when the
NVIDIA vLLM image shipped an xgrammar too old for its own tool-calling path,
the only route available was SSH — the workflow this design exists to replace.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tensorstead.domain.errors import NotFoundError, StillReferencedError
from tensorstead.domain.models import ImageBuildSpec, ImageOrigin, ImageRecord
from tensorstead.service.models_ import ImageBuildService

pytestmark = pytest.mark.unit

_FIX = "python3 -m pip install --no-deps xgrammar==0.2.1 apache-tvm-ffi==0.1.9"


class _Repo:
    def __init__(self) -> None:
        self.specs: dict[str, ImageBuildSpec] = {}
        self.images: list[ImageRecord] = []
        self.nodes: dict[str, Any] = {"node-a": object()}
        self.deployments: list[Any] = []

    def save_build_spec(self, spec: ImageBuildSpec) -> None:
        self.specs[spec.name] = spec

    def list_build_specs(self) -> list[ImageBuildSpec]:
        return list(self.specs.values())

    def get_build_spec(self, name: str) -> ImageBuildSpec | None:
        return self.specs.get(name)

    def delete_build_spec(self, name: str) -> None:
        self.specs.pop(name, None)

    def save_image(self, image: ImageRecord) -> None:
        self.images.append(image)

    def list_images(self) -> list[ImageRecord]:
        return list(self.images)

    def get_node(self, node_id: str) -> Any:
        return self.nodes.get(node_id)

    def list_deployments(self) -> list[Any]:
        return list(self.deployments)

    def get_revision(self, deployment_id: str, revision: int) -> Any:
        return None


class _Client:
    def __init__(self, image_id: str = "sha256:built") -> None:
        self.image_id = image_id
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def build_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("build", payload))
        return {"image_id": self.image_id}

    def import_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("import", payload))
        return {"image_id": self.image_id}


def _service(client: _Client | None = None) -> tuple[ImageBuildService, _Repo, _Client]:
    repo, cli = _Repo(), client or _Client()
    return ImageBuildService(repo, cli), repo, cli


def test_recording_a_spec_executes_nothing() -> None:
    """Recording is data; building is a separate operation."""
    service, repo, client = _service()

    result = service.record_spec(
        "vllm-xgrammar",
        base_image="nvcr.io/nvidia/vllm@sha256:abc",
        steps=[_FIX],
    )

    assert client.calls == [], "recording must not build"
    assert repo.specs["vllm-xgrammar"].steps == (_FIX,)
    assert result["base_is_pinned"] is True
    assert "warning" not in result


def test_an_unpinned_base_is_reported_and_never_refused() -> None:
    """Pinning is the operator's judgement."""
    service, repo, _ = _service()

    result = service.record_spec(
        "loose", base_image="nvcr.io/nvidia/vllm:26.07-py3", steps=["true"]
    )

    assert result["base_is_pinned"] is False
    assert "not reproducible" in result["warning"]
    assert "loose" in repo.specs, "it must still be recorded, not rejected"


def test_building_records_provenance_and_does_not_claim_a_registry_digest() -> None:
    """A locally built image has no registry digest."""
    service, repo, _ = _service()
    service.record_spec("s", base_image="base@sha256:abc", steps=["true"])

    result = service.build("s", node_id="node-a", reference="local/vllm:patched")

    assert result["origin"] == "built"
    assert result["produced_by"] == "s"
    assert result["is_registry_digest"] is False
    assert "digest" not in result, "the field is image_id; conflating them misleads"

    recorded = repo.images[0]
    assert recorded.origin is ImageOrigin.BUILT
    assert recorded.produced_by == "s"
    assert recorded.is_registry_digest is False


def test_importing_records_its_archive_as_provenance() -> None:
    service, repo, client = _service()

    result = service.import_archive(
        node_id="node-a",
        reference="local/vllm:vendor",
        archive_name="vllm-vendor.tar",
        expected_image_id=None,
    )

    assert result["origin"] == "imported"
    assert result["produced_by"] == "vllm-vendor.tar"
    assert repo.images[0].origin is ImageOrigin.IMPORTED
    # The agent receives a name, never a path (the MCP guardrail's rule).
    assert client.calls[0][1]["archive_name"] == "vllm-vendor.tar"


def test_building_an_unrecorded_spec_is_refused() -> None:
    service, _, client = _service()

    with pytest.raises(NotFoundError):
        service.build("absent", node_id="node-a", reference="x")
    assert client.calls == []


def test_building_on_an_unregistered_node_is_refused() -> None:
    service, _, client = _service()
    service.record_spec("s", base_image="base@sha256:abc", steps=[])

    with pytest.raises(NotFoundError):
        service.build("s", node_id="ghost", reference="x")
    assert client.calls == []


def test_deleting_a_spec_whose_image_is_deployed_is_refused() -> None:
    """Deleting a deployed spec is refused by the referential rule."""
    service, repo, _ = _service()
    service.record_spec("s", base_image="base@sha256:abc", steps=[])
    service.build("s", node_id="node-a", reference="local/vllm:patched")

    class _Dep:
        id, name, current_revision = "d1", "qwen", 1

    class _Rev:
        image_reference = "local/vllm:patched"

    repo.deployments = [_Dep()]
    repo.get_revision = lambda deployment_id, revision: _Rev()  # type: ignore[method-assign]

    with pytest.raises(StillReferencedError) as caught:
        service.delete_spec("s")
    assert "qwen" in str(caught.value)
    assert "s" in repo.specs, "a refused delete must change nothing"


def test_deleting_an_unreferenced_spec_succeeds() -> None:
    service, repo, _ = _service()
    service.record_spec("s", base_image="base@sha256:abc", steps=[])

    assert service.delete_spec("s")["status"] == "deleted"
    assert "s" not in repo.specs


def test_listing_specs_reports_reproducibility() -> None:
    service, _, _ = _service()
    service.record_spec("pinned", base_image="b@sha256:abc", steps=[])
    service.record_spec("loose", base_image="b:tag", steps=[])

    listed = {s["name"]: s["base_is_pinned"] for s in service.list_specs()}
    assert listed == {"pinned": True, "loose": False}


@pytest.mark.parametrize("base", ["repo/img@sha256:deadbeef", "reg.io/x/y@sha256:0"])
def test_pinned_bases_are_recognised(base: str) -> None:
    assert ImageBuildSpec(name="n", base_image=base, steps=()).base_is_pinned


@pytest.mark.parametrize("base", ["repo/img:latest", "repo/img", "img:26.07-py3"])
def test_unpinned_bases_are_recognised(base: str) -> None:
    assert not ImageBuildSpec(name="n", base_image=base, steps=()).base_is_pinned


def test_a_build_that_returns_no_identifier_records_nothing() -> None:
    """A build with no result must not create a bogus inventory row."""
    service, repo, _ = _service(client=_Client(image_id=""))
    service.record_spec("s", base_image="b@sha256:abc", steps=[])

    service.build("s", node_id="node-a", reference="x")
    assert repo.images == []


def test_image_record_defaults_to_pulled_provenance() -> None:
    """Existing rows predate provenance; they are pulled, not unknown."""
    record = ImageRecord(node_id="n", reference="r", digest="sha256:x", pulled_at=datetime.now())
    assert record.origin is ImageOrigin.PULLED
    assert record.is_registry_digest is True
    assert record.produced_by is None


def test_a_build_that_fails_on_the_node_names_the_node_and_the_reference() -> None:
    """A node-side failure is a named outcome, not an unhandled exception.

    ``AgentCallError`` is not a ``DomainError``, so before this it escaped the
    service, hit the coordinator's catch-all handler, and reached the operator
    as ``internal_error: unexpected internal error`` — no node, no reference,
    no cause. That is the outcome the error contract forbids, and it is what
    the first real build on hardware actually produced.
    """
    from types import SimpleNamespace

    from tensorstead.domain.errors import NodeOperationFailedError
    from tensorstead.ports.node_client import AgentCallError

    class _FailingClient(_Client):
        def build_image(self, node: Any, payload: dict[str, Any]) -> dict[str, Any]:
            raise AgentCallError(
                code="image_build_failed",
                message="returned a non-zero code: 1",
                node_id="node-a",
            )

    service, repo, _ = _service(_FailingClient())
    repo.nodes["node-a"] = SimpleNamespace(name="spark-01", id="node-a")
    service.record_spec("s", base_image="nvcr.io/nvidia/vllm@sha256:abc", steps=["false"])

    with pytest.raises(NodeOperationFailedError) as caught:
        service.build("s", node_id="node-a", reference="local/vllm:patched")

    message = caught.value.message
    assert "spark-01" in message
    assert "local/vllm:patched" in message
    assert "non-zero code: 1" in message
    assert caught.value.code == "node_operation_failed"
