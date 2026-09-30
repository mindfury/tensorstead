"""Contract test the export schema.

``GET /v1/deployments/{id}/export`` returns a projection of one revision that
carries every export field (minus secrets, structurally — no entity has one),
identifies the represented revision, and renders an explicit
unpinned statement when the source never pinned.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_FIX = "python3 -m pip install --no-deps xgrammar==0.2.1 apache-tvm-ffi==0.1.9"

_AUTH = {"Authorization": "Bearer test"}

# Every field the export must carry.
_FR031_FIELDS = (
    "apiVersion",
    "kind",
    "exported_at",
    "exported_from_revision",
    "deployment",
    "model",
    "runtime",
    "image",
    "runtime_config",
    "placement",
    "origin_platform",
    "operations",
)


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _setup_deployment(client: TestClient) -> str:
    """Register → acquire → create; return the deployment id."""
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert resp.status_code == 201
    node_id = str(resp.json()["id"])

    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "llama-70b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    return str(
        next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "llama-70b")
    )


def test_export_carries_every_fr031_field(client: TestClient) -> None:
    """Every field is present in the export."""
    dep_id = _setup_deployment(client)
    export = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH)
    assert export.status_code == 200
    body = export.json()

    for field in _FR031_FIELDS:
        assert field in body, f"export missing {field!r}"

    # Identity and desired state come from the deployment.
    assert body["deployment"]["name"] == "llama-70b"
    assert body["deployment"]["id"] == dep_id
    assert body["deployment"]["desired_state"] in ("stopped", "running")


def test_export_identifies_represented_revision(client: TestClient) -> None:
    """The export names which revision it represents."""
    dep_id = _setup_deployment(client)
    body = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()
    assert body["exported_from_revision"] == 1

    # Model identity and resolved revision are present.
    model = body["model"]
    assert model["source"] == "huggingface"
    assert model["id"] == "org/model"
    assert model["revision"] == "e1f2a3b"
    assert model["revision_pinned"] is True

    # Runtime, image, configuration, placement all present.
    assert body["runtime"]["type"] == "vllm"
    assert body["runtime"]["version"] == "0.6.0"
    assert body["image"]["reference"] == "repo/vllm:tag"
    assert body["runtime_config"] == {"tensor_parallel_size": 1}
    assert body["placement"]["endpoint"] == "10.0.0.11:8000"
    assert len(body["placement"]["nodes"]) == 1


def test_export_origin_platform_records_not_constrains(client: TestClient) -> None:
    """origin_platform is recorded from the node's facts."""
    dep_id = _setup_deployment(client)
    body = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()
    origin = body["origin_platform"]
    # The fake node reports aarch64/linux and unified memory.
    assert origin["cpu_arch"] == "aarch64"
    assert origin["os_family"] == "linux"
    assert origin["memory_is_unified"] is True


def test_export_excludes_secret_material(client: TestClient) -> None:
    """No secret field appears in an export.

    This is structural — the projection draws only from a DeploymentRevision,
    which holds no field a secret value could occupy.
    """
    dep_id = _setup_deployment(client)
    body = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH).json()
    serialized = str(body).lower()
    for token in ("secret", "password", "token", "credential"):
        assert token not in serialized, f"export leaked secret-like token {token!r}"


# ------------------------------------------------ image provenance


def test_a_built_image_export_carries_the_spec_that_produced_it() -> None:
    """A reference alone cannot be reproduced on another estate.

    `local/vllm:patched` means nothing to a different estate: it cannot be
    pulled. Naming the build spec makes the image reproducible, which is what
    an export must be sufficient for.
    """
    from datetime import datetime

    from tensorstead.domain.models import ImageBuildSpec, ImageOrigin, ImageRecord
    from tensorstead.service.export import ExportService

    spec = ImageBuildSpec(
        name="vllm-xgrammar",
        base_image="nvcr.io/nvidia/vllm@sha256:abc",
        steps=(_FIX,),
    )

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list[ImageRecord]:
            return [
                ImageRecord(
                    node_id="n1",
                    reference="local/vllm:patched",
                    digest="sha256:built",
                    pulled_at=datetime.now().astimezone(),
                    origin=ImageOrigin.BUILT,
                    produced_by="vllm-xgrammar",
                )
            ]

        def get_build_spec(self, name: str) -> ImageBuildSpec | None:
            return spec if name == "vllm-xgrammar" else None

    service = ExportService(repository=_Repo())
    image = service._image_origin("local/vllm:patched")

    assert image["origin"] == "built"
    assert image["produced_by"] == "vllm-xgrammar"
    assert image["build_spec"]["base_image"] == "nvcr.io/nvidia/vllm@sha256:abc"
    assert image["build_spec"]["steps"] == [_FIX]
    assert image["build_spec"]["base_is_pinned"] is True
    assert "note" not in image, "a pinned base needs no caveat"


def test_an_unpinned_build_spec_carries_a_caveat_in_the_export() -> None:
    """A reader must not believe a rebuild reproduces the same image."""
    from datetime import datetime

    from tensorstead.domain.models import ImageBuildSpec, ImageOrigin, ImageRecord
    from tensorstead.service.export import ExportService

    spec = ImageBuildSpec(name="loose", base_image="nvcr.io/nvidia/vllm:26.07-py3", steps=())

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list[ImageRecord]:
            return [
                ImageRecord(
                    node_id="n1",
                    reference="local/vllm:loose",
                    digest="sha256:x",
                    pulled_at=datetime.now().astimezone(),
                    origin=ImageOrigin.BUILT,
                    produced_by="loose",
                )
            ]

        def get_build_spec(self, name: str) -> ImageBuildSpec | None:
            return spec

    image = ExportService(repository=_Repo())._image_origin("local/vllm:loose")

    assert image["build_spec"]["base_is_pinned"] is False
    assert "may not produce the same image" in image["note"]


def test_an_imported_image_says_the_archive_is_required() -> None:
    from datetime import datetime

    from tensorstead.domain.models import ImageOrigin, ImageRecord
    from tensorstead.service.export import ExportService

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list[ImageRecord]:
            return [
                ImageRecord(
                    node_id="n1",
                    reference="local/vllm:vendor",
                    digest="sha256:y",
                    pulled_at=datetime.now().astimezone(),
                    origin=ImageOrigin.IMPORTED,
                    produced_by="vendor.tar",
                )
            ]

        def get_build_spec(self, name: str) -> None:
            return None

    image = ExportService(repository=_Repo())._image_origin("local/vllm:vendor")

    assert image["origin"] == "imported"
    assert image["produced_by"] == "vendor.tar"
    assert "requires that archive" in image["note"]
    assert "build_spec" not in image


def test_a_pulled_image_gains_no_provenance_block() -> None:
    """An ordinary registry image needs none: the reference is reproducible."""
    from tensorstead.service.export import ExportService

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list:
            return []

        def get_build_spec(self, name: str) -> None:
            return None

    assert ExportService(repository=_Repo())._image_origin("nvcr.io/x:1") == {}


def test_the_export_states_the_digest_that_actually_ran() -> None:
    """Observed, not predicted.

    `DeploymentRevision.image_digest` is empty on every deployment and always
    was. Rather than make create pull an image — which would break the rule
    that the host stays untouched until start — the export reads back the digest
    recorded when the agent pulled it.
    """
    from datetime import datetime

    from tensorstead.domain.models import ImageOrigin, ImageRecord
    from tensorstead.service.export import ExportService

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list[ImageRecord]:
            return [
                ImageRecord(
                    node_id="n1",
                    reference="nvcr.io/nvidia/vllm:26.07-py3",
                    digest="sha256:95c498a4",
                    pulled_at=datetime.now().astimezone(),
                    origin=ImageOrigin.PULLED,
                )
            ]

        def get_build_spec(self, name: str) -> None:
            return None

    service = ExportService(repository=_Repo())

    assert service._recorded_digest("nvcr.io/nvidia/vllm:26.07-py3") == "sha256:95c498a4"
    assert service._recorded_digest("something/else:1") is None


def test_an_image_never_pulled_reports_no_digest_rather_than_a_guess() -> None:
    """Absent is an honest answer; a fabricated digest is not."""
    from tensorstead.service.export import ExportService

    class _Repo:
        def get_node(self, node_id: str) -> None:
            return None

        def list_images(self) -> list:
            return []

        def get_build_spec(self, name: str) -> None:
            return None

    assert ExportService(repository=_Repo())._recorded_digest("x:1") is None
