"""An image the product produced must be runnable, and its failures legible.

The design let an operator build a runtime image on a node, and the first real
build on hardware found two things that no fake had ever exercised:

1. ``create_deployment`` pulled the image unconditionally, so a locally built
   reference — which names no registry repository — could never start. The
   product could produce an image that nothing was able to run.
2. Every image failure was raised and then dropped. ``ImageBuildError`` had no
   handler anywhere in the agent, so a build that failed on the node reached
   the operator as ``internal_error: unexpected internal error``: a terminal
   outcome naming neither the reference nor the cause, which is precisely what
   this design exists to prevent.

Both were invisible to the suite because the tests that covered the build path
stopped at "the build returns an identifier" and never asked whether a
deployment could then use it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app
from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.coordinator.node_http import NodeHTTPClient
from tensorstead.domain.models import Node
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager
from tests.helpers import make_vllm_model_dir

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_BUILT = "local/vllm:26.07-xgrammar"


def _create(client: TestClient, image_reference: str, model_path: str) -> Any:
    return client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": "01J00000000000000000000000",
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": image_reference,
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers=_AUTH,
    )


def _agent(engine: Any, tmp_path: Path) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def test_a_locally_built_image_starts_and_is_never_pulled(tmp_path: Path) -> None:
    """The image the node already holds is used as-is.

    ``unpullable`` is not a contrivance: a pull of ``local/...`` genuinely
    fails, because the reference names no registry repository. If the start
    path reaches for the registry at all, this deployment cannot be created —
    which is exactly what happened on the appliance.
    """
    engine = FakeContainerEngine()
    engine.local_images[_BUILT] = "sha256:9d91468dbuilt"
    engine.unpullable.add(_BUILT)

    resp = _create(_agent(engine, tmp_path), _BUILT, make_vllm_model_dir(tmp_path))

    assert resp.status_code == 200
    assert resp.json()["image_digest"] == "sha256:9d91468dbuilt"
    assert _BUILT not in engine.pulled


def test_an_image_the_node_lacks_is_still_pulled(tmp_path: Path) -> None:
    """Local-first must not become never-pull: an absent image is fetched."""
    engine = FakeContainerEngine()

    resp = _create(
        _agent(engine, tmp_path), "nvcr.io/nvidia/vllm:26.07-py3", make_vllm_model_dir(tmp_path)
    )

    assert resp.status_code == 200
    assert engine.pulled == ["nvcr.io/nvidia/vllm:26.07-py3"]


def test_a_failed_build_names_the_reference_and_the_cause(tmp_path: Path) -> None:
    """A build failure leaves the node as a structured error."""

    class FailingEngine(FakeContainerEngine):
        def build_image(
            self,
            *,
            reference: str,
            base_image: str,
            steps: list[str],
            entrypoint: list[str] | None = None,
        ) -> str:
            raise ImageBuildError(
                "The command '/bin/sh -c pip install xgrammar' returned a non-zero code: 1",
                reference=reference,
            )

    resp = _agent(FailingEngine(), tmp_path).post(
        "/agent/v1/images:build",
        json={"reference": _BUILT, "base_image": "nvcr.io/nvidia/vllm:26.07-py3", "steps": ["x"]},
        headers=_AUTH,
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "image_build_failed"
    assert "non-zero code: 1" in body["message"]
    assert body["detail"]["reference"] == _BUILT
    # The failure that started this: a body that named nothing at all.
    assert body["message"] != "Internal Server Error"


def test_the_coordinator_keeps_the_agents_own_error(tmp_path: Path) -> None:
    """A structured agent error survives the hop to the coordinator.

    FastAPI's ``HTTPException`` nests the error under ``detail`` while the
    agent's own handlers put it at the top level. Reading only the top level
    reduced every nested error to ``http_error`` carrying the raw JSON blob,
    so an agent that said exactly what went wrong was reported as if it had
    said nothing.
    """
    node = Node(
        id="01J0000000000000000000000A",
        name="spark-01",
        agent_endpoint="https://10.0.0.11:8443",
        agent_cert_fingerprint="",
        agent_contract_version="1.1",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )
    client = NodeHTTPClient(management_token="t")

    nested = httpx.Response(
        401, json={"detail": {"code": "unauthorized", "message": "management token required"}}
    )
    top_level = httpx.Response(
        500, json={"code": "image_build_failed", "message": "step 2 failed", "detail": {"r": "x"}}
    )

    from_nested = client._error_from_response(node, nested)
    from_top = client._error_from_response(node, top_level)

    assert (from_nested.code, from_nested.message) == ("unauthorized", "management token required")
    assert (from_top.code, from_top.message) == ("image_build_failed", "step 2 failed")
