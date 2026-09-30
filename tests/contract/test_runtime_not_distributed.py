"""Contract test the non-distributed rejection.

A multi-node request against llama.cpp is refused with a stated reason, and
**no attempt to distribute is made**. Those are two separate claims and the
second is the one that matters: the product coordinates, the runtime
distributes, and where a runtime cannot distribute there is no fallback path in
which we quietly do it ourselves.

llama.cpp is the real counter-example here rather than a stub declaring
``supports_distributed = False`` — a stub would only prove the check reads an
attribute, not that a shipped runtime actually trips it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def _register(client: TestClient, name: str, endpoint: str) -> str:
    resp = client.post("/v1/nodes", json={"name": name, "agent_endpoint": endpoint}, headers=_AUTH)
    assert resp.status_code == 201
    return str(resp.json()["id"])


def _acquire(client: TestClient, nodes: list[str]) -> str:
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/model",
            "revision": "e1f2a3b",
            "nodes": nodes,
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    models = client.get("/v1/models", headers=_AUTH).json()
    return str(next(m["id"] for m in models if m["source_model_id"] == "org/model"))


# ------------------------------------------------------- capability declaration
def test_llamacpp_declares_it_cannot_distribute() -> None:
    """The counter-example is real: a shipped adapter that says no."""
    assert LlamaCppAdapter.supports_distributed is False
    assert VLLMAdapter.supports_distributed is True


def test_both_runtimes_are_reachable_through_runtime_list(client: TestClient) -> None:
    """``runtime.list`` surfaces the capability an operator needs."""
    runtimes = client.get("/v1/runtimes", headers=_AUTH).json()
    by_type = {r["type"]: r for r in runtimes}
    assert by_type["vllm"]["supports_distributed"] is True
    assert by_type["llamacpp"]["supports_distributed"] is False


# --------------------------------------------------------------- the rejection
def test_multi_node_against_llamacpp_is_rejected_with_a_reason(client: TestClient) -> None:
    """Two nodes + a runtime that cannot distribute → a stated refusal."""
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    model_id = _acquire(client, [node_a, node_b])

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "llama-two-node",
            "model_id": model_id,
            "runtime_type": "llamacpp",
            "runtime_version": "b4000",
            "image_reference": "repo/llamacpp:tag",
            "runtime_config": {"n_gpu_layers": -1},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "runtime_not_distributed"
    # The reason names the runtime, so the operator knows what to change.
    assert "llamacpp" in body["message"]


def test_the_rejected_deployment_was_never_created(client: TestClient) -> None:
    """A refusal leaves no partial deployment behind."""
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    model_id = _acquire(client, [node_a, node_b])

    client.post(
        "/v1/deployments",
        json={
            "name": "llama-two-node",
            "model_id": model_id,
            "runtime_type": "llamacpp",
            "runtime_version": "b4000",
            "image_reference": "repo/llamacpp:tag",
            "runtime_config": {},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )

    assert client.get("/v1/deployments", headers=_AUTH).json() == []


def test_no_attempt_to_distribute_was_made(client: TestClient) -> None:
    """The refusal precedes any host contact — nothing was asked of any agent.

    The real content: there is no code path that reacts to "this runtime
    cannot distribute" by distributing on its behalf. The agent is never called.
    """
    app, _ = build_test_coordinator()
    client = TestClient(app)
    agent = app.state.fake_agent  # type: ignore[attr-defined]

    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    model_id = _acquire(client, [node_a, node_b])
    deployments_before = len(agent.deployments)

    client.post(
        "/v1/deployments",
        json={
            "name": "llama-two-node",
            "model_id": model_id,
            "runtime_type": "llamacpp",
            "runtime_version": "b4000",
            "image_reference": "repo/llamacpp:tag",
            "runtime_config": {},
            "participating_nodes": [node_a, node_b],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )

    assert len(agent.deployments) == deployments_before, (
        "a rejected multi-node request must not reach any agent"
    )


# ----------------------------------------------------------- single node is fine
def test_single_node_against_llamacpp_is_permitted(client: TestClient) -> None:
    """The refusal is about distribution, not about the runtime."""
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    model_id = _acquire(client, [node_a])

    resp = client.post(
        "/v1/deployments",
        json={
            "name": "llama-one-node",
            "model_id": model_id,
            "runtime_type": "llamacpp",
            "runtime_version": "b4000",
            "image_reference": "repo/llamacpp:tag",
            "runtime_config": {"n_gpu_layers": -1, "ctx_size": 8192},
            "participating_nodes": [node_a],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert resp.status_code == 202


def test_modify_cannot_widen_a_llamacpp_deployment_to_two_nodes(client: TestClient) -> None:
    """The check holds on the modify path too, not only on create."""
    node_a = _register(client, "spark-01", "https://10.0.0.11:8443")
    node_b = _register(client, "spark-02", "https://10.0.0.12:8443")
    model_id = _acquire(client, [node_a, node_b])

    created = client.post(
        "/v1/deployments",
        json={
            "name": "llama-one-node",
            "model_id": model_id,
            "runtime_type": "llamacpp",
            "runtime_version": "b4000",
            "image_reference": "repo/llamacpp:tag",
            "runtime_config": {},
            "participating_nodes": [node_a],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(deployments[0]["declared"]["id"])

    resp = client.patch(
        f"/v1/deployments/{dep_id}",
        json={"participating_nodes": [node_a, node_b]},
        headers=_AUTH,
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "runtime_not_distributed"


# ---------------------------------------------------------------- config schema
def test_llamacpp_config_is_its_own_schema(client: TestClient) -> None:
    """No cross-runtime config union — vLLM keys are rejected."""
    adapter = LlamaCppAdapter()
    validated = adapter.validate_config({"n_gpu_layers": 32, "ctx_size": 4096})
    assert validated == {"n_gpu_layers": 32, "ctx_size": 4096}

    with pytest.raises(ValueError, match="invalid llamacpp config"):
        adapter.validate_config({"tensor_parallel_size": 2})


def test_llamacpp_launch_args_derive_the_model_path() -> None:
    """The operator cannot point the runtime at weights it has not acquired."""
    args = LlamaCppAdapter().build_launch_args(
        {"n_gpu_layers": 32, "ctx_size": 4096}, model_path="/var/lib/tensorstead/models/m"
    )
    assert args[:2] == ["--model", "/var/lib/tensorstead/models/m"]
    assert "--n-gpu-layers" in args and "32" in args


def test_vllm_launch_args_support_qwen_reasoning_and_tool_calling() -> None:
    """A vLLM image receives the Qwen 3.6 serving options unchanged."""
    args = VLLMAdapter().build_launch_args(
        {
            "quantization": "modelopt",
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
            "enable_auto_tool_choice": True,
            "max_num_seqs": 4,
            "served_model_name": "qwen36-27b",
        },
        model_path="/var/lib/tensorstead/models/huggingface:nvidia/Qwen3.6-27B-NVFP4",
    )
    assert args == [
        "--model",
        "/var/lib/tensorstead/models/huggingface:nvidia/Qwen3.6-27B-NVFP4",
        "--quantization",
        "modelopt",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--enable-auto-tool-choice",
        "--max-num-seqs",
        "4",
        "--served-model-name",
        "qwen36-27b",
    ]
