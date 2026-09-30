"""The runtime's own account of itself.

An earlier change made the product able to report *that* a deployment stopped serving.
It still could not say **why**. On 2026-08-10 the answer existed for the whole
incident window, in vLLM's output on the node, reachable only by an operator
with SSH — which is the position the management plane exists to remove.

Three facts, none of them computed by the product: the argv the runtime was
actually launched with, how many times the engine has relaunched it, and the
last N lines it wrote.

The argv matters on its own. It is read back from the container rather than
recomputed from the deployment revision, so it can *disagree* with the revision
— which is the only way to catch a container still running yesterday's
arguments. A value derived from config could only ever agree with config.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


def _running_deployment(client: TestClient) -> str:
    node_id = client.post(
        "/v1/nodes",
        json={"name": "spark-alpha", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    ).json()["id"]
    resp = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "nvidia/Qwen3.6-27B-NVFP4",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    # Acquisition runs on a background thread; poll to terminal before reading the model list.
    poll_operation(client, resp.json()["operation_id"])
    model_id = next(m["id"] for m in client.get("/v1/models", headers=_AUTH).json())
    create = client.post(
        "/v1/deployments",
        json={
            "name": "qwen36-27b",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "local/vllm:26.07-xgrammar-0.2.1",
            "runtime_config": {"tensor_parallel_size": 1},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert create.status_code == 202, create.text
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(
        next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "qwen36-27b")
    )
    client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    return dep_id


def _runtime(client: TestClient, dep_id: str, **params: object) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    path = f"/v1/deployments/{dep_id}/runtime" + (f"?{query}" if query else "")
    response = client.get(path, headers=_AUTH)
    assert response.status_code == 200, response.text
    return dict(response.json())


def _only_node(body: dict) -> dict:
    return next(iter(body["per_node"].values()))


def test_a_failed_runtimes_last_words_reach_the_operator() -> None:
    """The whole point: vLLM's reason, without SSH.

    The August incident's shape — a runtime that died, and a control plane that
    could describe the corpse but not the cause.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.deployments[dep_id].running = False
    agent.deployments[dep_id].runtime_log = (
        "INFO 08-10 22:14:01 loading model...\n"
        "ERROR 08-10 22:14:09 ValueError: speculative decoding method 'mtp' "
        "requires the checkpoint to provide MTP weights\n"
    )

    report = _only_node(_runtime(client, dep_id))

    assert "requires the checkpoint to provide MTP weights" in report["log_tail"]
    assert report["running"] is False


def test_the_argv_is_what_ran_not_what_was_declared() -> None:
    """Read back from the container, so it can disagree with the revision.

    A container still running yesterday's arguments is invisible to any check
    that recomputes argv from config — such a check compares config to itself
    and always agrees.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.deployments[dep_id].argv = ["vllm", "serve", "--model", "/models/old", "--enforce-eager"]

    report = _only_node(_runtime(client, dep_id))

    assert report["argv"] == [
        "vllm",
        "serve",
        "--model",
        "/models/old",
        "--enforce-eager",
    ]


def test_a_crash_loop_is_distinguishable_from_a_healthy_runtime() -> None:
    """Both read as "running" at the instant you happen to look.

    Only the relaunch count separates a stable deployment from one dying and
    being restarted by its unit over and over.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.deployments[dep_id].restart_count = 47

    assert _only_node(_runtime(client, dep_id))["restart_count"] == 47


def test_the_tail_length_is_honoured() -> None:
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.deployments[dep_id].runtime_log = "\n".join(f"line {n}" for n in range(500))

    report = _only_node(_runtime(client, dep_id, tail=5))

    assert report["log_tail"].splitlines() == [f"line {n}" for n in range(495, 500)]


def test_an_absurd_tail_is_refused_rather_than_served() -> None:
    """A bounded read. An unbounded one turns an observation into a transfer."""
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    assert (
        client.get(f"/v1/deployments/{dep_id}/runtime?tail=100000", headers=_AUTH).status_code
        == 422
    )
    assert client.get(f"/v1/deployments/{dep_id}/runtime?tail=0", headers=_AUTH).status_code == 422


def test_an_unreachable_node_is_named_rather_than_omitted() -> None:
    """Absence from the map would read as "that node had nothing to say"."""
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.unreachable = True

    report = _only_node(_runtime(client, dep_id))

    assert "unreachable" in str(report.get("detail", "")).lower()
    assert report.get("log_tail") is None, (
        "an unreachable node must not report an empty log, which reads as a silent runtime"
    )


def test_nothing_the_runtime_wrote_is_persisted() -> None:
    """Structurally: read through, stored nowhere.

    A runtime may log request content. The management plane must not become the
    place that content comes to rest, so the check is that a second read after
    the node goes quiet returns nothing — there is no cache to serve from.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    agent.deployments[dep_id].runtime_log = "a prompt the operator would not want stored"
    first = _only_node(_runtime(client, dep_id))
    assert "would not want stored" in first["log_tail"]

    # The runtime's output goes away. If the product kept a copy, it would
    # still be able to answer — and that is exactly what must not happen.
    agent.deployments[dep_id].runtime_log = ""
    second = _only_node(_runtime(client, dep_id))

    assert not second["log_tail"].strip()


def test_a_completed_start_does_not_claim_the_deployment_is_serving() -> None:
    """The operator's defect #4: succeeded in seconds, serving in minutes.

    `_drive_nodes(..., "start", ...)` returns once every container is created
    and started. A 27B model then spends minutes loading weights and compiling,
    during which the container is up, the port is bound, and no inference is
    possible. The operation record said `succeeded` and an operator read it as
    "it is up".

    Success is still success — the containers really did start. It just says
    which fact it established.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    response = client.post(f"/v1/deployments/{dep_id}:restart", headers=_AUTH)

    assert response.status_code == 202, response.text
    notes = response.json().get("warnings", [])
    assert notes, "a restart reported success with no word about what that means"
    assert any("not serving until it has loaded" in note for note in notes)
    assert any("deployment show" in note for note in notes), (
        "the note must name the command that answers the question it raises"
    )


def test_the_declared_running_revision_is_not_an_observation() -> None:
    """Written because a start call returned, not because anything was seen.

    Keeping it is right — it records the revision the nodes were last asked to
    run, which is a real and useful fact. What it must not be mistaken for is
    the measured one, which lives per-node in observed state and comes from the
    container's own label, and the two are deliberately separate.
    """
    agent = FakeNodeAgent()
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    dep_id = _running_deployment(client)

    # The node is actually running something else -- a container that survived
    # an earlier revision, which is precisely what the August incident looked
    # like from the outside.
    agent.deployments[dep_id].running_revision = 7

    body = client.get(f"/v1/deployments/{dep_id}", headers=_AUTH).json()
    observed = next(iter(body["observed"]["per_node"].values()))

    assert body["declared"]["running_revision"] == 1, (
        "the declared value records what the nodes were asked to run and must "
        "not be quietly corrected by an observation"
    )
    assert observed["running_revision"] == 7, "the measured value must be reported as measured"
    assert any(d["kind"] == "revision_mismatch" for d in body["divergences"]), (
        "declared and observed disagreed and nothing said so"
    )


def test_an_entrypoint_that_ignores_its_argv_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entrypoint rule, and the reason it needed something that can notice.

    The recipe this was written against builds its own complete `vllm serve`
    from environment variables, ignoring what it was handed. A deployment like
    that has a declared `runtime_config` the runtime never read — change a value
    and nothing happens, and nothing says so.

    It is invisible from every other angle: the *configured* argv reads exactly
    right, which is what `deployment runtime` showed before this.
    """
    from tensorstead.agent.routes.runtime_logs import _argv_honoured

    configured = ["vllm", "serve", "--model", "/models/deepseek", "--tensor-parallel-size", "2"]

    honoured = _argv_honoured(configured, ["vllm serve --model /models/deepseek --tp 2"])
    ignored = _argv_honoured(configured, ["vllm serve --model /some/other/path --tp 4"])

    assert honoured is True
    assert ignored is False, "an entrypoint that rebuilt its own command line was not noticed"


def test_an_unestablished_answer_is_not_rendered_as_either() -> None:
    """No process listing means unknown, which is neither honoured nor ignored."""
    from tensorstead.agent.routes.runtime_logs import _argv_honoured

    assert _argv_honoured(["vllm", "serve", "--model", "/x"], None) is None
    assert _argv_honoured(["vllm", "serve", "--model", "/x"], []) is None
    # Nothing distinctive to look for -- bare switches are too common to be
    # evidence either way.
    assert _argv_honoured(["--enforce-eager"], ["vllm serve"]) is None
