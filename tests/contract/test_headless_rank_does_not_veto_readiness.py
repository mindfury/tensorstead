"""A rank that was never meant to serve must not report a group broken.

Multi-node shipped and every TP=2 start failed. The cause turned out to
be `--headless`: vLLM's non-head ranks start no API server, and launching one as
a leader builds an engine core that calls `collective_rpc` on a follower
executor. The launch now derives the flag from the rank's position.

Fixing the launch exposed the next problem, before any of it reached hardware.
The product probes every participating node's endpoint and aggregates the
answers worst-case, so a headless rank — doing precisely what it was told, with
nothing listening because nothing was supposed to listen — would answer
`inference_ready: false` and veto the whole deployment. A **correctly working**
two-node cluster would have reported itself broken for as long as it ran.

That is the same defect as reporting a broken deployment healthy, in the other
direction: the record stops matching reality and nothing compares them. It is
worth stating plainly because the instinct is to treat a false negative as the
safe kind of wrong. It is not. An operator who learns that `inference_ready:
false` sometimes means "fine, actually" has lost the field entirely, and this
product's whole claim is that its records mean something.

The shape of the fix follows the seam: the **adapter** knows which ranks serve,
because that is runtime-specific; the **agent** records it on the
container at start, next to the revision and endpoint it already records; the
**coordinator** leaves non-serving ranks out of the verdict rather than letting
them cast one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.app import build_agent_app
from tensorstead.ports.runtime_adapter import NodePosition
from tensorstead.service.observation import ObservationService
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_DEPLOYMENT_ID = "01J00000000000000000000010"
_CONTAINER = f"tensorstead-{_DEPLOYMENT_ID}"

_HEAD = NodePosition(
    node_index=0,
    node_count=2,
    self_address="10.100.184.1",
    peer_addresses=["10.100.184.1", "10.100.184.2"],
)
_WORKER = NodePosition(
    node_index=1,
    node_count=2,
    self_address="10.100.184.2",
    peer_addresses=["10.100.184.1", "10.100.184.2"],
)


# --------------------------------------------------------------------------
# The adapter's declaration
# --------------------------------------------------------------------------


def test_only_the_head_declares_that_it_serves() -> None:
    """Derived from the same position that decides who gets ``--headless``.

    Stated once, in one place, so the argv and the observation cannot come to
    different conclusions about which rank serves.
    """
    adapter = VLLMAdapter()
    config = {"tensor_parallel_size": 2}

    assert adapter.container_requirements(config, position=_HEAD).serves_inference is True
    assert adapter.container_requirements(config, position=_WORKER).serves_inference is False
    # No position at all is a single-node deployment, which serves.
    assert adapter.container_requirements(config).serves_inference is True


def test_the_rank_that_serves_is_the_rank_without_headless() -> None:
    """The two derivations agree, asserted rather than assumed.

    They are computed in different methods from the same input. If they ever
    disagreed, the product would either probe a rank that cannot answer or skip
    one that can — and both failures are silent.
    """
    adapter = VLLMAdapter()
    config = {"tensor_parallel_size": 2}

    for position in (_HEAD, _WORKER):
        args = adapter.build_launch_args(config, model_path="/models/m", position=position)
        serves = adapter.container_requirements(config, position=position).serves_inference
        assert serves is ("--headless" not in args), (
            f"rank {position.node_index} has --headless={'--headless' in args} but "
            f"declares serves_inference={serves}"
        )


# --------------------------------------------------------------------------
# What the agent records and reports
# --------------------------------------------------------------------------


@pytest.fixture
def engine() -> FakeContainerEngine:
    return FakeContainerEngine()


@pytest.fixture(autouse=True)
def _no_rendezvous_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the head's rendezvous wait, which is not what these tests are about.

    Starting rank 0 of a real group blocks until its rendezvous port accepts
    — up to three minutes with nothing listening. That
    behaviour has its own suite in ``tests/integration/test_staged_distributed_start``;
    here it is 180 seconds of machinery between a deployment and the label it
    was supposed to write. Stubbed rather than satisfied with a real listener,
    which would make these tests depend on a free fixed port.
    """
    from tensorstead.agent.routes import deployments

    monkeypatch.setattr(deployments, "_await_rendezvous", lambda *a, **k: None)


def _client(engine: FakeContainerEngine, tmp_path: Path) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _deploy(client: TestClient, position: NodePosition | None) -> None:
    model_dir = cast(FastAPI, client.app).state.acquisition.local_path("qwen")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    body: dict[str, Any] = {
        "deployment_id": _DEPLOYMENT_ID,
        "revision": 7,
        "runtime_type": "vllm",
        "image_reference": "local/vllm:26.07-xgrammar-0.2.1",
        "runtime_config": {"tensor_parallel_size": 2},
        "model_path": str(model_dir),
        "endpoint": "0.0.0.0:8000",
    }
    if position is not None:
        body["node_position"] = {
            "node_index": position.node_index,
            "node_count": position.node_count,
            "self_address": position.self_address,
            "peer_addresses": list(position.peer_addresses),
        }
    response = client.post("/agent/v1/deployments", json=body, headers=_AUTH)
    assert response.status_code == 200, response.text


def _observe(client: TestClient) -> dict[str, Any]:
    response = client.get(f"/agent/v1/deployments/{_DEPLOYMENT_ID}/observed", headers=_AUTH)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_the_agent_records_which_rank_serves_on_the_container(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """A label, for the same reasons the revision and endpoint are labels.

    It survives an agent restart, and it is read back from the same object whose
    liveness is being reported, so the two facts cannot drift apart. The agent
    cannot re-derive it — deriving needs the deployment's node order, which
    lives on the coordinator — and a second derivation would be a second thing
    that can disagree with the first.
    """
    client = _client(engine, tmp_path)
    _deploy(client, _WORKER)

    labels = engine.containers[_CONTAINER].labels
    assert labels["tensorstead.serves_inference"] == "false"


def test_a_headless_rank_is_not_probed_and_says_why(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """Nothing is listening, and nothing was supposed to be.

    The endpoint fields stay ``None``: not ``False``, which would be a claim
    about a listener nobody looked for, and not ``True``, which would be a claim
    about a service that does not exist on this node.
    """
    client = _client(engine, tmp_path)
    _deploy(client, _WORKER)
    observed = _observe(client)

    assert observed["status"] == "running"
    assert observed["serves_inference"] is False
    assert observed["inference_ready"] is None
    assert observed["endpoint_reachable"] is None
    assert "serves no API" in observed["detail"], (
        "an operator seeing a blank node needs to be told it is blank on purpose"
    )


def test_the_head_is_still_probed(engine: FakeContainerEngine, tmp_path: Path) -> None:
    """The exemption must not quietly become "stop checking multi-node"."""
    client = _client(engine, tmp_path)
    _deploy(client, _HEAD)
    observed = _observe(client)

    assert observed["serves_inference"] is True
    assert "serves no API" not in (observed["detail"] or "")


def test_a_single_node_deployment_is_unaffected(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """Every deployment that worked before this must behave exactly as before."""
    client = _client(engine, tmp_path)
    _deploy(client, None)
    observed = _observe(client)

    assert observed["serves_inference"] is True


def test_a_headless_rank_is_not_reported_as_unfinished_work(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """Reconcile must not find permanent work to do on a healthy group.

    ``inference_not_ready`` on every multi-node deployment, forever, would train
    an operator to ignore the one signal that means something.
    """
    client = _client(engine, tmp_path)
    _deploy(client, _WORKER)

    response = client.post(
        f"/agent/v1/deployments/{_DEPLOYMENT_ID}:reconcile", json={}, headers=_AUTH
    )
    assert response.status_code == 200, response.text
    kinds = [item.get("kind") for item in response.json().get("remaining", [])]
    assert "inference_not_ready" not in kinds


# --------------------------------------------------------------------------
# The verdict the coordinator reaches
# --------------------------------------------------------------------------


def _nodes(*nodes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {f"node-{i}": node for i, node in enumerate(nodes)}


def test_a_working_group_reports_that_it_serves() -> None:
    """The whole point, in one assertion.

    Head serving, worker headless: this deployment is working, and before this
    change the product said ``inference_ready: false`` about it.
    """
    per_node = _nodes(
        {"inference_ready": True, "serves_inference": True},
        {"inference_ready": None, "serves_inference": False},
    )
    assert ObservationService._worst_case(per_node, "inference_ready") is True


def test_a_failing_head_still_vetoes() -> None:
    """The exemption removes a false negative; it must not remove a true one."""
    per_node = _nodes(
        {"inference_ready": False, "serves_inference": True},
        {"inference_ready": None, "serves_inference": False},
    )
    assert ObservationService._worst_case(per_node, "inference_ready") is False


def test_a_group_where_nobody_serves_is_unknown_not_ready() -> None:
    """Excluding every node must not turn "no evidence" into a clean answer.

    The degenerate case of an exemption is where it goes wrong: filter away
    every contributor and an ``all()`` over the empty set is vacuously true.
    That would report a deployment with no serving rank at all as serving.
    """
    per_node = _nodes(
        {"inference_ready": None, "serves_inference": False},
        {"inference_ready": None, "serves_inference": False},
    )
    assert ObservationService._worst_case(per_node, "inference_ready") is None


def test_an_unknown_head_is_still_unknown() -> None:
    """A serving rank that could not be established collapses the answer."""
    per_node = _nodes(
        {"inference_ready": None, "serves_inference": True},
        {"inference_ready": None, "serves_inference": False},
    )
    assert ObservationService._worst_case(per_node, "inference_ready") is None


def test_an_agent_too_old_to_say_still_vetoes() -> None:
    """Silence means "I probed", because that is what an older agent did.

    An agent below contract 1.12 genuinely probed its headless rank and
    genuinely got nothing, so its ``False`` is a real observation of a real
    probe. Assuming otherwise would have the coordinator invent a fact the node
    never reported — which is the habit this whole area exists to break. The
    repair is upgrading the node.
    """
    per_node = _nodes(
        {"inference_ready": True, "serves_inference": True},
        {"inference_ready": False},  # no serves_inference key at all
    )
    assert ObservationService._worst_case(per_node, "inference_ready") is False


def test_reachability_is_exempted_on_the_same_terms() -> None:
    """Both endpoint facts, not just the interesting one.

    ``endpoint_reachable`` aggregates through the same function, and a headless
    rank has no listener to be reachable either. Fixing one and not the other
    would leave a working group reporting ``endpoint_reachable: false`` beside
    ``inference_ready: true``, which is a contradiction on its face.
    """
    per_node = _nodes(
        {"endpoint_reachable": True, "serves_inference": True},
        {"endpoint_reachable": None, "serves_inference": False},
    )
    assert ObservationService._worst_case(per_node, "endpoint_reachable") is True


def test_a_stopped_headless_rank_still_reports_that_it_never_served(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """The fact must survive the container it describes.

    ``serves_inference`` was computed *after* the not-running early return, so it
    was truthful while a worker ran and reverted to the model default of ``true``
    the instant one stopped. That is precisely inverted: an operator reads this
    field when they are looking at a stopped or failed rank, trying to work out
    whether its silence means anything. Telling them a headless worker was
    supposed to be serving is the worst moment to be wrong.

    The label outlives the process — which is why the label is where this lives.
    """
    client = _client(engine, tmp_path)
    _deploy(client, _WORKER)
    assert engine.containers[_CONTAINER].labels["tensorstead.serves_inference"] == "false"

    engine.stop_container(_CONTAINER, exit_code=1)
    observed = _observe(client)

    assert observed["status"] == "not_running"
    assert observed["serves_inference"] is False, (
        "a stopped headless rank claimed it was meant to serve, so its dead endpoint "
        "reads as a fault rather than as the absence it always was"
    )


def test_a_stopped_head_still_reports_that_it_served(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """The other side of it: a dead head is a real fault and must still look like one."""
    client = _client(engine, tmp_path)
    _deploy(client, _HEAD)
    engine.stop_container(_CONTAINER, exit_code=1)

    observed = _observe(client)
    assert observed["status"] == "not_running"
    assert observed["serves_inference"] is True


def test_a_container_predating_the_label_is_assumed_to_have_served(
    engine: FakeContainerEngine, tmp_path: Path
) -> None:
    """Legacy containers keep the old meaning, stopped or running.

    Every deployment predating multi-node served from every node it ran on, so
    an absent label means ``true``. Asserted on the stopped path too, because
    that is the branch that was silently returning the model default and would
    have passed a running-only test either way.
    """
    client = _client(engine, tmp_path)
    _deploy(client, None)
    engine.containers[_CONTAINER].labels.pop("tensorstead.serves_inference", None)

    assert _observe(client)["serves_inference"] is True
    engine.stop_container(_CONTAINER, exit_code=0)
    assert _observe(client)["serves_inference"] is True
