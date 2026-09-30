"""A node's current agent version is observed, not remembered.

``Node.agent_contract_version`` is captured once, when the node registers, and
never written again. That is the right thing for it to be — it records what the
node said at a known moment — but it was the *only* version the product held,
and it was rendered as though it described the node now.

The gap opens during a rolling upgrade, which is precisely when the question is
asked. An agent upgraded from 1.1 to 1.2 keeps being described as 1.1, so an
operator checking whether the fleet can report inference readiness is told no by
a two-day-old fact.

The reachability call already asked the agent for this and discarded the answer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}


def _register(client: TestClient, name: str = "spark-alpha") -> str:
    response = client.post(
        "/v1/nodes",
        json={"name": name, "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def test_reachability_reports_the_agents_current_version() -> None:
    """The live call answers with what the agent says now."""
    agent = FakeNodeAgent(contract_version="1.2")
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    body = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()

    assert body["status"] == "reachable"
    assert body["contract_version"] == "1.2"


def test_an_upgraded_agent_is_not_described_by_its_registration() -> None:
    """The whole point: the record and the node disagree, and both are reported.

    The stored value is not corrected. It is a true statement about a past
    moment, and overwriting it would destroy that fact rather than add to it.
    The observed value is what an operator acts on.
    """
    agent = FakeNodeAgent(contract_version="1.1")
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    # The node is upgraded in place, as `make deploy` does.
    agent.contract_version = "1.2"

    record = client.get(f"/v1/nodes/{node_id}", headers=_AUTH).json()
    observed = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()

    assert record["agent_contract_version"] == "1.1", (
        "the registration-time record was rewritten; it records what the node "
        "said when it registered and must keep saying so"
    )
    assert observed["contract_version"] == "1.2", (
        "an upgraded agent was still described by its registration snapshot"
    )


def test_an_agent_that_reports_no_version_is_not_given_one() -> None:
    """Absence is reported as absence, never as a guess."""
    agent = FakeNodeAgent(contract_version="1.2")
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    agent.contract_version = ""

    body = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()

    assert body["contract_version"] is None


def test_a_fact_the_agent_learned_to_report_after_registration_is_not_lost() -> None:
    """The same defect as the version above, one field wider.

    `Node.platform_facts` is captured once at registration and never rewritten.
    That is correct as a *record* — it says what the node reported that day —
    and wrong as an input to any judgement about the hardware now. A fact the
    agent only learned to report in a later release cannot appear in it at all.

    Observed on a running estate hours after the quantization
    advisory shipped: both nodes reported `accelerator_compute_capability` and the
    coordinator was reading a three-day-old snapshot that predated the field,
    so the measured half of the note could never have been rendered.

    The registration record is deliberately *not* corrected. Overwriting it
    would destroy a true statement about a past moment.
    """
    agent = FakeNodeAgent()
    agent.platform_facts = {"cpu_arch": "aarch64", "os_family": "linux"}
    app, repository = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    # The agent is upgraded and begins reporting something new about itself.
    agent.platform_facts = {
        "cpu_arch": "aarch64",
        "os_family": "linux",
        "accelerator_compute_capability": "12.1",
    }

    observed = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()
    recorded = repository.get_node(node_id)
    assert recorded is not None

    assert observed["platform_facts"]["accelerator_compute_capability"] == "12.1", (
        "a fact the agent reports now was not observable through the product"
    )
    assert "accelerator_compute_capability" not in (recorded.platform_facts or {}), (
        "the registration snapshot was rewritten; it records what the node said "
        "when it registered and must keep saying so"
    )


def test_an_advisory_reads_the_node_as_it_is_now() -> None:
    """The functional half: the note must use the live fact, not the snapshot.

    Reporting the current capability somewhere is not enough if the code that
    needs it still reads the registration record — which is exactly the state
    the advisory shipped in.
    """
    agent = FakeNodeAgent()
    agent.platform_facts = {"cpu_arch": "aarch64"}
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    _register(client, name="spark-alpha")

    agent.platform_facts = {
        "cpu_arch": "aarch64",
        "accelerator_compute_capability": "12.1",
    }

    notes = app.state.deployments_service.config_advisories(
        "vllm", {"quantization": "modelopt"}, ["spark-alpha"]
    )

    assert notes, "a quantised deployment produced no advisory at all"
    assert "12.1" in notes[0], (
        "the advisory read the registration snapshot rather than the node as it is now"
    )


def test_a_live_reading_does_not_drop_facts_the_record_still_holds() -> None:
    """A partial live answer must not overwrite a fuller record.

    Found on the appliance immediately after deploying the live-facts fix. The
    two sets are not the same shape: both nodes report
    `accelerator_compute_capability` now and neither reports `memory_is_unified`
    any more, which their registration records still hold. A live read that won
    wholesale would silently discard it.

    That is the harder version of this session's recurring defect — a non-empty
    answer looks authoritative while being partial, where an empty one would at
    least have prompted a fallback.
    """
    agent = FakeNodeAgent()
    agent.platform_facts = {"cpu_arch": "aarch64", "memory_is_unified": True}
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    _register(client, name="spark-alpha")

    # The agent is upgraded: it learns a new fact and stops volunteering an old.
    agent.platform_facts = {"cpu_arch": "aarch64", "accelerator_compute_capability": "12.1"}

    facts = app.state.deployments_service._participating_platform_facts(["spark-alpha"])[0]

    assert facts["accelerator_compute_capability"] == "12.1", "the current fact must win"
    assert facts["memory_is_unified"] is True, (
        "a fact the node no longer volunteers was dropped rather than retained "
        "from the registration record"
    )


def test_the_inventory_field_does_not_claim_to_be_current() -> None:
    """The inventory field reported 1.0 while every live probe said 1.6.

    Same two nodes, same moment, two different answers depending on which call
    you trusted. The value was never wrong — it is what each node said when it
    registered, and the product keeps it that way — but the field was called
    ``agent_contract_version``, which reads as the node's version *now*.

    A caller gating capability on it (\"do not send this to a node below 1.4\")
    would refuse a capability the node has had for days, with nothing in the
    response hinting the value was old. Renaming was chosen over refreshing:
    ``node.list`` is the cheap inventory call, and a probe per node would make
    it as slow as the slowest agent and fail outright when one is unreachable.

    Nothing asserted this field before, which is why renaming it broke no test.
    """
    agent = FakeNodeAgent(contract_version="1.0")
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    _register(client)

    agent.contract_version = "1.6"  # the node is upgraded in place

    listed = client.get("/v1/nodes", headers=_AUTH).json()[0]

    assert "agent_contract_version" not in listed, (
        "the inventory still exposes a field whose name claims to be current"
    )
    assert listed["registered_contract_version"] == "1.0", (
        "the registration record must keep saying what the node said that day"
    )


def test_the_inventory_and_the_live_probe_are_both_available_and_disagree() -> None:
    """Both answers exist, and the caller can tell which is which.

    The defect was never that two values differ — a node upgraded after
    registration *should* produce two. It was that only one of them was named
    in a way that said which question it answered.
    """
    agent = FakeNodeAgent(contract_version="1.0")
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    agent.contract_version = "1.6"

    listed = client.get("/v1/nodes", headers=_AUTH).json()[0]
    probed = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()

    assert listed["registered_contract_version"] == "1.0"
    assert probed["contract_version"] == "1.6"


def test_a_node_reports_which_build_it_is_running() -> None:
    """The question `agent release` could never answer.

    `agent_version` reports the package version, which read `0.1.8` for sixty
    consecutive builds. An operator comparing a live appliance against a
    controller had nothing to compare, and could not tell a current node from
    one six deploys behind — which is the record-and-reality gap this product
    exists to close, reproduced in its own release process.
    """
    agent = FakeNodeAgent()
    agent.build = {"build_number": 62, "git_revision": "2178b9f", "dirty": False}
    app, _ = build_test_coordinator(agent=agent)
    client = TestClient(app)
    node_id = _register(client)

    body = client.get(f"/v1/nodes/{node_id}/reachability", headers=_AUTH).json()

    assert body["build"]["build_number"] == 62
    assert body["build"]["git_revision"] == "2178b9f"


def test_an_agent_that_cannot_identify_its_build_says_so() -> None:
    """Unknown, not guessed.

    A wheel not produced by the build script carries no identity. Inventing a
    build number at import time would be the confident fiction this reports in
    order to prevent.
    """
    from tensorstead.version import build_identity

    identity = build_identity()

    # This process runs from a source checkout, so there is nothing to report.
    assert identity["build_number"] is None
    assert identity["git_revision"] is None
