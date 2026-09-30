"""🚫 GUARDRAIL — no distributed execution of our own.

The product **coordinates**; the runtime **distributes**. This asserts that the
boundary is real rather than intended, by checking the two things that would
erode it:

- **No inter-node communication beyond the two sanctioned paths.** Exactly two
  exist: coordinator→agent management calls, and agent→agent artifact
  replication. Anything else — a node addressing a node for any other reason —
  would be a distribution mechanism growing quietly inside a management tool.
- **No distribution primitives.** No ranks, no world size, no collective ops,
  no process-group bootstrapping, no launcher. Where a runtime distributes, it
  does so with its own machinery configured through its own adapter schema; we
  pass configuration and stay out of it.

Sibling guardrail: ``test_no_data_path.py`` covers inference *traffic*. This one
covers inference *coordination*. Both have to hold — a product could stay off
the data path and still be secretly running the cluster.

These are absence tests. Nothing fails when they are missing, which is exactly
why they are written down (note on guardrail tasks).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_AGENT = _SRC / "agent"

# Distribution primitives. If the product ever implements distribution itself,
# it will be by reaching for one of these.
_DISTRIBUTION_PRIMITIVES = (
    "torch.distributed",
    "init_process_group",
    "world_size",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "local_rank",
    "all_reduce",
    "all_gather",
    "broadcast_object",
    "nccl",
    "NCCL",
    "torchrun",
    "mpirun",
    "deepspeed",
    "ray.init",
)

# The complete set of agent paths the coordinator or a peer may call. A new
# endpoint must be added here consciously, which is the point.
_SANCTIONED_AGENT_PATHS = {
    # coordinator -> agent (management)
    "/agent/v1/info",
    "/agent/v1/resources",
    "/agent/v1/models",
    "/agent/v1/models:acquire",
    "/agent/v1/models/{model_id}",
    "/agent/v1/deployments",
    "/agent/v1/deployments/{deployment_id}",
    "/agent/v1/deployments/{deployment_id}/observed",
    # Runtime observability. A coordinator -> agent management read:
    # the runtime's argv, restart count, and log tail. Not an inter-node
    # channel, and not a data path -- it carries the runtime's own output to an
    # operator on request and the product stores none of it.
    "/agent/v1/deployments/{deployment_id}/runtime",
    "/agent/v1/deployments/{deployment_id}:stop",
    "/agent/v1/deployments/{deployment_id}:reconcile",
    "/agent/v1/endpoint-check",
    "/agent/v1/images:pull",
    # Management calls, both of them: the coordinator telling one node what to
    # remove, and asking one node what it holds. Neither crosses a node
    # boundary between peers.
    "/agent/v1/images:remove",
    "/agent/v1/images",
    # Managed runtime images. Coordinator -> agent management calls,
    # not a new inter-node path: they produce an image on one node, which is
    # then distributed over the existing artifact-replication hop.
    "/agent/v1/images:build",
    # Asks an image what options it accepts. A coordinator ->
    # agent management read, not an inter-node channel: the container runs to
    # completion with no model, no network and no published port, and nothing
    # about it is a serving instance. It lives on the node because the answer is
    # a property of that build on that architecture -- an emulated probe on the
    # x86 controller reported a platform failure unrelated to the image.
    "/agent/v1/images:probe",
    "/agent/v1/images:import",
    "/agent/v1/images:distribute",
    # agent -> agent: image archive content, the second artifact endpoint.
    # Added deliberately -- an image built per node yields a different identifier
    # for the same spec, so it is produced once and copied.
    "/agent/v1/images/{reference}/content",
    # agent -> agent (artifact replication only)
    "/agent/v1/models:replicate",
    "/agent/v1/models/{model_id}/content",
}


def _walk_py(root: Path) -> list[Path]:
    """Every Python file under ``root``, initialisers included.

    Initialisers are not skipped: the coordinator's whole route surface lives
    in one, so skipping them would blind the guardrail to its largest target.
    """
    return sorted(root.rglob("*.py"))


# Names that appear only as *environment variables selecting among a runtime's
# own transports*, with the file that may contain them. Added 2026-08-13, and it
# widens this guardrail, so it is argued rather than asserted.
#
# The distinction this guardrail draws is between implementing distribution and
# configuring a runtime that implements its own. Everything in
# ``_DISTRIBUTION_PRIMITIVES`` is something the product would have to *call* --
# `init_process_group`, `all_reduce`, `torchrun`. `NCCL_IB_HCA` is not called;
# it is declared, for a library this product does not ship, link against, or
# invoke. It is the same act as ``--distributed-executor-backend``, which this
# adapter already emits and whose comment says it exactly: naming the backend
# tells vLLM which of *its own* mechanisms to use rather than supplying one.
#
# The list is already inconsistent on this point, which is worth recording
# rather than quietly exploiting: it forbids ``MASTER_ADDR`` while the same
# adapter emits ``--master-addr``, the identical fact in flag spelling. The
# scan catches spelling there, not capability.
#
# What it must keep catching, and still does: any *import* of a distribution
# library, any call into one, and any of these names appearing outside a
# declared environment mapping. The companion test below pins that.
_TRANSPORT_SELECTION_ENVIRONMENT: dict[str, tuple[str, ...]] = {
    "adapters/runtimes/vllm.py": ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"),
    "agent/routes/deployments.py": ("NCCL_IB_HCA", "NCCL_IB_GID_INDEX"),
    "agent/host_network.py": (),
}


def test_no_distribution_primitives_anywhere() -> None:
    """The product implements no distribution of its own."""
    for path in _walk_py(_SRC):
        text = path.read_text()
        permitted = _TRANSPORT_SELECTION_ENVIRONMENT.get(str(path.relative_to(_SRC)), ())
        for allowed in permitted:
            text = text.replace(allowed, "")
        for primitive in _DISTRIBUTION_PRIMITIVES:
            assert primitive not in text, (
                f"{path.relative_to(_SRC)} references {primitive!r}, a distributed-execution "
                f"primitive. The runtime distributes; we coordinate"
            )


def test_the_product_imports_no_distribution_library() -> None:
    """The half of the rule that the string scan cannot express.

    Blanking a permitted environment-variable name above would also blank it
    inside an import or a call, so the exemption is paired with a check that
    asks the stronger question directly: does any product module *depend* on a
    distribution library at all.
    """
    import ast

    forbidden = {"torch", "deepspeed", "ray", "mpi4py", "horovod"}
    for path in _walk_py(_SRC):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module.split(".")[0] not in forbidden, (
                    f"{path.relative_to(_SRC)} imports {module!r}. The product configures a "
                    f"runtime that distributes; it never links against one"
                )


def test_agent_exposes_only_sanctioned_paths() -> None:
    """Every agent route is a management call or artifact replication.

    A route outside this set is either a new management operation that belongs
    in the contract, or a channel between nodes that should not exist.
    """
    from tensorstead.agent.app import build_agent_app

    app = build_agent_app(management_token="test", replication_token="repl")
    published = {path for path in app.openapi()["paths"] if path.startswith("/agent/")}
    unexpected = published - _SANCTIONED_AGENT_PATHS
    assert unexpected == set(), (
        f"agent publishes unsanctioned inter-node routes: {sorted(unexpected)}. "
        f"Only management calls and artifact replication may cross a node boundary"
    )


def test_the_only_agent_to_agent_path_is_artifact_replication() -> None:
    """Node-to-node traffic carries artifacts and nothing else.

    The replication-token-gated surface is the entire agent→agent API. If a
    second such route appears, a node has gained a reason to talk to a node
    that is not "send me those weights".
    """
    from tensorstead.agent.routes import (
        deployments,
        endpoint,
        images,
        models,
        observation,
        replication,
        resources,
    )

    # Read the routers directly rather than ``app.routes``: since Starlette 1.4
    # / FastAPI 0.141, ``include_router`` no longer flattens routes onto the
    # app, so walking the app would see nothing and pass vacuously.
    modules = (deployments, endpoint, images, models, observation, replication, resources)

    replication_gated: set[str] = set()
    for module in modules:
        for route in module.router.routes:
            for dependency in getattr(route, "dependencies", []):
                if getattr(dependency.dependency, "__name__", "") == "require_replication":
                    # Strip the Starlette path converter (``{id:path}``), which a
                    # model id needs because it contains ':' and '/'.
                    replication_gated.add(
                        re.sub(r"\{(\w+):[^}]+\}", r"{\1}", getattr(route, "path", ""))
                    )

    # Both entries are artifact content endpoints, which is the only reason
    # nodes may talk to one another at all. The image one was
    # added consciously: an image built independently on each node
    # yields a different identifier for the same spec, so it must be produced
    # once and copied. A third entry here that is not "send me those bytes"
    # should fail this test and be argued for.
    assert replication_gated == {
        "/agent/v1/models/{model_id}/content",
        "/agent/v1/images/{reference}/content",
    }, (
        f"agent-to-agent surface is {sorted(replication_gated)}; it must be exactly the "
        f"artifact content endpoints"
    )


def test_no_inference_traffic_between_nodes() -> None:
    """No node forwards, proxies, or relays inference to another.

    Distinct from the data-path guardrail: that one says we never *serve*
    inference. This one says we never *route* it between hosts either.
    """
    forbidden = ("proxy_pass", "forward_request", "relay_inference", "inference_proxy")
    for path in _walk_py(_SRC):
        text = path.read_text()
        for token in forbidden:
            assert token not in text, (
                f"{path.relative_to(_SRC)} references {token!r}; inference clients reach "
                f"the runtime's endpoint directly"
            )


def test_multi_node_support_is_delegated_not_implemented() -> None:
    """Multi-node permission is a capability *question*, never our own logic.

    The service layer's only involvement in distribution is asking the adapter
    whether the runtime supports it. Nothing in the service layer configures a
    distributed group.
    """
    service = _SRC / "service"
    for path in _walk_py(service):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "init_process_group",
                "spawn",
                "launch",
            }:
                raise AssertionError(
                    f"{path.relative_to(_SRC)} calls {node.attr!r}; the product does not "
                    f"launch or coordinate distributed processes"
                )

    # The capability check exists and is a read of the adapter's declaration.
    deployments = (service / "deployments.py").read_text()
    assert "supports_distributed" in deployments, (
        "the multi-node decision must be delegated to the runtime adapter's declared capability"
    )


def test_a_shipped_runtime_actually_refuses_to_distribute() -> None:
    """The seam has a real counter-example, not just a flag we could set.

    Without a runtime that declares ``False``, the rejection path would be
    reachable only from a test double — and a seam only exercised by doubles is
    a seam nobody has actually tried to push on.
    """
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
    from tensorstead.adapters.runtimes.vllm import VLLMAdapter

    declarations = {
        VLLMAdapter.runtime_type: VLLMAdapter.supports_distributed,
        LlamaCppAdapter.runtime_type: LlamaCppAdapter.supports_distributed,
    }
    assert True in declarations.values(), "a runtime that distributes"
    assert False in declarations.values(), "a shipped runtime that does not"
