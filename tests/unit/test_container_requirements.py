"""A runtime's container-level needs, which are not launch flags.

Some things a runtime needs cannot be said in argv. vLLM's workers talk through
``/dev/shm``, and Docker gives a container 64MB of it by default — enough for a
single worker and nowhere near enough for a tensor-parallel group. The failure
names neither shared memory nor Docker: a worker dies, or initialisation hangs.

This was the last structural reason a new model might require *editing Project
X* before it could be served. The earlier spec made every vLLM flag reachable; this makes
the container reachable too.

The knowledge belongs to the adapter. vLLM's need for shared
memory is a fact about vLLM, and putting it in the agent's runtime-agnostic
deployment route is where the next runtime's variant gets bolted on as
``if runtime_type ==`` — a mistake already found twice.
"""

from __future__ import annotations

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.ports.runtime_adapter import NodePosition

pytestmark = pytest.mark.unit

_GIB = 1024 * 1024 * 1024


def _requirements(config: dict) -> object:
    adapter = VLLMAdapter()
    return adapter.container_requirements(adapter.validate_config(config))


def test_vllm_asks_for_more_shared_memory_than_docker_gives_by_default() -> None:
    """The default is 64MB, which is the whole problem."""
    assert _requirements({}).shm_size > 64 * 1024 * 1024  # type: ignore[attr-defined]


def test_shared_memory_scales_with_parallelism() -> None:
    """A single-GPU deployment is not taxed for a capability it does not use.

    Reserving the tensor-parallel amount unconditionally would cost every
    ordinary deployment address space for nothing.
    """
    single = _requirements({}).shm_size  # type: ignore[attr-defined]
    parallel = _requirements({"tensor_parallel_size": 4}).shm_size  # type: ignore[attr-defined]

    assert parallel > single


def test_pipeline_parallelism_counts_too() -> None:
    """Workers are workers however the model was split across them."""
    tp_only = _requirements({"tensor_parallel_size": 2}).shm_size  # type: ignore[attr-defined]
    both = _requirements(  # type: ignore[attr-defined]
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 2}
    ).shm_size

    assert both > tp_only


def test_an_operator_can_override_what_the_adapter_decided() -> None:
    """The adapter's number is a default, not a ceiling.

    Its scaling rule is a guess about hardware this product cannot see, so an
    operator who has measured must be able to win.
    """
    assert _requirements({"host_config": {"shm_size": 32 * _GIB}}).shm_size == 32 * _GIB  # type: ignore[attr-defined]


def test_the_things_a_distributed_group_needs_are_expressible() -> None:
    requirements = _requirements(
        {
            "tensor_parallel_size": 2,
            "host_config": {
                "ipc_mode": "host",
                "extra_ports": {"6379/tcp": 6379},
                "ulimits": {"memlock": [-1, -1]},
                "environment": {"NCCL_DEBUG": "INFO"},
            },
        }
    )
    assert requirements.ipc_mode == "host"  # type: ignore[attr-defined]
    assert requirements.extra_ports == {"6379/tcp": 6379}  # type: ignore[attr-defined]
    assert requirements.ulimits == {"memlock": (-1, -1)}  # type: ignore[attr-defined]
    # Subset, not equality: the adapter also contributes cache-root variables
    # of its own, and an exact match would make every future
    # adapter-declared variable look like a regression in this test.
    assert requirements.environment["NCCL_DEBUG"] == "INFO"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "escape",
    ["privileged", "network_mode", "binds", "volumes", "cap_add", "pid_mode", "devices"],
)
def test_host_config_is_not_a_general_docker_passthrough(escape: str) -> None:
    """An escape hatch granting these is not configuration.

    ``privileged``, host networking, arbitrary bind mounts, and added
    capabilities each let a container do something the product cannot describe
    in its own records — at which point the deployment definition stops being a
    true account of what is running, which is the property the whole product
    exists to maintain.
    """
    with pytest.raises(ValueError) as exc:
        _requirements({"host_config": {escape: True}})
    assert escape in str(exc.value)


def test_host_config_environment_cannot_set_the_managed_credential() -> None:
    """A managed credential must not be reachable through the passthrough.

    ``host_config.environment`` is an operator-declared, unvalidated
    passthrough. Without this, a deployment could name ``VLLM_API_KEY`` there
    and silently override the credential the agent route resolves and sets --
    the merge this value eventually reaches used to apply it last, so the
    free-form entry won.
    """
    with pytest.raises(ValueError, match="VLLM_API_KEY"):
        _requirements({"host_config": {"environment": {"VLLM_API_KEY": "attacker-chosen"}}})


def test_a_nonsense_shared_memory_size_is_refused() -> None:
    for bad in (0, -1, "8g"):
        with pytest.raises(ValueError):
            _requirements({"host_config": {"shm_size": bad}})


def test_an_unused_host_config_leaves_no_trace_in_the_record() -> None:
    """A deployment that needed nothing should not grow a line saying so."""
    assert "host_config" not in VLLMAdapter().validate_config({"max_model_len": 4096})


def test_llamacpp_declares_no_special_needs_and_says_so() -> None:
    """A real counter-example, not an adapter that forgot to implement this.

    llama.cpp is single-process and shares no memory between workers, so the
    vLLM requirement genuinely does not apply. The seam is only worth having if
    a second implementation can honestly answer differently.
    """
    requirements = LlamaCppAdapter().container_requirements({})

    assert requirements.shm_size is None
    assert requirements.ipc_mode is None
    assert requirements.extra_ports == {}


# ------------------------------------------------ the durable cache


def test_vllm_asks_for_somewhere_durable_to_put_what_it_compiles() -> None:
    """A path inside the container, never a host path.

    The adapter states what must be true — "durable storage visible here" — and
    the agent decides where that lives. That is the difference that lets this be
    granted at all while `host_config` still refuses `volumes` and `binds`: an
    operator naming a host path describes a mount the product cannot record; a
    product-chosen location is as accountable as the model store.
    """
    requirements = _requirements({})

    assert requirements.cache_at  # type: ignore[attr-defined]
    assert requirements.cache_at.startswith("/")  # type: ignore[attr-defined]


def test_the_cache_variables_point_inside_the_durable_path() -> None:
    """Declaring the mount without pointing the runtime at it would do nothing.

    vLLM caches Triton kernels and torch.compile artefacts under home-relative
    directories. A durable mount the runtime never writes to is a directory that
    stays empty while the runtime recompiles every start — the silent no-op that
    looks like success.
    """
    env = _requirements({}).environment  # type: ignore[attr-defined]
    cache_at = _requirements({}).cache_at  # type: ignore[attr-defined]

    assert env["VLLM_CACHE_ROOT"].startswith(cache_at)
    assert env["TRITON_CACHE_DIR"].startswith(cache_at)
    # HOME too, because a library consulting neither variable still lands
    # somewhere, and several do.
    assert env["HOME"] == cache_at


def test_an_operator_can_still_redirect_the_cache_variables() -> None:
    """Adapter defaults, not adapter mandates."""
    env = _requirements(  # type: ignore[attr-defined]
        {"host_config": {"environment": {"VLLM_CACHE_ROOT": "/somewhere/else"}}}
    ).environment

    assert env["VLLM_CACHE_ROOT"] == "/somewhere/else"


def test_llamacpp_asks_for_no_cache() -> None:
    """It does not compile kernels at runtime, so it needs nowhere to put them."""
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter

    assert LlamaCppAdapter().container_requirements({}).cache_at is None


# ---------------------------------------- distributed position


def _position(index: int, count: int = 2) -> NodePosition:
    peers = [f"10.0.0.{11 + i}" for i in range(count)]
    return NodePosition(
        node_index=index,
        node_count=count,
        self_address=peers[index],
        peer_addresses=peers,
    )


def test_rank_is_derived_from_declared_order_not_configured() -> None:
    """The product states a position; the adapter turns it into a rank.

    Nothing about a rank appears in `runtime_config`. That is the point: a
    `per_node` override map would make one deployment into a set of
    related-but-different ones, and divergence, export and comparability all
    compare *one* declared shape against reality.
    """
    adapter = VLLMAdapter()
    config = adapter.validate_config({})

    head = adapter.build_launch_args(config, model_path="/m", position=_position(0))
    worker = adapter.build_launch_args(config, model_path="/m", position=_position(1))

    assert head[head.index("--node-rank") + 1] == "0"
    assert worker[worker.index("--node-rank") + 1] == "1"
    # Both rendezvous at the first declared node, and both are told the size.
    for args in (head, worker):
        assert args[args.index("--master-addr") + 1] == "10.0.0.11"
        assert args[args.index("--nnodes") + 1] == "2"


def test_a_single_node_deployment_is_untouched() -> None:
    """Every deployment that exists today produces exactly the argv it did."""
    adapter = VLLMAdapter()
    config = adapter.validate_config({})

    assert adapter.build_launch_args(config, model_path="/m") == ["--model", "/m"]
    assert adapter.build_launch_args(config, model_path="/m", position=_position(0, count=1)) == [
        "--model",
        "/m",
    ]


def test_host_access_is_granted_only_when_the_deployment_spans_nodes() -> None:
    """A single-node deployment must not get the fabric.

    Host networking and RDMA devices are what a distributed group needs and
    what an ordinary deployment must never be handed because a distributed one
    needed it.
    """
    adapter = VLLMAdapter()
    config = adapter.validate_config({})

    alone = adapter.container_requirements(config)
    grouped = adapter.container_requirements(config, _position(0))

    assert alone.network_mode is None and alone.devices == []
    assert grouped.network_mode == "host"
    assert "/dev/infiniband" in grouped.devices
    assert grouped.environment["VLLM_HOST_IP"] == "10.0.0.11"


@pytest.mark.parametrize("forbidden", ["network_mode", "devices"])
def test_an_operator_still_cannot_ask_for_host_access(forbidden: str) -> None:
    """The asymmetry that keeps the boundary.

    The product may grant what its own reviewed code declares. An operator may
    not name arbitrary host access in configuration, because a deployment
    record that cannot describe what its container can reach has stopped being
    a true account of what is running.
    """
    with pytest.raises(ValueError) as exc:
        _requirements({"host_config": {forbidden: "host"}})
    assert forbidden in str(exc.value)
