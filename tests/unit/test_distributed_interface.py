"""A group's collectives ride a declared interface.

The first live TP=2 start handed vLLM `VLLM_HOST_IP=spark-alpha.internal` — a
hostname whose only records are public IPv6 — and set no interface hint at all.
Gloo could not resolve a usable IPv4 peer and fell back to loopback:

    Gloo connectFullMesh failed ... Connection refused, remote=[127.0.0.1]

The crash was the *lucky* outcome. The estate's node names route over the
general LAN while two 200 Gb/s RoCE interfaces sit idle, so the same
misconfiguration that happened to form a group would have run tensor-parallel
traffic across the house network and reported success.

Three facts, three owners:

- **which** interface — declared on the deployment, because choosing between a
  machine's interfaces is infrastructure policy;
- **the address on it** — resolved by the owning agent, because it is a property
  of that host at that moment and ``platform_facts`` is captured once at
  registration and never written back;
- **the RoCE device behind it** — likewise the agent's, read from sysfs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent import host_network
from tensorstead.ports.runtime_adapter import NodePosition

pytestmark = pytest.mark.unit


def _position(count: int = 2) -> NodePosition:
    return NodePosition(
        node_index=0,
        node_count=count,
        self_address="spark-alpha.internal",
        peer_addresses=["spark-alpha.internal", "spark-beta.internal"],
    )


def _environment(config: dict[str, Any], position: NodePosition | None) -> dict[str, str]:
    return dict(VLLMAdapter().container_requirements(config, position).environment)


def test_a_declared_interface_steers_the_transport() -> None:
    """The defect: nothing told the runtime which wire to use."""
    env = _environment(
        {"tensor_parallel_size": 2, "distributed_interface": "enp1s0f0np0"}, _position()
    )

    assert env["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0"
    assert env["NCCL_SOCKET_IFNAME"] == "enp1s0f0np0"


def test_no_interface_is_declared_by_default() -> None:
    """The product must not pick between a machine's interfaces."""
    from tensorstead.adapters.runtimes.vllm import VLLMConfig

    assert VLLMConfig().distributed_interface is None


def test_a_single_node_deployment_is_untouched() -> None:
    """One node forms no group; its environment must not change."""
    env = _environment({"tensor_parallel_size": 1, "distributed_interface": "enp1s0f0np0"}, None)

    assert "GLOO_SOCKET_IFNAME" not in env
    assert "NCCL_SOCKET_IFNAME" not in env


def test_a_multi_node_deployment_without_an_interface_still_starts() -> None:
    """Absent is a real answer: the runtime chooses, as it did before.

    Refusing here would break every existing multi-node record, and the product
    does not know that a given estate has a better interface to offer.
    """
    env = _environment({"tensor_parallel_size": 2}, _position())

    assert "GLOO_SOCKET_IFNAME" not in env
    assert env["VLLM_HOST_IP"] == "spark-alpha.internal"


@pytest.mark.parametrize("variable", ["GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"])
def test_free_form_environment_cannot_contradict_the_declaration(variable: str) -> None:
    """A record must not claim one transport while the container gets another.

    This file previously contained the opposite test, ``test_an_operator_
    override_wins``, asserting that ``host_config.environment`` beat the
    declaration. That was reasoned from a good general rule -- an operator's
    explicit setting beats a product default -- which does not apply here,
    because these are not two settings but two spellings of one fact.
    """
    with pytest.raises(ValueError, match="may not set"):
        VLLMAdapter().validate_config(
            {
                "tensor_parallel_size": 2,
                "distributed_interface": "enp1s0f0np0",
                "host_config": {"environment": {variable: "enP2p1s0f0np0"}},
            }
        )


def test_even_an_agreeing_value_is_refused() -> None:
    """Two sources that match today are two sources that can diverge tomorrow.

    Equality is also uncheckable for ``NCCL_IB_HCA``, which the agent resolves
    from the host after this validation has already run.
    """
    with pytest.raises(ValueError, match="may not set"):
        VLLMAdapter().validate_config(
            {
                "tensor_parallel_size": 2,
                "distributed_interface": "enp1s0f0np0",
                "host_config": {"environment": {"GLOO_SOCKET_IFNAME": "enp1s0f0np0"}},
            }
        )


def test_the_refusal_says_what_to_do_instead() -> None:
    """A refusal an operator cannot act on becomes a habit of ignoring refusals."""
    with pytest.raises(ValueError) as caught:
        VLLMAdapter().validate_config(
            {
                "distributed_interface": "enp1s0f0np0",
                "host_config": {"environment": {"NCCL_IB_HCA": "rocep1s0f0"}},
            }
        )

    assert "distributed_interface" in str(caught.value)


def test_unrelated_environment_is_still_forwarded() -> None:
    """Only the transport variables are refused; the door stays open otherwise."""
    env = _environment(
        {
            "tensor_parallel_size": 2,
            "distributed_interface": "enp1s0f0np0",
            "host_config": {"environment": {"VLLM_LOGGING_LEVEL": "DEBUG"}},
        },
        _position(),
    )

    assert env["VLLM_LOGGING_LEVEL"] == "DEBUG"
    assert env["GLOO_SOCKET_IFNAME"] == "enp1s0f0np0"


def test_environment_is_unconstrained_without_a_declaration() -> None:
    """No declaration, no contradiction: the operator is the only source."""
    env = _environment(
        {
            "tensor_parallel_size": 2,
            "host_config": {"environment": {"GLOO_SOCKET_IFNAME": "enP2p1s0f0np0"}},
        },
        _position(),
    )

    assert env["GLOO_SOCKET_IFNAME"] == "enP2p1s0f0np0"


class TestHostResolution:
    """The two facts only the owning agent can establish."""

    def test_the_rdma_device_is_read_from_sysfs(self, tmp_path: Path) -> None:
        """``enp1s0f0np0`` -> ``rocep1s0f0``, the mapping ``rdma link`` prints."""
        device = tmp_path / "enp1s0f0np0" / "device" / "infiniband" / "rocep1s0f0"
        device.mkdir(parents=True)

        assert host_network.rdma_device("enp1s0f0np0", sys_class_net=tmp_path) == "rocep1s0f0"

    def test_an_ethernet_interface_reports_no_device(self, tmp_path: Path) -> None:
        """A true answer, not a failure: it has no RDMA device."""
        (tmp_path / "eth0").mkdir(parents=True)

        assert host_network.rdma_device("eth0", sys_class_net=tmp_path) is None

    def test_an_absent_interface_reports_none(self, tmp_path: Path) -> None:
        assert host_network.rdma_device("nosuch", sys_class_net=tmp_path) is None

    def test_an_absent_interface_has_no_address(self) -> None:
        """``None`` means "could not establish", never a guessed address."""
        assert host_network.interface_address("tensorstead-nosuch-if") is None

    def test_loopback_resolves_to_its_own_address(self) -> None:
        """A positive case that needs no special hardware, proving the lookup works.

        Skipped where the platform has no ``lo`` -- the agent runs on Linux and
        this module is never exercised on the controller in production.
        """
        address = host_network.interface_address("lo")
        if address is None:
            pytest.skip("no loopback interface addressable by ioctl on this platform")
        assert address == "127.0.0.1"


def test_the_address_is_parsed_out_of_the_ioctl_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parsing, which the loopback case above skips on this platform.

    ``SIOCGIFADDR`` returns a packed ``sockaddr_in``; the address is four bytes
    at offset 20. An off-by-anything here yields a *plausible* wrong address
    rather than an error, so it is worth a test that runs everywhere rather than
    one that skips on the machine the code is written on.
    """
    import socket as socket_module
    import struct

    packed = bytearray(32)
    packed[16:20] = struct.pack("H H", socket_module.AF_INET, 0)[:4]
    packed[20:24] = bytes([10, 100, 184, 1])

    monkeypatch.setattr(host_network.fcntl, "ioctl", lambda *_a, **_k: bytes(packed))

    assert host_network.interface_address("enp1s0f0np0") == "10.100.184.1"


def test_an_interface_without_an_address_fails_the_start(tmp_path: Path) -> None:
    """Refuse rather than let the runtime choose a wire nobody declared.

    A declared interface that cannot supply an address means the deployment's
    record and the group's actual transport are about to disagree -- which on
    this estate means the general LAN, silently.
    """
    from fastapi import HTTPException

    from tensorstead.agent.routes.deployments import _resolve_host_network
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    requirements = ContainerRequirements(environment={"VLLM_HOST_IP": "spark-alpha.internal"})

    with pytest.raises(HTTPException) as caught:
        _resolve_host_network(requirements, {"distributed_interface": "tensorstead-nosuch-if"})

    assert caught.value.detail["code"] == "distributed_interface_unusable"
    assert "tensorstead-nosuch-if" in caught.value.detail["message"]


def test_the_resolved_address_replaces_the_management_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: a peer must reach this rank on the RoCE link."""
    from tensorstead.agent.routes import deployments as route
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    monkeypatch.setattr(route.host_network, "interface_address", lambda _i: "10.100.184.1")
    monkeypatch.setattr(route.host_network, "rdma_device", lambda _i: "rocep1s0f0")
    monkeypatch.setattr(route.host_network, "rdma_gid_index", lambda _d, _a: 3)
    requirements = ContainerRequirements(environment={"VLLM_HOST_IP": "spark-alpha.internal"})

    route._resolve_host_network(requirements, {"distributed_interface": "enp1s0f0np0"})

    assert requirements.environment["VLLM_HOST_IP"] == "10.100.184.1"
    assert requirements.environment["NCCL_IB_HCA"] == "rocep1s0f0"
    assert requirements.environment["NCCL_IB_GID_INDEX"] == "3"


def test_an_ethernet_interface_sets_no_rdma_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain Ethernet is a valid choice; it just has no device to name."""
    from tensorstead.agent.routes import deployments as route
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    monkeypatch.setattr(route.host_network, "interface_address", lambda _i: "198.51.100.20")
    monkeypatch.setattr(route.host_network, "rdma_device", lambda _i: None)
    requirements = ContainerRequirements(environment={})

    route._resolve_host_network(requirements, {"distributed_interface": "enP7s7"})

    assert requirements.environment["VLLM_HOST_IP"] == "198.51.100.20"
    assert "NCCL_IB_HCA" not in requirements.environment


# ------------------------------------------------ using the device we granted
#
# Declaring /dev/infiniband alone was incoherent: RDMA registers memory with the
# NIC, which pins it, and pinning needs IPC_LOCK and a raised memlock. The
# container was handed the device and withheld the ability to use it, and the
# first group to reach its own collectives died there.


def _requirements(config: dict[str, Any], position: NodePosition | None) -> Any:
    return VLLMAdapter().container_requirements(config, position)


def test_a_distributed_group_may_pin_memory() -> None:
    """The grant that makes the declared RDMA device usable."""
    requirements = _requirements({"tensor_parallel_size": 2}, _position())

    assert "IPC_LOCK" in requirements.capabilities
    assert requirements.ulimits["memlock"] == (-1, -1)


def test_the_rdma_device_and_its_permissions_travel_together() -> None:
    """Either both or neither: granting one without the other is the defect."""
    requirements = _requirements({"tensor_parallel_size": 2}, _position())

    assert "/dev/infiniband" in requirements.devices
    assert requirements.capabilities and requirements.ulimits


def test_a_single_node_deployment_gets_neither() -> None:
    """One node forms no group and needs no RDMA grant."""
    requirements = _requirements({"tensor_parallel_size": 1}, None)

    assert requirements.capabilities == []
    assert "memlock" not in requirements.ulimits
    assert requirements.devices == []


def test_an_operator_ulimit_still_wins() -> None:
    """A limit tuned for an estate must not be overruled by our default."""
    requirements = _requirements(
        {"tensor_parallel_size": 2, "host_config": {"ulimits": {"memlock": [8, 16]}}},
        _position(),
    )

    assert requirements.ulimits["memlock"] == (8, 16)


def test_capabilities_are_not_operator_settable() -> None:
    """Adapters declare what a container may do; configuration may not.

    The same asymmetry as ``network_mode`` and ``devices``. An operator who
    could name capabilities could grant the container anything, and the record
    would no longer describe what is running.
    """
    with pytest.raises(ValueError, match="host_config does not accept"):
        VLLMAdapter().validate_config(
            {"tensor_parallel_size": 2, "host_config": {"capabilities": ["SYS_ADMIN"]}}
        )


def test_the_product_never_reaches_for_privileged() -> None:
    """Privileged would also have fixed this, and grants everything besides.

    Pinned as a guardrail rather than left to judgement: it is the shortcut that
    makes an RDMA problem disappear, and taking it would mean a deployment
    record that cannot say what its container may do.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "tensorstead"
    for path in sorted(src.rglob("*.py")):
        text = path.read_text()
        assert "privileged=True" not in text and '"privileged"' not in text, (
            f"{path.relative_to(src)} reaches for privileged; declare the named "
            f"capability the runtime actually needs instead"
        )


# ------------------------------------------- who chooses the executor backend
#
# The old code hardcoded `--distributed-executor-backend mp` for every multi-node
# group. That was mine, argued as "naming a backend is legitimate" and then
# quietly choosing one without testing it. On hardware the follower rank died
# inside vLLM's KV-cache setup with "collective_rpc should not be called on
# follower node".


def _argv(config: dict[str, Any], position: NodePosition | None) -> list[str]:
    return VLLMAdapter().build_launch_args(config, model_path="/models/m", position=position)


def test_no_backend_is_imposed_on_a_group() -> None:
    """The defect: every multi-node group got `mp` whether it suited or not."""
    argv = _argv({"tensor_parallel_size": 2}, _position())

    assert "--distributed-executor-backend" not in argv, (
        "the adapter is choosing an executor backend the operator did not declare"
    )


def test_a_declared_backend_is_emitted() -> None:
    """Declared, so an operator can try another without a code change."""
    argv = _argv({"tensor_parallel_size": 2, "distributed_executor_backend": "ray"}, _position())

    index = argv.index("--distributed-executor-backend")
    assert argv[index + 1] == "ray"


def test_the_backend_is_not_checked_against_a_list() -> None:
    """Which backends a build supports is a fact about that build, not about us.

    An allowlist here would go stale on vLLM's release schedule rather than
    ours -- the closed-allowlist failure the design exists to avoid.
    """
    argv = _argv(
        {"tensor_parallel_size": 2, "distributed_executor_backend": "some_future_backend"},
        _position(),
    )

    assert "some_future_backend" in argv


def test_a_single_node_deployment_may_still_declare_one() -> None:
    """`mp` is a legitimate single-node choice; the field is not group-only."""
    argv = _argv({"distributed_executor_backend": "mp"}, None)

    assert "--distributed-executor-backend" in argv
    assert "--nnodes" not in argv


def test_the_rendezvous_facts_are_unaffected() -> None:
    """Removing the backend must not disturb what position legitimately supplies."""
    argv = _argv({"tensor_parallel_size": 2}, _position())

    for flag in ("--nnodes", "--node-rank", "--master-addr", "--master-port"):
        assert flag in argv, f"{flag} was lost"


# ------------------------------------------------- which GID, resolved per rank
#
# The recipe reads NCCL_IB_GID_INDEX from each node's own sysfs. Tensorstead
# carries one shared runtime_config, so a declared literal would send one rank's
# index to both -- review caught that in the first draft. It
# belongs with the other rank-local facts the agent resolves.


def _fake_infiniband(root: Path, device: str, entries: list[tuple[int, str, str]]) -> Path:
    """Build a sysfs tree of ``(index, gid, type)`` for one RoCE port."""
    port = root / device / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "types").mkdir(parents=True)
    for index, gid, kind in entries:
        (port / "gids" / str(index)).write_text(gid + "\n")
        (port / "gid_attrs" / "types" / str(index)).write_text(kind + "\n")
    return root


# The shape a ConnectX port actually presents: RoCE v1 and v2 for each address,
# plus a link-local IPv6 GID that must not be chosen.
_REAL_SHAPE = [
    (0, "fe80:0000:0000:0000:4ebb:47ff:fe00:471c", "IB/RoCE v1"),
    (1, "fe80:0000:0000:0000:4ebb:47ff:fe00:471c", "RoCE v2"),
    (2, "0000:0000:0000:0000:0000:ffff:0a64:b801", "IB/RoCE v1"),
    (3, "0000:0000:0000:0000:0000:ffff:0a64:b801", "RoCE v2"),
]


def test_the_gid_index_matches_v2_and_this_ranks_address(tmp_path: Path) -> None:
    """Index 3: RoCE v2 *and* encoding 10.100.184.1. Not 1, not 2."""
    _fake_infiniband(tmp_path, "rocep1s0f0", _REAL_SHAPE)

    index = host_network.rdma_gid_index("rocep1s0f0", "10.100.184.1", sys_class_infiniband=tmp_path)

    assert index == 3


def test_the_other_rank_resolves_its_own_index(tmp_path: Path) -> None:
    """The reason this cannot be declared: each rank has a different answer."""
    _fake_infiniband(
        tmp_path,
        "rocep1s0f0",
        [
            (0, "0000:0000:0000:0000:0000:ffff:0a64:b801", "RoCE v2"),
            (1, "0000:0000:0000:0000:0000:ffff:0a64:b802", "RoCE v2"),
        ],
    )

    assert (
        host_network.rdma_gid_index("rocep1s0f0", "10.100.184.2", sys_class_infiniband=tmp_path)
        == 1
    )


def test_a_v1_only_match_is_refused(tmp_path: Path) -> None:
    """RoCE v1 carries the right address and is still the wrong GID."""
    _fake_infiniband(
        tmp_path, "rocep1s0f0", [(0, "0000:0000:0000:0000:0000:ffff:0a64:b801", "IB/RoCE v1")]
    )

    assert (
        host_network.rdma_gid_index("rocep1s0f0", "10.100.184.1", sys_class_infiniband=tmp_path)
        is None
    )


def test_an_unknown_device_reports_none(tmp_path: Path) -> None:
    assert (
        host_network.rdma_gid_index("nosuch", "10.100.184.1", sys_class_infiniband=tmp_path) is None
    )


def test_an_unresolvable_gid_fails_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse rather than let the library choose unaided.

    An RDMA device is present, so the collectives *will* use it. Leaving the
    index unset is the silent misconfiguration this whole thread has been about.
    """
    from fastapi import HTTPException

    from tensorstead.agent.routes import deployments as route
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    monkeypatch.setattr(route.host_network, "interface_address", lambda _i: "10.100.184.1")
    monkeypatch.setattr(route.host_network, "rdma_device", lambda _i: "rocep1s0f0")
    monkeypatch.setattr(route.host_network, "rdma_gid_index", lambda _d, _a: None)

    with pytest.raises(HTTPException) as caught:
        route._resolve_host_network(
            ContainerRequirements(environment={}), {"distributed_interface": "enp1s0f0np0"}
        )

    assert caught.value.detail["code"] == "rdma_gid_unresolved"
    assert "rocep1s0f0" in caught.value.detail["message"]


def test_plain_ethernet_needs_no_gid(monkeypatch: pytest.MonkeyPatch) -> None:
    """No RDMA device means no GID to resolve, and that is not a failure."""
    from tensorstead.agent.routes import deployments as route
    from tensorstead.ports.runtime_adapter import ContainerRequirements

    monkeypatch.setattr(route.host_network, "interface_address", lambda _i: "198.51.100.20")
    monkeypatch.setattr(route.host_network, "rdma_device", lambda _i: None)
    requirements = ContainerRequirements(environment={})

    route._resolve_host_network(requirements, {"distributed_interface": "enP7s7"})

    assert requirements.environment["VLLM_HOST_IP"] == "198.51.100.20"
    assert "NCCL_IB_GID_INDEX" not in requirements.environment


# ------------------------------------------------------- the head serves alone
#
# Read from the working recipe's compose command, which ends
# `${HEADLESS:+--headless}` and sets HEADLESS on the worker only. Its absence is
# what failed every distributed start: vLLM's MultiprocExecutor gives a follower
# no rpc_broadcast_mq, and we launched rank 1 as a leader, so its engine core
# reached collective_rpc and the build refused.


def test_only_the_head_runs_a_server() -> None:
    """The defect: every rank was launched as a leader."""
    from tensorstead.adapters.runtimes.vllm import _distributed_args

    head = _distributed_args(_position_at(0))
    worker = _distributed_args(_position_at(1))

    assert "--headless" not in head, "the head must serve; it owns the API"
    assert "--headless" in worker, (
        "a non-head rank was launched as a leader; its engine core will call "
        "collective_rpc on an executor that has no broadcast queue"
    )


def test_every_rank_beyond_the_head_is_headless() -> None:
    """Three nodes, one server. Nothing about this is specific to a pair."""
    from tensorstead.adapters.runtimes.vllm import _distributed_args

    for index in (1, 2):
        assert "--headless" in _distributed_args(_position_at(index, count=3))


def test_a_single_node_deployment_is_never_headless() -> None:
    """One node is the head by definition, and must keep serving."""
    from tensorstead.adapters.runtimes.vllm import _distributed_args

    assert _distributed_args(_position_at(0, count=1)) == []


def _position_at(index: int, count: int = 2) -> NodePosition:
    return NodePosition(
        node_index=index,
        node_count=count,
        self_address=f"10.100.184.{index + 1}",
        peer_addresses=[f"10.100.184.{n + 1}" for n in range(count)],
    )


# ------------------------------- derived topology is not settable by an operator
#
# extra_args accepted headless/nnodes/node-rank/master-addr/master-port, so a
# deployment could claim two nodes and launch nine, name a master nothing
# rendezvouses at, or hand a second --headless to a rank that already had one.
# Found by review before it reached hardware.


@pytest.mark.parametrize(
    "flag", ["headless", "nnodes", "node-rank", "node_rank", "master-addr", "master-port"]
)
def test_derived_topology_cannot_be_supplied(flag: str) -> None:
    """One shared runtime_config reaches every rank; these are rank-local."""
    with pytest.raises(ValueError, match="extra_args may not set"):
        VLLMAdapter().validate_config({"tensor_parallel_size": 2, "extra_args": {flag: 1}})


def test_the_refusal_says_why_each_is_derived() -> None:
    """A refusal that does not say where the value comes from is not actionable."""
    with pytest.raises(ValueError) as caught:
        VLLMAdapter().validate_config({"extra_args": {"master-addr": "wrong.invalid"}})

    assert "derived from the head node" in str(caught.value)


def test_the_rendered_argv_carries_each_flag_once() -> None:
    """The property the refusal protects, asserted end to end."""
    from collections import Counter

    argv = VLLMAdapter().build_launch_args(
        {"tensor_parallel_size": 2, "extra_args": {"block-size": 256}},
        model_path="/models/m",
        position=_position_at(1),
    )

    counts = Counter(argv)
    for flag in ("--headless", "--nnodes", "--node-rank", "--master-addr", "--master-port"):
        assert counts[flag] <= 1, f"{flag} rendered {counts[flag]} times: {argv}"
    assert "--block-size" in argv, "an ordinary passthrough must still be forwarded"
