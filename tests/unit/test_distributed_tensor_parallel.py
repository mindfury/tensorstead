"""One authoritative tensor-parallel value, emitted once.

``--tensor-parallel-size`` was assigned from two independent sources:
``_distributed_args`` derived it from ``node_count``, and ``build_launch_args``
appended the declared value. A two-node deployment emitted the flag twice, and a
record declaring 3 across two nodes emitted ``2 ... 3`` — leaving vLLM's
duplicate-option behaviour to decide which declared fact was real.

Deriving it from the node count was only ever accidentally right. Tensor
parallelism counts GPU shards, which equals the node count only when every node
has exactly one GPU — true of a Spark pair and not true in general. It also
ignored pipeline parallelism, so a legitimate two-node ``PP=2, TP=1`` group was
silently overridden to ``TP=2``.

Position now supplies rank and rendezvous; parallelism is declared.

The existing position tests proved rank, node count, master address, host
networking, and ``VLLM_HOST_IP`` — every fact except how many times the tensor
flag appeared. That is the shape again: each half tested, the join untested.
"""

from __future__ import annotations

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.ports.runtime_adapter import NodePosition

pytestmark = pytest.mark.unit


def _position(index: int = 0, count: int = 2) -> NodePosition:
    return NodePosition(
        node_index=index,
        node_count=count,
        self_address=f"10.0.0.1{index}",
        peer_addresses=[f"10.0.0.1{n}" for n in range(count)],
    )


def _argv(config: dict, position: NodePosition | None) -> list[str]:
    return VLLMAdapter().build_launch_args(config, model_path="/models/m", position=position)


def _values_of(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, token in enumerate(argv) if token == flag]


@pytest.mark.parametrize("declared", [1, 2, 4])
def test_the_tensor_flag_appears_exactly_once(declared: int) -> None:
    """The defect itself, across the values that used to collide."""
    argv = _argv({"tensor_parallel_size": declared}, _position())

    assert _values_of(argv, "--tensor-parallel-size") == [str(declared)], argv


def test_the_declared_value_wins_over_the_node_count() -> None:
    """A four-GPU pair declaring TP=4 must not be rewritten to 2.

    The old derivation would have emitted 2 first and 4 second and left the
    parser to choose. Only correct when every node has exactly one GPU.
    """
    argv = _argv({"tensor_parallel_size": 4}, _position(count=2))

    assert _values_of(argv, "--tensor-parallel-size") == ["4"]
    assert _values_of(argv, "--nnodes") == ["2"]


def test_pipeline_only_parallelism_is_not_overridden() -> None:
    """A two-node PP=2, TP=1 group is legitimate and used to be rewritten."""
    argv = _argv({"tensor_parallel_size": 1, "pipeline_parallel_size": 2}, _position())

    assert _values_of(argv, "--tensor-parallel-size") == ["1"]
    assert _values_of(argv, "--pipeline-parallel-size") == ["2"]


def test_a_multi_node_group_states_its_shape_explicitly() -> None:
    """TP=1 across nodes is emitted rather than left to a default."""
    argv = _argv({"tensor_parallel_size": 1, "pipeline_parallel_size": 2}, _position())

    assert "--tensor-parallel-size" in argv


def test_position_no_longer_derives_parallelism() -> None:
    """Rank and rendezvous come from position; parallelism does not.

    Pinned so a future change cannot quietly reintroduce a second source.
    """
    from tensorstead.adapters.runtimes.vllm import _distributed_args

    assert "--tensor-parallel-size" not in _distributed_args(_position())
    assert "--pipeline-parallel-size" not in _distributed_args(_position())
    assert _values_of(_distributed_args(_position(index=1)), "--node-rank") == ["1"]


def test_single_node_argv_is_unchanged() -> None:
    """Every running deployment must be unaffected by all of the above."""
    default = _argv({"tensor_parallel_size": 1}, None)
    explicit = _argv({"tensor_parallel_size": 2}, None)

    assert "--tensor-parallel-size" not in default
    assert _values_of(explicit, "--tensor-parallel-size") == ["2"]
    assert "--nnodes" not in default and "--nnodes" not in explicit


class TestValidateDistribution:
    """The declared group and the declared parallelism must agree."""

    def test_parallelism_smaller_than_the_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot span"):
            VLLMAdapter().validate_distribution({"tensor_parallel_size": 1}, node_count=2)

    def test_the_refusal_names_both_numbers(self) -> None:
        with pytest.raises(ValueError) as caught:
            VLLMAdapter().validate_distribution({"tensor_parallel_size": 1}, node_count=4)

        assert "4 nodes" in str(caught.value)

    def test_pipeline_parallelism_counts_toward_spanning_the_group(self) -> None:
        """TP=1, PP=2 over two nodes is valid and must not be refused."""
        VLLMAdapter().validate_distribution(
            {"tensor_parallel_size": 1, "pipeline_parallel_size": 2}, node_count=2
        )

    def test_the_deepseek_shape_is_accepted(self) -> None:
        """TP=2 across the two Sparks — the deployment this was found blocking."""
        VLLMAdapter().validate_distribution({"tensor_parallel_size": 2}, node_count=2)

    def test_a_single_node_deployment_is_never_refused(self) -> None:
        VLLMAdapter().validate_distribution({"tensor_parallel_size": 1}, node_count=1)

    def test_an_uneven_division_is_left_to_vllm(self) -> None:
        """Deliberately permitted: TP=3 over 2 nodes is vLLM's rule, not ours.

        Refusing here would mean guessing at a runtime we do not ship. vLLM will
        refuse it at launch in its own words, and its refusal will be right where
        ours might not be.
        """
        VLLMAdapter().validate_distribution({"tensor_parallel_size": 3}, node_count=2)
