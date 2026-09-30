"""A bind is not always a publish.

``port`` is a reserved configuration key, on the reasoning that the endpoint is
the deployment's declared binding and *the container publishes it*. That held
for every deployment until the vLLM adapter began asking for host networking,
which it does for exactly one case: a group spanning nodes.

Under host networking there is no port map. The engine has nothing to translate
with, and nothing was telling the runtime where to listen -- so vLLM bound its
own default while the deployment record named 8010, and `deployment status`
reported the head unreachable on a port no process had ever been asked to bind.
The record described something that was never true, which is the failure this
product exists to prevent.

Found on hardware during the first two-node TP=2 start of the Qwen3.8-27B.
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
        self_address=f"10.100.88.{index + 1}",
        peer_addresses=[f"10.100.88.{n + 1}" for n in range(count)],
    )


def _args(**kwargs: object) -> list[str]:
    return VLLMAdapter().build_launch_args(
        {"tensor_parallel_size": 2},
        model_path="/models/m",
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_group_spanning_nodes_is_told_which_port_to_bind() -> None:
    """The defect, directly.

    Host networking is declared for this case, so no published map exists to
    translate the runtime's default into the declared endpoint.
    """
    args = _args(position=_position(), endpoint_port=8010)

    assert "--port" in args
    assert args[args.index("--port") + 1] == "8010"


def test_every_rank_binds_the_declared_port() -> None:
    """Not just the head.

    Only rank 0 serves an API, but the flag is derived from the deployment
    rather than from rank, and a worker that someday serves must not be
    listening somewhere else.
    """
    for index in (0, 1):
        args = _args(position=_position(index=index), endpoint_port=8010)
        assert args[args.index("--port") + 1] == "8010"


def test_a_single_node_deployment_is_told_nothing() -> None:
    """The load-bearing case for every deployment already running.

    A published port map does the translation, so naming the port here would
    be a second source for a fact the engine already owns -- and would change
    the argv of every existing deployment, which the design was careful to avoid.
    """
    assert "--port" not in _args(position=None, endpoint_port=8010)


def test_a_single_node_position_is_also_told_nothing() -> None:
    """``node_count == 1`` still publishes, so it still translates."""
    assert "--port" not in _args(position=_position(count=1), endpoint_port=8010)


def test_an_unparseable_endpoint_emits_no_flag() -> None:
    """Silence beats inventing a port.

    ``endpoint_port`` returns None for anything it cannot read. Emitting a
    guess would put a number in the argv that the deployment record does not
    contain, which is the same class of untruth this fixes.
    """
    assert "--port" not in _args(position=_position(), endpoint_port=None)


def test_the_port_is_not_accepted_from_configuration() -> None:
    """The reservation still stands.

    The product derives this from the endpoint; a config that also supplied it
    would let the record and the argv disagree with nothing comparing them.
    """
    with pytest.raises(ValueError, match="port"):
        VLLMAdapter().build_launch_args(
            {"tensor_parallel_size": 2, "extra_args": {"port": "9999"}},
            model_path="/models/m",
            position=_position(),
            endpoint_port=8010,
        )
