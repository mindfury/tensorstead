"""The TensorFold adapter.

Each test pins a fact read from ``tensorfold-native`` 1.1.1 itself (``serve
--help``, ``capabilities --json``) or measured on the estate's GB10s, so a
TensorFold change that invalidates one fails here by name rather than as a
container that exits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.tensorfold import TensorFoldAdapter
from tensorstead.ports.runtime_adapter import NodePosition

pytestmark = pytest.mark.unit

_A = TensorFoldAdapter()


def _model(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"]}))
    (directory / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return directory


def _position(index: int, count: int = 2) -> NodePosition:
    peers = ["10.100.89.1", "10.100.89.2"][:count]
    return NodePosition(
        node_index=index, node_count=count, self_address=peers[index], peer_addresses=peers
    )


def _after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# --- single node -------------------------------------------------------------


def test_the_model_path_comes_first_and_the_server_binds_every_interface(tmp_path: Path) -> None:
    model = _model(tmp_path / "m")
    args = _A.build_launch_args({}, model_path=str(model))
    assert args[0] == str(model)
    assert _after(args, "--host") == "0.0.0.0"
    assert "--port" not in args, "single node keeps the in-container default, 8080"


def test_the_entrypoint_is_the_native_binary_serving() -> None:
    assert _A.container_entrypoint({}) == ["tensorfold-native", "serve"]


def test_single_node_publishes_8080_over_bridge_networking() -> None:
    requirements = _A.container_requirements({})
    assert requirements.api_port == 8080
    assert requirements.network_mode is None
    assert requirements.serves_inference
    assert requirements.environment == {"TENSORFOLD_NO_UPDATE_CHECK": "1"}


def test_the_credential_travels_as_environment_never_argv(tmp_path: Path) -> None:
    assert _A.inference_credential_env("k") == {"TENSORFOLD_API_KEY": "k"}
    args = _A.build_launch_args({}, model_path=str(_model(tmp_path / "m")))
    assert not any("api-key" in a for a in args)


def test_modelled_options_render_as_tensorfold_spells_them(tmp_path: Path) -> None:
    config = _A.validate_config(
        {
            "served_model_name": "nemotron-super",
            "context": 32768,
            "parallel": 8,
            "max_tokens": 4096,
            "thinking": False,
            "reasoning_effort": "high",
            "drafts": False,
            "mtp_drafts": 4,
            "kv_cache": "fp8",
            "prompt_cache_gib": 8,
            "vision": True,
            "precision": "checkpoint",
        }
    )
    args = _A.build_launch_args(config, model_path=str(_model(tmp_path / "m")))
    assert _after(args, "--name") == "nemotron-super"
    assert _after(args, "--context") == "32768"
    assert _after(args, "--parallel") == "8"
    assert _after(args, "--max-tokens") == "4096"
    assert "--no-thinking" in args and "--thinking" not in args
    assert _after(args, "--reasoning-effort") == "high"
    assert "--no-drafts" in args
    assert _after(args, "--mtp-drafts") == "4"
    assert _after(args, "--kv-cache") == "fp8"
    assert _after(args, "--prompt-cache-gib") == "8"
    assert "--vision" in args
    assert _after(args, "--precision") == "checkpoint"


def test_extra_args_follow_and_a_false_switch_is_left_off(tmp_path: Path) -> None:
    config = _A.validate_config(
        {"extra_args": {"metrics-open": True, "dashboard": False, "top-k": 20}}
    )
    args = _A.build_launch_args(config, model_path=str(_model(tmp_path / "m")))
    assert "--metrics-open" in args
    assert not any("dashboard" in a for a in args)
    assert _after(args, "--top-k") == "20"


@pytest.mark.parametrize(
    "key",
    ["model", "host", "port", "name", "api-key", "api-key-file", "drafter", "speed-up", "cortheon"],
)
def test_product_owned_and_unrecorded_effects_are_refused_in_extra_args(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _A.validate_config({"extra_args": {key: "x"}})


def test_an_unknown_key_names_the_accepted_ones() -> None:
    with pytest.raises(ValueError, match="accepted tensorfold keys"):
        _A.validate_config({"tensor_parallel_size": 2})


def test_a_drafter_must_be_a_store_path_and_is_mounted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        _A.validate_config({"drafter": "z-lab/Qwen3.8-27B-DFlash2"})
    drafter = _model(tmp_path / "drafter")
    config = _A.validate_config({"drafter": str(drafter)})
    assert _A.container_requirements(config).model_references == [str(drafter)]
    args = _A.build_launch_args(config, model_path=str(_model(tmp_path / "m")))
    assert _after(args, "--drafter") == str(drafter)


def test_a_drafter_missing_on_this_node_refuses_the_launch(tmp_path: Path) -> None:
    config = _A.validate_config({"drafter": str(tmp_path / "absent")})
    with pytest.raises(ValueError, match="not a model directory"):
        _A.build_launch_args(config, model_path=str(_model(tmp_path / "m")))


def test_the_readiness_probe_is_the_authenticated_model_list() -> None:
    probe = _A.readiness_probe()
    assert (probe.path, probe.expect_status) == ("/v1/models", 200)


# --- model path ----------------------------------------------------------------


def test_a_checkpoint_directory_is_accepted(tmp_path: Path) -> None:
    _A.validate_model_path(_model(tmp_path / "m"))


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda d: d.mkdir(), "no config.json"),
        (lambda d: (d.mkdir(), (d / "config.json").write_text("{}")), "no .safetensors"),
    ],
)
def test_an_unusable_directory_is_refused(tmp_path: Path, make: object, message: str) -> None:
    target = tmp_path / "m"
    make(target)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        _A.validate_model_path(target)


# --- two nodes -----------------------------------------------------------------


def test_two_nodes_need_a_declared_interface_and_three_are_refused() -> None:
    with pytest.raises(ValueError, match="distributed_interface"):
        _A.validate_distribution({}, node_count=2)
    _A.validate_distribution({"distributed_interface": "enP2p1s0f0np0"}, node_count=2)
    with pytest.raises(ValueError, match="at most 2"):
        _A.validate_distribution({"distributed_interface": "enP2p1s0f0np0"}, node_count=3)


@pytest.mark.parametrize("index", [0, 1])
def test_each_rank_is_told_its_position_for_both_families(index: int) -> None:
    config = {"distributed_interface": "enP2p1s0f0np0"}
    requirements = _A.container_requirements(config, _position(index))
    env = requirements.environment
    for family in ("NEMOTRON", "GLM"):
        assert env[f"TF_{family}_RANK"] == str(index)
        assert env[f"TF_{family}_WORLD"] == "2"
    assert env["NCCL_COMM_ID"] == "10.100.89.1:29741", "rank 0's fabric address"
    assert env["NCCL_SOCKET_IFNAME"] == "enP2p1s0f0np0"
    assert requirements.network_mode == "host"
    assert requirements.devices == ["/dev/infiniband"]
    assert requirements.capabilities == ["IPC_LOCK"]
    assert requirements.ulimits == {"memlock": (-1, -1)}
    assert requirements.serves_inference is (index == 0)


def test_a_hostname_for_rank_0_is_resolved_to_an_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """TensorFold refuses a hostname rendezvous (``NoCommId``, measured on the estate)."""
    import socket

    def resolve(host: str, *_a: object, **_k: object) -> list[tuple[object, ...]]:
        assert host == "spark-6d42.internal"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.100.89.2", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    position = NodePosition(
        node_index=1,
        node_count=2,
        self_address="spark-471b.internal",
        peer_addresses=["spark-6d42.internal", "spark-471b.internal"],
    )
    env = _A.container_requirements({"distributed_interface": "x"}, position).environment
    assert env["NCCL_COMM_ID"] == "10.100.89.2:29741"


def test_an_unresolvable_rank_0_refuses_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def fail(*_a: object, **_k: object) -> None:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    position = NodePosition(
        node_index=1, node_count=2, self_address="b", peer_addresses=["nowhere.invalid", "b"]
    )
    with pytest.raises(ValueError, match="does not resolve"):
        _A.container_requirements({"distributed_interface": "x"}, position)


def test_rank_0_serves_on_the_deployment_port_and_rank_1_serves_nothing(tmp_path: Path) -> None:
    model = str(_model(tmp_path / "m"))
    config = {"served_model_name": "super", "context": 16384, "distributed_interface": "x"}
    head = _A.build_launch_args(config, model_path=model, position=_position(0), endpoint_port=8150)
    worker = _A.build_launch_args(
        config, model_path=model, position=_position(1), endpoint_port=8150
    )
    assert _after(head, "--port") == "8150" and _after(head, "--name") == "super"
    for serving_flag in ("--host", "--port", "--name"):
        assert serving_flag not in worker
    assert _after(worker, "--context") == "16384", "model-shaping options reach both ranks"


# --- schema probe ----------------------------------------------------------------

_CAPABILITIES = json.dumps(
    {
        "version": "1.1.1",
        "schema": 1,
        "serve": {
            "flags": {
                "--port": {},
                "--kv-cache": {"values": ["bf16", "fp8"]},
                "--max-tokens": {},
            }
        },
    }
)


def test_the_probe_asks_the_binary_with_no_script() -> None:
    probe = _A.schema_probe()
    assert probe.entrypoint == ["tensorfold-native"]
    assert probe.command == ["capabilities", "--json"]
    assert probe.script is None


def test_the_schema_reads_flags_and_choices_and_leaves_shapes_unknown() -> None:
    options = {o.name: o for o in _A.parse_schema("notice: CUDA found\n" + _CAPABILITIES)}
    assert set(options) == {"port", "kv_cache", "max_tokens"}
    assert options["kv_cache"].flags == ["--kv-cache"]
    assert options["kv_cache"].choices == ["bf16", "fp8"]
    assert options["max_tokens"].kind is None, "a type TensorFold does not state is not guessed"
    assert all(o.policy_version for o in options.values())


def test_output_without_the_payload_is_unknown_not_empty() -> None:
    with pytest.raises(ValueError, match=r"serve\.flags"):
        _A.parse_schema('{"version": "1.1.1"}\n')
