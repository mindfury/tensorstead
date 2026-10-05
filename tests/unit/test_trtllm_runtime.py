"""The TensorRT-LLM (``trtllm-serve``) adapter.

Each test pins a fact read from TensorRT-LLM's source, so an upstream change
that invalidates one fails here by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.trtllm import TRTLLMAdapter

pytestmark = pytest.mark.unit


def _checkpoint(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (directory / "model.safetensors").write_bytes(b"weights")
    return directory


def _flag(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# --- launch -----------------------------------------------------------------


def test_the_model_is_the_positional_argument_and_the_server_binds_everywhere() -> None:
    args = TRTLLMAdapter().build_launch_args({}, model_path="/models/llama")
    assert args[0] == "/models/llama"
    assert _flag(args, "--host") == "0.0.0.0"


def test_the_entrypoint_is_trtllm_serve() -> None:
    assert TRTLLMAdapter().container_entrypoint({}) == ["trtllm-serve"]


def test_telemetry_is_off_unless_asked_for() -> None:
    """TensorRT-LLM reports usage to NVIDIA by default."""
    off = TRTLLMAdapter().build_launch_args({}, model_path="/m")
    on = TRTLLMAdapter().build_launch_args({"telemetry": True}, model_path="/m")
    assert "--no-telemetry" in off
    assert "--telemetry" not in off
    assert "--telemetry" in on
    assert "--no-telemetry" not in on


def test_the_default_telemetry_setting_leaves_no_trace_in_the_record() -> None:
    assert TRTLLMAdapter().validate_config({}) == {}
    assert TRTLLMAdapter().validate_config({"telemetry": True}) == {"telemetry": True}


def test_modelled_options_render_in_trtllm_spelling() -> None:
    args = TRTLLMAdapter().build_launch_args(
        {
            "tp_size": 2,
            "max_num_tokens": 8192,
            "kv_cache_free_gpu_memory_fraction": 0.6,
            "kv_cache_dtype": "fp8",
            "served_model_name": "qwen",
            "enable_chunked_prefill": True,
        },
        model_path="/m",
    )
    assert _flag(args, "--tp_size") == "2"
    assert _flag(args, "--max_num_tokens") == "8192"
    assert _flag(args, "--kv_cache_free_gpu_memory_fraction") == "0.6"
    assert _flag(args, "--kv_cache_dtype") == "fp8"
    assert _flag(args, "--served_model_name") == "qwen"
    assert "--enable_chunked_prefill" in args


def test_extra_args_keep_the_operators_spelling() -> None:
    """Click does not equate ``_`` and ``-``, and trtllm-serve uses both."""
    args = TRTLLMAdapter().build_launch_args(
        {"extra_args": {"generation-config": "auto", "enable_attention_dp": True}},
        model_path="/m",
    )
    assert _flag(args, "--generation-config") == "auto"
    assert "--enable_attention_dp" in args


def test_a_false_switch_is_refused_rather_than_dropped() -> None:
    with pytest.raises(ValueError, match="leave the key out"):
        TRTLLMAdapter().validate_config({"extra_args": {"enable_attention_dp": False}})


# --- config -----------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "host",
        "port",
        "grpc",
        "tokenizer",
        "hf_revision",
        "revision",
        "config",
        "extra_llm_api_options",
        "set",
        "custom_module_dirs",
        "post_processor_hook",
        "middleware",
        "custom_tokenizer",
        "report_addr",
        # Second spellings of modelled fields.
        "tensor_parallel_size",
        "free_gpu_memory_fraction",
        "no-telemetry",
        # Modelled fields themselves.
        "tp_size",
        "trust_remote_code",
    ],
)
def test_extra_args_cannot_reach_what_the_product_owns(key: str) -> None:
    with pytest.raises(ValueError, match="extra_args"):
        TRTLLMAdapter().validate_config({"extra_args": {key: "x"}})


@pytest.mark.parametrize(
    "config",
    [
        {"kv_cache_dtype": "int4"},
        {"kv_cache_free_gpu_memory_fraction": 1.5},
        {"tp_size": 0},
        {"tensor_parallel_size": 2},
    ],
)
def test_invalid_config_is_refused_naming_the_accepted_keys(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="accepted trtllm keys"):
        TRTLLMAdapter().validate_config(config)


def test_trust_remote_code_needs_an_approval() -> None:
    with pytest.raises(ValueError, match="trust_remote_code"):
        TRTLLMAdapter().validate_config({"trust_remote_code": True})
    approved = TRTLLMAdapter().validate_config(
        {"trust_remote_code": True}, approved_options=frozenset({"trust_remote_code"})
    )
    assert approved["trust_remote_code"] is True


# --- container and observation ----------------------------------------------


def test_container_requirements_follow_nvidias_docker_guidance() -> None:
    requirements = TRTLLMAdapter().container_requirements({})
    assert requirements.api_port == 8000
    assert requirements.shm_size == 8 * 1024**3
    assert requirements.ulimits == {"memlock": (-1, -1), "stack": (67108864, 67108864)}
    assert requirements.ipc_mode is None


def test_readiness_uses_the_engine_health_route() -> None:
    assert TRTLLMAdapter().readiness_probe().path == "/health"


def test_no_inference_key_mechanism_is_declared() -> None:
    assert TRTLLMAdapter().inference_credential_env("SECRET") == {}


def test_the_kv_cache_share_is_advised_but_never_used_to_refuse() -> None:
    adapter = TRTLLMAdapter()
    assert any("unified-memory" in note for note in adapter.config_advisories({}))
    assert not any(
        "unified-memory" in note
        for note in adapter.config_advisories({"kv_cache_free_gpu_memory_fraction": 0.6})
    )
    assert adapter.declared_memory_fraction({"kv_cache_free_gpu_memory_fraction": 0.9}) is None


def test_it_does_not_distribute() -> None:
    assert TRTLLMAdapter.supports_distributed is False


# --- model directory ----------------------------------------------------------


def test_a_checkpoint_directory_is_accepted(tmp_path: Path) -> None:
    TRTLLMAdapter().validate_model_path(_checkpoint(tmp_path / "m"))


def test_a_directory_without_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"config\.json"):
        TRTLLMAdapter().validate_model_path(tmp_path)


def test_a_missing_shard_is_refused(tmp_path: Path) -> None:
    model = _checkpoint(tmp_path / "m")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    with pytest.raises(ValueError, match="shard"):
        TRTLLMAdapter().validate_model_path(model)
