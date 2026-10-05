"""The ExLlama (TabbyAPI) adapter.

Each test pins a fact read from TabbyAPI's source, so a change upstream that
invalidates one fails here by name rather than as a container that exits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.exllama import ExLlamaAdapter

pytestmark = pytest.mark.unit


def _model(directory: Path, quant_method: str | None = "exl3") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {"architectures": ["LlamaForCausalLM"]}
    if quant_method is not None:
        config["quantization_config"] = {"quant_method": quant_method, "bits": 4.0}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(b"weights")
    return directory


def _flag(args: list[str], flag: str) -> list[str]:
    """Every value following ``flag``, up to the next flag."""
    index = args.index(flag)
    values = []
    for token in args[index + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


# --- launch -----------------------------------------------------------------


def test_the_model_is_named_as_tabbyapi_loads_it(tmp_path: Path) -> None:
    """TabbyAPI loads ``model_dir / model_name``; it takes no direct path."""
    model = _model(tmp_path / "store" / "Qwen-exl3")
    args = ExLlamaAdapter().build_launch_args({}, model_path=str(model))
    assert _flag(args, "--model-dir") == [str(model.parent)]
    assert _flag(args, "--model-name") == ["Qwen-exl3"]


def test_it_binds_every_interface_and_turns_auth_off(tmp_path: Path) -> None:
    model = _model(tmp_path / "m")
    args = ExLlamaAdapter().build_launch_args({}, model_path=str(model))
    assert _flag(args, "--host") == ["0.0.0.0"]
    assert _flag(args, "--disable-auth") == ["true"]


def test_the_entrypoint_runs_main_py() -> None:
    """The image's CMD names main.py, and our argv replaces CMD."""
    assert ExLlamaAdapter().container_entrypoint({}) == ["python3", "main.py"]


def test_booleans_are_spelled_as_values_never_bare_switches(tmp_path: Path) -> None:
    """TabbyAPI declares no switches: a bare ``--vision`` would eat the next flag."""
    model = _model(tmp_path / "m")
    args = ExLlamaAdapter().build_launch_args(
        {"vision": True, "tensor_parallel": False, "extra_args": {"log_prompt": True}},
        model_path=str(model),
    )
    assert _flag(args, "--vision") == ["true"]
    assert _flag(args, "--tensor-parallel") == ["false"]
    assert _flag(args, "--log-prompt") == ["true"]


def test_modelled_options_render_in_tabbyapi_spelling(tmp_path: Path) -> None:
    model = _model(tmp_path / "m")
    args = ExLlamaAdapter().build_launch_args(
        {
            "max_seq_len": 32768,
            "cache_mode": "8,8",
            "cache_size": 65536,
            "gpu_split": [20, 23.5],
        },
        model_path=str(model),
    )
    assert _flag(args, "--max-seq-len") == ["32768"]
    assert _flag(args, "--cache-mode") == ["8,8"]
    assert _flag(args, "--cache-size") == ["65536"]
    assert _flag(args, "--gpu-split") == ["20.0", "23.5"]


def test_unset_options_are_left_to_tabbyapi(tmp_path: Path) -> None:
    model = _model(tmp_path / "m")
    args = ExLlamaAdapter().build_launch_args({}, model_path=str(model))
    assert args == [
        "--model-dir",
        str(model.parent),
        "--model-name",
        "m",
        "--host",
        "0.0.0.0",
        "--disable-auth",
        "true",
    ]


# --- model directory resolution --------------------------------------------


def test_a_single_nested_quantization_is_resolved(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _model(root / "4.0bpw")
    args = ExLlamaAdapter().build_launch_args({}, model_path=str(root))
    assert _flag(args, "--model-dir") == [str(root)]
    assert _flag(args, "--model-name") == ["4.0bpw"]


def test_several_quantizations_are_refused_not_guessed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _model(root / "4.0bpw")
    _model(root / "6.0bpw")
    with pytest.raises(ValueError, match="model_subdir") as caught:
        ExLlamaAdapter().build_launch_args({}, model_path=str(root))
    assert "4.0bpw" in str(caught.value)
    assert "6.0bpw" in str(caught.value)


def test_model_subdir_chooses_between_them(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _model(root / "4.0bpw")
    _model(root / "6.0bpw")
    args = ExLlamaAdapter().build_launch_args({"model_subdir": "6.0bpw"}, model_path=str(root))
    assert _flag(args, "--model-dir") == [str(root)]
    assert _flag(args, "--model-name") == ["6.0bpw"]


@pytest.mark.parametrize("escape", ["../other", "/etc", "a/../../b"])
def test_model_subdir_cannot_leave_the_acquired_model(escape: str) -> None:
    with pytest.raises(ValueError, match="model_subdir"):
        ExLlamaAdapter().validate_config({"model_subdir": escape})


def test_a_model_without_weights_is_refused(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="safetensors"):
        ExLlamaAdapter().validate_model_path(tmp_path)


def test_an_empty_shard_is_not_a_model(tmp_path: Path) -> None:
    """A zero-byte shard is an interrupted download, as for llama.cpp."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="safetensors"):
        ExLlamaAdapter().validate_model_path(tmp_path)


def test_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        ExLlamaAdapter().validate_model_path(tmp_path / "absent")


@pytest.mark.parametrize("quant_method", ["exl3", None])
def test_exl3_and_unquantized_models_are_accepted(tmp_path: Path, quant_method: str | None) -> None:
    ExLlamaAdapter().validate_model_path(_model(tmp_path / "m", quant_method))


@pytest.mark.parametrize("quant_method", ["exl2", "gptq", "EXL2"])
def test_formats_tabbyapi_retired_are_refused_by_name(tmp_path: Path, quant_method: str) -> None:
    model = _model(tmp_path / "m", quant_method)
    with pytest.raises(ValueError, match=quant_method.lower()):
        ExLlamaAdapter().validate_model_path(model)
    with pytest.raises(ValueError, match=quant_method.lower()):
        ExLlamaAdapter().build_launch_args({}, model_path=str(model))


def test_a_chosen_exl2_subdir_is_refused_at_launch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _model(root / "exl3", "exl3")
    _model(root / "exl2", "exl2")
    ExLlamaAdapter().validate_model_path(root)  # one candidate is loadable
    with pytest.raises(ValueError, match="exl2"):
        ExLlamaAdapter().build_launch_args({"model_subdir": "exl2"}, model_path=str(root))


# --- config -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["FP16", "Q8", "Q4", "8,8", "4, 6"])
def test_cache_modes_tabbyapi_accepts(mode: str) -> None:
    assert ExLlamaAdapter().validate_config({"cache_mode": mode})["cache_mode"] == mode


@pytest.mark.parametrize(
    "config",
    [
        {"cache_mode": "Q3"},
        {"cache_mode": "9,9"},
        {"cache_size": 1000},
        {"max_seq_len": 0},
        {"gpu_split": [0]},
        {"gpu_split": []},
        {"n_gpu_layers": 10},
    ],
)
def test_invalid_config_is_refused_naming_the_accepted_keys(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="accepted exllama keys"):
        ExLlamaAdapter().validate_config(config)


def test_max_seq_len_minus_one_reads_it_from_the_model() -> None:
    assert ExLlamaAdapter().validate_config({"max_seq_len": -1}) == {"max_seq_len": -1}


@pytest.mark.parametrize(
    "key",
    [
        "model-dir",
        "model_name",
        "--host",
        "port",
        "disable-auth",
        # Replaces every argument rather than adding one.
        "config",
        # argparse prefix matching reaches the full option.
        "model-n",
        "disable-a",
        "max-seq",
    ],
)
def test_extra_args_cannot_reach_what_the_product_owns(key: str) -> None:
    with pytest.raises(ValueError, match="extra_args"):
        ExLlamaAdapter().validate_config({"extra_args": {key: "x"}})


def test_an_unmodelled_flag_still_passes_through(tmp_path: Path) -> None:
    model = _model(tmp_path / "m")
    config = ExLlamaAdapter().validate_config({"extra_args": {"cpu_moe_offload_layers": 999}})
    args = ExLlamaAdapter().build_launch_args(config, model_path=str(model))
    assert args[-2:] == ["--cpu-moe-offload-layers", "999"]


# --- container and observation ----------------------------------------------


def test_container_requirements_follow_tabbyapis_compose_file() -> None:
    requirements = ExLlamaAdapter().container_requirements({})
    assert requirements.api_port == 5000
    assert requirements.shm_size == 8 * 1024**3
    assert requirements.ulimits == {"memlock": (-1, -1)}
    assert requirements.network_mode is None


def test_readiness_asks_for_the_loaded_model_not_the_directory() -> None:
    assert ExLlamaAdapter().readiness_probe().path == "/v1/model"


def test_no_credential_mechanism_is_declared() -> None:
    """TabbyAPI logs whatever key it is given; see the adapter's docstring."""
    assert ExLlamaAdapter().inference_credential_env("SECRET") == {}


def test_the_open_admin_surface_is_always_advised() -> None:
    notes = ExLlamaAdapter().config_advisories({})
    assert any("/v1/model/load" in note for note in notes)


def test_it_does_not_distribute_or_declare_a_memory_fraction() -> None:
    adapter = ExLlamaAdapter()
    assert adapter.supports_distributed is False
    assert adapter.declared_memory_fraction({"gpu_split": [20]}) is None
    assert adapter.schema_probe() is None
