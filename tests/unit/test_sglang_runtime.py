"""The SGLang adapter — the third runtime, and what it proves about the seam.

Two adapters can share a defect and look like agreement. A third is what shows
the seam is a seam: SGLang distributes like vLLM but spells nothing the way
vLLM does, and it inherits the ``extra_args`` authorization rule without
restating it — which is exactly what that module was extracted for after two
adapters implemented the rule separately and both got it wrong the same way.

The configuration exercised here is the published
``MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`` recipe, so these tests fail if the
adapter stops being able to express the thing it was written for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.ports.runtime_adapter import NodePosition

pytestmark = pytest.mark.unit

_MODEL = "/var/lib/tensorstead/models/qwen38-27b"

# The recipe's DSpark variant, in SGLang's own vocabulary.
_RECIPE = {
    "mem_fraction_static": 0.90,
    "context_length": 262144,
    "max_running_requests": 10,
    "chunked_prefill_size": 8192,
    "kv_cache_dtype": "fp8_e4m3",
    "max_mamba_cache_size": 40,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "speculative_algorithm": "dspark",
    "speculative_dspark_block_size": 7,
    "speculative_num_draft_tokens": 8,
    "enable_metrics": True,
}


def _args(config: dict, position: NodePosition | None = None, port: int | None = None) -> list[str]:
    adapter = SGLangAdapter()
    return adapter.build_launch_args(
        adapter.validate_config(config), model_path=_MODEL, position=position, endpoint_port=port
    )


def test_the_published_recipe_is_expressible() -> None:
    """Every flag the recipe sets survives validation and reaches argv."""
    args = _args(_RECIPE)
    for flag, value in (
        ("--model-path", _MODEL),
        ("--mem-fraction-static", "0.9"),
        ("--context-length", "262144"),
        ("--max-running-requests", "10"),
        ("--chunked-prefill-size", "8192"),
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--max-mamba-cache-size", "40"),
        ("--reasoning-parser", "qwen3"),
        ("--tool-call-parser", "qwen3_coder"),
        ("--speculative-algorithm", "dspark"),
        ("--speculative-dspark-block-size", "7"),
        ("--speculative-num-draft-tokens", "8"),
    ):
        assert flag in args, f"{flag} missing"
        assert args[args.index(flag) + 1] == value
    assert "--enable-metrics" in args


def test_it_binds_all_interfaces_inside_the_container() -> None:
    """SGLang defaults to loopback, like llama.cpp and unlike vLLM.

    The container reports healthy, logs that the server is ready, and answers
    nothing through the published port map.
    """
    assert _args({})[_args({}).index("--host") + 1] == "0.0.0.0"


def test_the_model_path_is_derived_not_declared() -> None:
    """An operator cannot point the runtime at weights it has not acquired."""
    assert _args({})[:2] == ["--model-path", _MODEL]
    with pytest.raises(ValueError, match=r"model-path|model_path"):
        SGLangAdapter().validate_config({"extra_args": {"model-path": "/tmp/other"}})


def test_it_inherits_the_reserved_flag_rule_without_restating_it() -> None:
    """The alternate spellings that defeated two adapters separately (034).

    Embedded values and negative aliases both reproduced before the rule was
    extracted; a third adapter getting them right for free is the point.
    """
    for hostile in (
        {"model-path=/tmp/other": True},
        {"api_key": "secret"},
        {"--host": "0.0.0.0"},
        {"port": 9999},
        {"nnodes": 4},
        {"node-rank": 1},
    ):
        with pytest.raises(ValueError):
            SGLangAdapter().validate_config({"extra_args": hostile})


def test_a_modelled_option_cannot_also_be_forwarded() -> None:
    """Otherwise the same flag reaches argv twice with two values."""
    with pytest.raises(ValueError):
        SGLangAdapter().validate_config(
            {"context_length": 262144, "extra_args": {"context-length": 4096}}
        )


def test_unmodelled_flags_pass_through_verbatim() -> None:
    args = _args({"extra_args": {"cpuset-cpus": "5-9,15-19", "enable-cache-report": True}})
    assert args[args.index("--cpuset-cpus") + 1] == "5-9,15-19"
    assert "--enable-cache-report" in args


# ------------------------------------------------------------------ topology
def test_a_single_node_deployment_states_no_group() -> None:
    """Distribution must not leak into the argv of a deployment that has none."""
    args = _args(_RECIPE, position=NodePosition(0, 1, "10.100.88.1", ["10.100.88.1"]))
    for flag in ("--nnodes", "--node-rank", "--dist-init-addr", "--port"):
        assert flag not in args


def test_a_group_that_spans_nodes_derives_every_rank_flag() -> None:
    peers = ["10.100.88.1", "10.100.88.2"]
    head = _args({"tp_size": 2}, position=NodePosition(0, 2, peers[0], peers), port=8010)
    worker = _args({"tp_size": 2}, position=NodePosition(1, 2, peers[1], peers), port=8010)

    assert head[head.index("--node-rank") + 1] == "0"
    assert worker[worker.index("--node-rank") + 1] == "1"
    for args in (head, worker):
        assert args[args.index("--nnodes") + 1] == "2"
        # Both ranks rendezvous at the first declared participant.
        assert args[args.index("--dist-init-addr") + 1] == "10.100.88.1:29500"
        # Host networking removes the published port map, so
        # the runtime must be told the port the record declares.
        assert args[args.index("--port") + 1] == "8010"


def test_a_shape_that_cannot_be_formed_is_refused_before_anything_starts() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        SGLangAdapter().validate_distribution({"tp_size": 3}, node_count=2)
    SGLangAdapter().validate_distribution({"tp_size": 2}, node_count=2)


def test_only_rank_zero_is_reported_as_serving() -> None:
    """The same derivation argv uses, so observation cannot disagree (029)."""
    peers = ["a", "b"]
    adapter = SGLangAdapter()
    assert adapter.container_requirements({}, NodePosition(0, 2, "a", peers)).serves_inference
    assert not adapter.container_requirements({}, NodePosition(1, 2, "b", peers)).serves_inference


# ----------------------------------------------------------------- container
def test_it_declares_sglangs_own_port_not_another_runtimes() -> None:
    """The agent held one runtime's port for every runtime."""
    assert SGLangAdapter().container_requirements({}).api_port == 30000


def test_it_declares_shared_memory_because_ranks_share_it() -> None:
    assert SGLangAdapter().container_requirements({}).shm_size == 8 * 1024**3
    assert (
        SGLangAdapter().container_requirements({"host_config": {"shm_size": 1234}}).shm_size == 1234
    )


def test_a_drafter_is_declared_as_a_second_set_of_weights() -> None:
    """Only SGLang knows which of its keys names a model."""
    requirements = SGLangAdapter().container_requirements(
        {"speculative_draft_model_path": "z-lab/Qwen3.8-27B-DFlash2"}
    )
    assert requirements.model_references == ["z-lab/Qwen3.8-27B-DFlash2"]
    assert SGLangAdapter().container_requirements({}).model_references == []


def test_it_serves_unauthenticated_and_says_so() -> None:
    """Declaring no mechanism is honest; guessing an env var would not be.

    A guessed variable name produces a deployment that looks authenticated and
    is not, which is worse than one that reports the truth.
    """
    assert SGLangAdapter().inference_credential_env("secret") == {}


def test_it_reports_its_option_surface_as_unknown_rather_than_empty() -> None:
    assert SGLangAdapter().schema_probe() is None


def test_readiness_asks_a_question_only_a_loaded_model_answers() -> None:
    probe = SGLangAdapter().readiness_probe()
    assert probe is not None
    assert probe.path == "/v1/models"


# --------------------------------------------------------------- model trees
def test_a_tree_sglang_cannot_load_fails_the_start_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"config\.json"):
        SGLangAdapter().validate_model_path(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    SGLangAdapter().validate_model_path(tmp_path)


def test_a_missing_directory_is_named_as_such(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        SGLangAdapter().validate_model_path(tmp_path / "absent")


# ---------------------------------------------------------------- advisories
def test_advisories_report_interactions_without_refusing_them() -> None:
    """Advice, never a reason a request was refused."""
    notes = SGLangAdapter().config_advisories(
        {"speculative_algorithm": "dflash2", "mem_fraction_static": 0.99}
    )
    assert any("speculative_draft_model_path" in n for n in notes)
    assert any("headroom" in n for n in notes)
    # The same config still validates: an advisory is not a refusal.
    SGLangAdapter().validate_config({"speculative_algorithm": "dflash2"})


def test_dspark_needs_a_drafter_too() -> None:
    """The recipe's own variant, and the one the advisory used to miss.

    Starting it produced `ValueError: DSpark dense speculative decoding
    requires setting --speculative-draft-model-path`.
    """
    notes = SGLangAdapter().config_advisories({"speculative_algorithm": "dspark"})
    assert any("speculative_draft_model_path" in n for n in notes)


def test_the_memory_fraction_is_reported_as_a_share_of_the_whole_node() -> None:
    """The trap that cost this estate a node.

    SGLang sizes against *total* device memory and assumes it owns the device,
    so on a node already serving something else this is not "the share I may
    use" but "the share I will take".
    """
    notes = SGLangAdapter().config_advisories({"mem_fraction_static": 0.90})
    assert any("TOTAL" in n and "already serving" in n for n in notes)


# --------------------------------------------------------------- trust_remote_code


def test_trust_remote_code_is_refused() -> None:
    """Same first-class-field gap as vLLM's identical field: this is modelled
    config, so it never passed through ``authorize_extra_arg`` at all -- the
    authorization policy's ``LOADS_CODE`` classification was never actually
    enforced against a real deployment before this fix.
    """
    with pytest.raises(ValueError, match="loads code"):
        SGLangAdapter().validate_config({"trust_remote_code": True})


def test_trust_remote_code_left_unset_is_unaffected() -> None:
    """The refusal must not tax the ordinary case: most models need nothing."""
    validated = SGLangAdapter().validate_config({"enable_metrics": True})
    assert validated["enable_metrics"] is True
    assert "trust_remote_code" not in validated or validated["trust_remote_code"] is None
