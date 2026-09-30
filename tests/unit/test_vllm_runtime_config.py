"""The vLLM configuration surface, including speculative decoding.

The 2026-08-10 incident began with an operator wanting MTP — multi-token
prediction — on ``qwen36-27b``. Tensorstead could not express the request at all:
``VLLMConfig`` is ``extra="forbid"`` over a hand-written allowlist, and
``speculative_config`` was not on it, so the deployment was refused before vLLM
ever saw it.

A closed allowlist does not prevent the operator from getting what they need.
It only prevents them from getting it *through the product* — after which the
coordinator's records describe something other than what is running, which is
the drift this product exists to report. A schema that forces operators around
the management plane manufactures its own unexpected instances.

So this covers two things: that the flags the incident named are now
expressible and validated, and that the escape hatch for everything else stays
honest about being unvalidated — without becoming a way to override the
product's own decisions.
"""

from __future__ import annotations

import json

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter

pytestmark = pytest.mark.unit

_MODEL = "/var/lib/tensorstead/models/nvidia/Qwen3.6-27B-NVFP4"


def _args(config: dict) -> list[str]:
    adapter = VLLMAdapter()
    return adapter.build_launch_args(adapter.validate_config(config), model_path=_MODEL)


def _flag_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# ------------------------------------------------------------ speculative/MTP


def test_the_declared_memory_fraction_reaches_argv_unrounded() -> None:
    """0.835 must not become 0.83.

    Two-decimal formatting truncated any finer value, so a deployment declaring
    the MiaAI DeepSeek recipe's 0.835 would have run on a number nobody wrote
    down. The record said one thing and the runtime was told another, with
    nothing comparing them -- the divergence this product exists to prevent.

    Found by rendering that recipe's argv rather than by starting it.
    """
    adapter = VLLMAdapter()
    for declared in (0.835, 0.6, 0.9, 0.125):
        args = adapter.build_launch_args(
            adapter.validate_config({"gpu_memory_utilization": declared}),
            model_path="/models/m",
        )
        rendered = args[args.index("--gpu-memory-utilization") + 1]
        assert float(rendered) == declared, f"{declared} rendered as {rendered}"


def test_mtp_speculative_decoding_is_expressible() -> None:
    """The configuration the incident wanted and could not record."""
    args = _args({"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}})

    payload = json.loads(_flag_value(args, "--speculative-config"))
    assert payload == {"method": "mtp", "num_speculative_tokens": 1}


def test_speculative_config_serializes_deterministically() -> None:
    """The same config must produce byte-identical argv every time.

    Launch arguments are compared against what is running. A dict
    whose JSON key order varied per process would read as drift that nobody
    caused, and would send an operator looking for a change that never happened.
    """
    config = {"speculative_config": {"num_speculative_tokens": 3, "method": "eagle"}}
    first = _flag_value(_args(config), "--speculative-config")
    second = _flag_value(_args(config), "--speculative-config")

    assert first == second
    assert first == '{"method":"eagle","num_speculative_tokens":3}'


def test_a_draft_model_speculative_config_is_accepted() -> None:
    """Not every speculative method is named by ``method``; a draft model is one."""
    args = _args({"speculative_config": {"model": "/models/draft", "num_speculative_tokens": 5}})
    assert json.loads(_flag_value(args, "--speculative-config"))["model"] == "/models/draft"


def test_an_unknown_speculative_method_is_forwarded_not_refused() -> None:
    """This product is not the authority on which methods vLLM supports.

    Validating the method name against a list would refuse a working
    configuration the day vLLM adds one — the closed-allowlist failure,
    reproduced one level down. vLLM decides; the runtime's log tail is how its
    answer becomes visible.
    """
    args = _args({"speculative_config": {"method": "some_method_invented_next_year"}})
    assert "some_method_invented_next_year" in _flag_value(args, "--speculative-config")


@pytest.mark.parametrize(
    ("bad", "because"),
    [
        ({}, "empty"),
        ({"num_speculative_tokens": 2}, "no method and no draft model"),
        ({"method": "", "num_speculative_tokens": 1}, "blank method"),
        ({"method": "mtp", "num_speculative_tokens": 0}, "zero tokens"),
        ({"method": "mtp", "num_speculative_tokens": "two"}, "non-integer tokens"),
    ],
)
def test_incoherent_speculative_config_is_refused(bad: dict, because: str) -> None:
    """The shape we *can* be sure of is still checked before a deployment records it."""
    with pytest.raises(ValueError):
        _args({"speculative_config": bad})


# -------------------------------------------------------------- the rest of 010


def test_the_flags_the_incident_named_are_all_expressible() -> None:
    args = _args(
        {
            "kv_cache_dtype": "fp8",
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "async_scheduling": True,
            "load_format": "safetensors",
        }
    )

    assert _flag_value(args, "--kv-cache-dtype") == "fp8"
    assert _flag_value(args, "--max-num-batched-tokens") == "8192"
    assert _flag_value(args, "--load-format") == "safetensors"
    assert "--enable-prefix-caching" in args
    assert "--enable-chunked-prefill" in args
    assert "--async-scheduling" in args


def test_an_explicit_false_is_not_the_same_as_omission() -> None:
    """``None`` leaves vLLM's default; ``False`` asks for the feature off.

    The difference is load-bearing across a vLLM upgrade that flips a default:
    only the explicit form survives it. Collapsing them would make a deployment
    silently change behaviour because someone else changed their mind.
    """
    explicit = _args({"enable_prefix_caching": False})
    omitted = _args({})

    assert "--no-enable-prefix-caching" in explicit
    assert not any("prefix-caching" in arg for arg in omitted)


def test_omitted_switches_emit_nothing_at_all() -> None:
    assert _args({}) == ["--model", _MODEL]


# ------------------------------------------------------------------ extra_args


def test_an_unmodelled_flag_can_be_forwarded() -> None:
    """The door that keeps operators inside the management plane."""
    args = _args({"extra_args": {"cuda-graph-sizes": "1,2,4"}})
    assert _flag_value(args, "--cuda-graph-sizes") == "1,2,4"


def test_underscores_and_dashes_name_the_same_flag() -> None:
    assert "--long-prefill-token-threshold" in _args(
        {"extra_args": {"long_prefill_token_threshold": 512}}
    )


def test_forwarded_booleans_use_the_paired_switch_form() -> None:
    args = _args({"extra_args": {"some_new_switch": True, "another_one": False}})
    assert "--some-new-switch" in args
    assert "--no-another-one" in args


def test_passthrough_comes_last_so_the_boundary_is_readable() -> None:
    """Where this product's opinions end and the operator's begin, visibly."""
    args = _args({"max_model_len": 4096, "extra_args": {"zzz_flag": "x"}})
    assert args.index("--zzz-flag") > args.index("--max-model-len")


@pytest.mark.parametrize(
    "forbidden",
    ["model", "api_key", "api-key", "port", "host", "served_model_name"],
)
def test_passthrough_cannot_override_what_the_product_owns(forbidden: str) -> None:
    """An escape hatch that reaches these is not an escape hatch.

    ``--model`` would let a deployment serve a path it never acquired, so the
    recorded model id would describe something that is not running.
    ``--api-key`` would put the inference credential on the host-visible
    command line, undoing the whole reason it is delivered as environment.
    ``--host``/``--port`` would leave the runtime listening somewhere other than
    the endpoint the product reports.
    """
    with pytest.raises(ValueError) as exc:
        _args({"extra_args": {forbidden: "anything"}})
    assert forbidden in str(exc.value)


def test_passthrough_of_a_modelled_flag_is_refused_with_the_alternative() -> None:
    """Two routes to one flag would emit it twice; say which route is right."""
    with pytest.raises(ValueError) as exc:
        _args({"extra_args": {"max_model_len": 2048}})
    message = str(exc.value)
    assert "max_model_len" in message
    assert "validated field" in message


def test_the_refusal_message_points_at_the_escape_hatch() -> None:
    """An operator refused for an unknown key must learn there is a way through.

    Without this the error is a dead end, and the next move is SSH — which is
    how the coordinator's records stop describing reality.
    """
    adapter = VLLMAdapter()
    with pytest.raises(ValueError) as exc:
        adapter.validate_config({"totally_unknown_vllm_flag": 1})
    assert "extra_args" in str(exc.value)


def test_validated_config_keeps_modelled_and_forwarded_distinguishable() -> None:
    """The record itself shows which values this product actually checked."""
    validated = VLLMAdapter().validate_config(
        {"max_model_len": 4096, "extra_args": {"cuda-graph-sizes": "1,2"}}
    )
    assert validated["max_model_len"] == 4096
    assert validated["extra_args"] == {"cuda-graph-sizes": "1,2"}


# ------------------------------------------------------- other new modelled flags


def test_enforce_eager_unsticks_a_load_that_dies_during_startup() -> None:
    """The August restore attempt hit a runtime that died during startup.

    Eager execution skips graph capture, which was enough to get past it.
    """
    args = _args({"enforce_eager": True})
    assert "--enforce-eager" in args


def test_trust_remote_code_is_refused_even_though_some_architectures_need_it() -> None:
    """The recorded authorization policy: remote code loads code and is refused.

    Recent architectures -- including the ones carrying MTP heads -- will not
    load without remote code at all, which is a real operational need and not
    a reason to grant it through a bare boolean: the authorization policy
    classifies this as ``LOADS_CODE`` and refuses it, but that classification
    was only ever enforced on the probe/reporting surface, never against an
    actual deployment, because this is a first-class modelled field and never
    passed through ``authorize_extra_arg``. Until remote code loading is a
    separately reviewed, digest-bound decision rather than a runtime boolean
    (the suggested shape), there is no in-product way to grant
    it -- which is the point, not an oversight.
    """
    with pytest.raises(ValueError, match="loads code"):
        VLLMAdapter().validate_config({"trust_remote_code": True})


# ------------------------------------------------ llama.cpp had the same defect


def test_llamacpp_also_forwards_what_it_does_not_model() -> None:
    """Same defect on the llama.cpp adapter -- found by asking whether the fix applied here too.

    llama.cpp carried the identical closed allowlist: six fields and
    ``extra="forbid"``. Nobody had hit it, which is the only reason it looked
    fine. The reasoning that motivated the vLLM change is not vLLM-specific — a
    schema that cannot express what an operator needs does not stop them, it
    sends them around the management plane.
    """
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter

    adapter = LlamaCppAdapter()
    config = adapter.validate_config({"ctx_size": 8192, "extra_args": {"cache_type_k": "q8_0"}})
    args = adapter.build_launch_args(config, model_path="/models/g.gguf")

    assert args[args.index("--cache-type-k") + 1] == "q8_0"


def test_llamacpp_passthrough_has_the_same_boundary() -> None:
    """The credential must not reach argv on any runtime."""
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter

    with pytest.raises(ValueError):
        LlamaCppAdapter().validate_config({"extra_args": {"api_key": "secret"}})


def test_llamacpp_unused_passthrough_leaves_no_trace() -> None:
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter

    assert "extra_args" not in LlamaCppAdapter().validate_config({"ctx_size": 512})


# ----------------------------------------------------- advisories


def test_speculative_decoding_without_a_batch_budget_is_flagged() -> None:
    """The consequence the operator hit after MTP went live.

    vLLM accepts the configuration, then reduces its own per-step token budget
    to accommodate speculative tokens and mentions it in a startup line nobody
    reads until throughput is already disappointing. Valid config, surprising
    result — which is exactly what an advisory is for, as distinct from a
    refusal.
    """
    notes = VLLMAdapter().config_advisories(
        {"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}
    )

    assert any("max_num_batched_tokens" in note for note in notes)


def test_setting_the_batch_budget_silences_the_note() -> None:
    """An advisory that fires after you have acted on it is noise."""
    notes = VLLMAdapter().config_advisories(
        {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
        }
    )

    assert notes == []


def test_an_ordinary_configuration_produces_no_notes() -> None:
    assert VLLMAdapter().config_advisories({"max_model_len": 4096}) == []


def test_advisories_never_refuse() -> None:
    """A note must not become a gate.

    A product that refused on the strength of its own guesses about a runtime
    it does not ship is the closed-allowlist failure wearing a different hat.
    """
    config = {"speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}
    adapter = VLLMAdapter()

    validated = adapter.validate_config(config)
    assert adapter.config_advisories(validated)
    assert adapter.build_launch_args(validated, model_path=_MODEL)


# ------------------------------------------- quantization advisory


def test_a_hardware_dependent_quantization_is_flagged() -> None:
    """The appliance case: NVFP4 weights decompressed rather than executed.

    A dense 27B in ModelOpt NVFP4 logged that the GPU lacks native FP4
    computation, so the weights went through Marlin decompression instead. The
    model serves correctly and slower than expected — possibly slower than not
    quantising at all — and nothing said so before it was chosen, staged, and
    deployed.
    """
    notes = VLLMAdapter().config_advisories(
        {"quantization": "modelopt"}, {"accelerator_compute_capability": "12.1"}
    )

    assert len(notes) == 1
    assert "12.1" in notes[0], "the measured capability belongs in the note"
    assert "deployment runtime" in notes[0], "point at where the runtime states its verdict"


def test_the_advisory_does_not_claim_to_know_the_hardware() -> None:
    """The reason this is a note and not a verdict.

    Encoding which compute capability accelerates which numeric format would
    mean shipping a hardware table, and the first rule I would have written —
    "capability >= 10.0 is Blackwell, therefore FP4" — gets the appliance's own
    GB10 wrong. That is the exact machine the note exists for. Being
    confidently wrong about the one box in the room is worse than saying "this
    depends, and here is where to look".
    """
    note = (
        VLLMAdapter()
        .config_advisories(
            {"quantization": "modelopt"}, {"accelerator_compute_capability": "12.1"}
        )[0]
        .lower()
    )

    for verdict in ("unsupported", "not supported", "lacks", "cannot", "will be slow"):
        assert verdict not in note, f"the advisory asserts a hardware capability: {verdict!r}"
    assert "depends on the accelerator" in note


def test_an_unknown_capability_still_produces_the_note() -> None:
    """The dependency is worth naming even when the fact could not be measured."""
    notes = VLLMAdapter().config_advisories({"quantization": "modelopt"}, {})

    assert len(notes) == 1
    assert "compute capability" not in notes[0], (
        "an unmeasured capability must not be rendered as if it were measured"
    )


def test_an_unquantised_deployment_is_not_advised_about_quantization() -> None:
    assert VLLMAdapter().config_advisories({"max_model_len": 4096}, {}) == []


def test_an_unmanaged_drafter_is_flagged() -> None:
    """The unmanaged drafter: the door ``extra_args`` closes, one level down.

    ``--model`` is derived from an acquired model so a deployment cannot point
    at weights it never staged, and ``extra_args`` refuses ``model`` for that
    reason. ``speculative_config.model`` reaches the same place unguarded: on
    2026-08-12 a drafter given as a HuggingFace repo id worked, and vLLM
    downloaded it during container start. The deployment now depends on weights
    with no model record, no pinned revision and no replica.

    Not refused — a drafter is legitimate and this is how vLLM accepts one.
    Said out loud, because an unrecorded dependency an operator knows about is
    a choice and one they do not know about is a surprise at the next restart.
    """
    notes = VLLMAdapter().config_advisories(
        {
            "speculative_config": {
                "method": "dspark",
                "model": "nvidia/Nemotron-DSpark",
                "num_speculative_tokens": 3,
            },
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
        }
    )

    assert len(notes) == 1
    assert "nvidia/Nemotron-DSpark" in notes[0], "the note must name the drafter"
    assert "does not manage" in notes[0]


def test_speculation_without_a_drafter_is_not_flagged() -> None:
    """MTP uses the model's own head; there is no second model to account for."""
    notes = VLLMAdapter().config_advisories(
        {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
        }
    )

    assert notes == []


# ------------------------------------- self-starting images


def test_the_image_entrypoint_is_overridden_by_default() -> None:
    """Every deployment that exists today keeps its explicit `vllm serve`."""
    adapter = VLLMAdapter()

    assert adapter.container_entrypoint() == ["vllm", "serve"]
    assert adapter.container_entrypoint(adapter.validate_config({})) == ["vllm", "serve"]


def test_a_self_starting_image_is_allowed_to_start_itself() -> None:
    """Self-starting images: the encoder install cannot be a build step.

    The DSpark recipe installs an encoder that ships *inside the acquired model
    checkpoint* into the runtime, at container start, on both ranks. Its bytes
    are pinned to the model revision, not the image, so baking today's copy into
    an image would go stale the moment the model's pin moves — the record that
    stops matching reality, again.

    So the image does that work in its own entrypoint and execs the server
    afterwards. `None` means "use whatever the image declares".
    """
    adapter = VLLMAdapter()
    config = adapter.validate_config({"use_image_entrypoint": True})

    assert adapter.container_entrypoint(config) is None


def test_an_unused_entrypoint_switch_leaves_no_trace() -> None:
    """The `extra_args: {}` trap, not repeated.

    A default of `False` would have appeared in every existing deployment's
    exported configuration, announcing a feature nobody used.
    """
    assert "use_image_entrypoint" not in VLLMAdapter().validate_config({"max_model_len": 4096})


# ------------------------------ drafter MoE backend advisory


def _spec_backend_notes(config: dict) -> list[str]:
    return [n for n in VLLMAdapter().config_advisories(config) if "drafter" in n and "MoE" in n]


def test_a_drafter_without_its_own_moe_backend_is_flagged() -> None:
    """The mistake that cost an operator a working speculative configuration.

    An NVFP4 primary needs a backend an unquantised drafter cannot use. Leaving
    the drafter's unset does not inherit the primary's -- vLLM chooses for the
    drafter on its own, and reports the mismatch as a model-construction
    failure naming a backend and a device, which reads like the backend is
    unsupported on this hardware rather than like it was applied to the wrong
    one of two models. An operator reading it that way concludes speculative
    decoding is impossible here and turns it off.
    """
    notes = _spec_backend_notes(
        {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 8,
            "extra_args": {"moe-backend": "marlin"},
        }
    )

    assert len(notes) == 1
    assert "speculative_config.moe_backend" in notes[0]
    assert "marlin" in notes[0], "the note should name the backend already chosen"


def test_naming_the_drafters_backend_silences_the_note() -> None:
    """The live 35B configuration, which is where the answer came from.

    `qwen36-35b-a3b-nvfp4-perf` has served since 2026-08-16 with marlin for its
    NVFP4 primary and triton for its MTP drafter. If this ever starts producing
    a note, the advisory is contradicting a configuration that runs.
    """
    assert (
        _spec_backend_notes(
            {
                "speculative_config": {
                    "method": "mtp",
                    "num_speculative_tokens": 3,
                    "moe_backend": "triton",
                },
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 8,
                "extra_args": {"moe-backend": "marlin"},
            }
        )
        == []
    )


def test_the_drafters_backend_is_recognised_in_either_spelling() -> None:
    """``moe-backend`` and ``moe_backend`` name the same thing.

    The hyphen/underscore split is the trap that makes this advisory necessary
    in the first place; an advisory that fell for it would fire at an operator
    who had already done the right thing.
    """
    for spelling in ("moe_backend", "moe-backend"):
        config = {
            "speculative_config": {
                "method": "mtp",
                "num_speculative_tokens": 3,
                spelling: "triton",
            },
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 8,
            "extra_args": {spelling: "marlin"},
        }
        assert _spec_backend_notes(config) == [], spelling


def test_no_note_when_the_primary_backend_was_never_chosen() -> None:
    """Nothing to diverge from, so nothing to warn about.

    An operator who set neither took vLLM's defaults for both models. Firing
    here would make the note ambient, and an advisory that cannot be acted on
    is one people learn to scroll past.
    """
    assert (
        _spec_backend_notes(
            {
                "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 8,
            }
        )
        == []
    )


def test_no_note_without_speculative_decoding() -> None:
    """A primary-only backend choice is just a backend choice."""
    assert _spec_backend_notes({"extra_args": {"moe-backend": "marlin"}}) == []


def test_the_drafter_advisory_does_not_prescribe_a_backend() -> None:
    """The design declines to maintain a versioned runtime-capability table.

    The note's job is to say the choice exists and is separate. Naming the
    right value would mean encoding which vLLM release accelerates which
    quantisation on which device -- and the first plausible entry, "Blackwell
    therefore FP4", gets this appliance's own GB10 wrong.
    """
    notes = _spec_backend_notes(
        {
            "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 8,
            "extra_args": {"moe-backend": "marlin"},
        }
    )

    assert notes
    assert "triton" not in notes[0], "the advisory must not prescribe a drafter backend"


def test_the_drafter_advisory_never_refuses() -> None:
    """A note must not become a gate, on the path this note actually fires from."""
    config = {
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 3},
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 8,
        "extra_args": {"moe-backend": "marlin"},
    }
    adapter = VLLMAdapter()

    validated = adapter.validate_config(config)
    assert _spec_backend_notes(validated)
    assert adapter.build_launch_args(validated, model_path=_MODEL)
