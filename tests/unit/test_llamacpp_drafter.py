"""Speculative decoding on llama.cpp: a drafter the agent mounts.

On a node bound by memory bandwidth a single stream reads every weight once
per token, so drafting is the main way to go faster. The drafter is named the
way vLLM and SGLang drafters are -- its store directory -- so the agent decides
whether it is ours to mount, and the adapter refuses combinations that would
load a drafter and never use it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter

pytestmark = pytest.mark.unit


def _gguf(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"GGUF\x00")
    return directory


def _flag(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_a_dflash_drafter_reaches_llama_server(tmp_path: Path) -> None:
    target = _gguf(tmp_path / "store" / "target", "Qwen3.8-27B-UD-Q4_K_XL.gguf")
    drafter = _gguf(
        tmp_path / "store" / "drafter",
        "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
        "Qwen3.8-27B-DFlash2-Q8_0.gguf",
    )
    config = {
        "draft_model": str(drafter),
        "draft_model_file": "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
        "extra_args": {"spec-type": "draft-dflash", "spec-draft-n-max": 7},
    }

    args = LlamaCppAdapter().build_launch_args(config, model_path=str(target))

    assert _flag(args, "--model-draft") == str(drafter / "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
    assert _flag(args, "--spec-type") == "draft-dflash"
    assert _flag(args, "--spec-draft-n-max") == "7"


def test_the_drafter_is_declared_for_the_agent_to_mount(tmp_path: Path) -> None:
    config = {
        "draft_model": "/var/lib/tensorstead/models/drafter",
        "extra_args": {"spec-type": "draft-dflash"},
    }
    assert LlamaCppAdapter().container_requirements(config).model_references == [
        "/var/lib/tensorstead/models/drafter"
    ]
    assert LlamaCppAdapter().container_requirements({}).model_references == []


def test_a_drafter_missing_on_the_node_refuses_the_start(tmp_path: Path) -> None:
    target = _gguf(tmp_path / "target", "m.gguf")
    with pytest.raises(ValueError, match="not a model directory"):
        LlamaCppAdapter().build_launch_args(
            {"draft_model": str(tmp_path / "absent"), "extra_args": {"spec-type": "draft-dflash"}},
            model_path=str(target),
        )


_DFLASH = {"spec-type": "draft-dflash"}


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"draft_model": "/m/drafter"}, "names no drafting type"),
        ({"draft_model": "/m/drafter", "extra_args": {"spec-type": "ngram-simple"}}, "names no"),
        ({"extra_args": _DFLASH}, "needs a drafter"),
        ({"extra_args": {"--spec_type": "draft-eagle3"}}, "needs a drafter"),
        ({"draft_model": "/m/drafter", "extra_args": {"spec-type": "draft-turbo"}}, "not one"),
        ({"draft_model": "relative/drafter", "extra_args": _DFLASH}, "absolute path"),
        ({"draft_model": "/m/../etc", "extra_args": _DFLASH}, "absolute path"),
        (
            {"draft_model": "/m/d", "draft_model_file": "../x.gguf", "extra_args": _DFLASH},
            "relative path",
        ),
        ({"draft_model_file": "x.gguf"}, "needs draft_model"),
        ({"extra_args": {"model-draft": "/elsewhere/x.gguf"}}, "draft_model"),
        ({"extra_args": {"md": "/elsewhere/x.gguf"}}, "draft_model"),
    ],
)
def test_inconsistent_speculative_settings_are_refused(config: dict, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        LlamaCppAdapter().validate_config(config)


@pytest.mark.parametrize("spec_type", ["draft-mtp", "ngram-simple", "ngram-mod,ngram-simple"])
def test_types_without_a_drafter_need_none(tmp_path: Path, spec_type: str) -> None:
    target = _gguf(tmp_path / "target", "m.gguf")
    args = LlamaCppAdapter().build_launch_args(
        {"extra_args": {"spec-type": spec_type}}, model_path=str(target)
    )
    assert _flag(args, "--spec-type") == spec_type
    assert "--model-draft" not in args
