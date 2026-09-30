"""Pre-flight model-dir validation at the runtime boundary.

The 2026-08-11 incident: a model store corrupted by a concurrent ``model_acquire``
lost its ``config.json`` and was marked ``available`` anyway (a structural verify stops
that at the source). The deployment route then bind-mounted that directory read-only and
started a vLLM container against it; ``create_deployment`` returned
``status: created`` in ~3 seconds, and the failure surfaced only minutes later as
an exited container with ``exit code 1`` and no classified cause. Nothing was
wrong with observation — the runtime was never going to load, and the start path
never asked whether it could.

``validate_model_path`` is the comparison the start path lacked. It is called
before container creation, so a bad model dir fails the start fast with a named
reason (``model_directory_invalid``) instead of "succeeded" then "exit code 1".
The source-side verify checks what was downloaded; this checks what the
runtime will load — the same structural test, at the runtime boundary, so a
directory corrupted between acquire and start (a hand-edit, a partial rsync) is
still caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter

pytestmark = pytest.mark.unit

_DEFAULT_CONFIG = {"arch": "qwen3"}


# --------------------------------------------------------------------------- vLLM


def _vllm_dir(
    tmp_path: Path,
    *,
    config: dict | None = None,
    index: dict | None = None,
    shards: dict[str, bytes] | None = None,
) -> Path:
    """Build a vLLM model tree to order, omitting parts the test wants absent."""
    d = tmp_path / "vllm-model"
    d.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    if index is not None:
        (d / "model.safetensors.index.json").write_text(json.dumps(index))
    for name, body in (shards or {}).items():
        (d / name).write_bytes(body)
    return d


def test_vllm_rejects_a_missing_directory(tmp_path: Path) -> None:
    """No directory at all is the blunt case the start path never checked."""
    adapter = VLLMAdapter()
    with pytest.raises(ValueError, match=r"not a directory"):
        adapter.validate_model_path(tmp_path / "does-not-exist")


def test_vllm_rejects_a_missing_config_json(tmp_path: Path) -> None:
    """The incident itself: the tree exists, config.json does not.

    This is the exact shape that "succeeded" at launch and exited minutes later.
    The message names what is missing and where, so an operator can act.
    """
    adapter = VLLMAdapter()
    d = _vllm_dir(tmp_path, config=None)
    with pytest.raises(ValueError, match=r"missing config\.json"):
        adapter.validate_model_path(d)


def test_vllm_rejects_an_unparsable_config_json(tmp_path: Path) -> None:
    """A config.json that is not valid JSON is the same as no config to vLLM."""
    adapter = VLLMAdapter()
    d = tmp_path / "bad-json"
    d.mkdir()
    (d / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"not valid JSON"):
        adapter.validate_model_path(d)


def test_vllm_rejects_a_missing_shard_named_by_the_index(tmp_path: Path) -> None:
    """An index that names a shard the directory lacks is a half-transferred tree."""
    adapter = VLLMAdapter()
    d = _vllm_dir(
        tmp_path,
        config=_DEFAULT_CONFIG,
        index={"weight_map": {"a": "model-00001-of-00002.safetensors"}},
        shards={"model-00001-of-00002.safetensors": b"shard1"},
        # shard for weight "b" is named in the index below but absent from disk
    )
    # Re-add the index naming both shards but only write one.
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    with pytest.raises(ValueError, match=r"missing shard"):
        adapter.validate_model_path(d)


def test_vllm_rejects_an_empty_shard(tmp_path: Path) -> None:
    """A zero-byte shard is a truncated download, not a usable weight."""
    adapter = VLLMAdapter()
    d = _vllm_dir(
        tmp_path,
        config=_DEFAULT_CONFIG,
        index={"weight_map": {"a": "model-00001-of-00002.safetensors"}},
        shards={"model-00001-of-00002.safetensors": b""},
    )
    with pytest.raises(ValueError, match=r"empty shard"):
        adapter.validate_model_path(d)


def test_vllm_accepts_a_valid_sharded_tree(tmp_path: Path) -> None:
    """The happy path: config + index + all shards present and non-empty."""
    adapter = VLLMAdapter()
    d = _vllm_dir(
        tmp_path,
        config=_DEFAULT_CONFIG,
        index={
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        },
        shards={
            "model-00001-of-00002.safetensors": b"shard1",
            "model-00002-of-00002.safetensors": b"shard2",
        },
    )
    adapter.validate_model_path(d)  # does not raise


def test_vllm_accepts_a_tree_with_only_config_json(tmp_path: Path) -> None:
    """A non-sharded model needs only config.json; no index is not an error."""
    adapter = VLLMAdapter()
    d = _vllm_dir(tmp_path, config=_DEFAULT_CONFIG)
    adapter.validate_model_path(d)  # does not raise


# ------------------------------------------------------------------------- llama.cpp


def test_llamacpp_rejects_a_missing_directory(tmp_path: Path) -> None:
    adapter = LlamaCppAdapter()
    with pytest.raises(ValueError, match=r"not a directory"):
        adapter.validate_model_path(tmp_path / "does-not-exist")


def test_llamacpp_rejects_a_directory_with_no_gguf(tmp_path: Path) -> None:
    """A gguf-less directory would start a container that cannot find a model."""
    adapter = LlamaCppAdapter()
    d = tmp_path / "no-gguf"
    d.mkdir()
    (d / "README.txt").write_text("not a model")
    with pytest.raises(ValueError, match=r"no \.gguf file"):
        adapter.validate_model_path(d)


def test_llamacpp_accepts_a_directory_with_a_gguf(tmp_path: Path) -> None:
    adapter = LlamaCppAdapter()
    d = tmp_path / "gguf-model"
    d.mkdir()
    (d / "model.gguf").write_bytes(b"gguf-bytes")
    adapter.validate_model_path(d)  # does not raise
