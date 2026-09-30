"""Structural verification before promote.

The defect was an interrupted download leaving a tree *missing* ``config.json`` that
was nonetheless stamped ``state: "available"`` with a ``verified_at``: the
upstream ``acquire`` checked nothing, and ``verified_at`` was set
unconditionally on rename. The fix puts a structural verify between ``acquire``
and ``promote`` so a broken tree is marked ``failed`` and never promoted.
These tests pin the verifier itself and the agent call site.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorstead.adapters.sources.huggingface import HuggingFaceSource
from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.ports.model_source import AcquireProgress
from tests.fakes.hf_hub import FakeHuggingFaceHub

pytestmark = pytest.mark.unit


# ----------------------------------------------------------- the verifier itself


def _source() -> HuggingFaceSource:
    return HuggingFaceSource(hub=FakeHuggingFaceHub())


def test_verify_accepts_a_valid_non_sharded_tree(tmp_path: Path) -> None:
    """A tree with a parseable config.json and no index verifies."""
    (tmp_path / "config.json").write_text(json.dumps({"arch": "llama"}))
    _source().verify(tmp_path)  # no raise


def test_verify_accepts_a_valid_sharded_tree(tmp_path: Path) -> None:
    """A tree whose index names shards that all exist and are non-empty verifies."""
    (tmp_path / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                    "c": "model-00001-of-00002.safetensors",  # repeated shard
                }
            }
        )
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"shard1")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"shard2")
    _source().verify(tmp_path)  # no raise


def test_verify_rejects_missing_config(tmp_path: Path) -> None:
    """A tree missing config.json is the incident shape -- refuse to promote."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match=r"config\.json"):
        _source().verify(tmp_path)


def test_verify_rejects_unparsable_config(tmp_path: Path) -> None:
    """A config.json that is not valid JSON is not a usable tree."""
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        _source().verify(tmp_path)


def test_verify_rejects_a_missing_shard(tmp_path: Path) -> None:
    """A shard named by the index but absent on disk is an incomplete download."""
    (tmp_path / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    # The named shard is not written.
    with pytest.raises(ValueError, match="missing shard"):
        _source().verify(tmp_path)


def test_verify_rejects_an_empty_shard(tmp_path: Path) -> None:
    """A zero-byte shard is not a usable weight file."""
    (tmp_path / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty shard"):
        _source().verify(tmp_path)


def test_verify_rejects_an_unparsable_index(tmp_path: Path) -> None:
    """A safetensors index that is not JSON is not a usable tree."""
    (tmp_path / "config.json").write_text(json.dumps({"arch": "qwen3"}))
    (tmp_path / "model.safetensors.index.json").write_text("not json")
    with pytest.raises(ValueError, match=r"model\.safetensors\.index\.json"):
        _source().verify(tmp_path)


# ----------------------------------------------- the agent call site (acquire path)


class _RecordingSource:
    """A minimal source for the agent path: resolves, acquires a tree, verifies.

    ``verify_raises`` models an interrupted download: ``acquire`` writes a tree
    that ``verify`` then rejects, so the agent must mark the replica failed and
    re-raise rather than promote.
    """

    supports_revision_pinning = True
    requires_credential = False

    def __init__(self, *, verify_raises: bool) -> None:
        self._verify_raises = verify_raises
        self.verify_calls = 0

    def source_id(self) -> str:
        return "test"

    def resolve(self, model_id: str, revision: str | None) -> str:
        return revision or "rev1"

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> object:
        Path(destination_dir).mkdir(parents=True, exist_ok=True)
        (Path(destination_dir) / "config.json").write_text(json.dumps({"arch": "test"}))
        return type("R", (), {"size_bytes": 10, "content_digest": None})()

    def verify(self, destination_dir: Path) -> None:
        self.verify_calls += 1
        if self._verify_raises:
            raise ValueError("model tree is missing config.json")


def test_acquire_calls_verify_before_promote(tmp_path: Path) -> None:
    """A verify that passes lets the acquire promote to available."""
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    source = _RecordingSource(verify_raises=False)

    replica = service.acquire(
        source,
        model_id="test:org/model",
        source_model_id="org/model",
        revision="rev1",
        credential=None,
    )

    assert source.verify_calls == 1, "verify must run between acquire and promote"
    assert replica.state == "available"
    assert replica.verified_at is not None


def test_acquire_marks_failed_and_reraises_when_verify_fails(tmp_path: Path) -> None:
    """A verify that fails marks the replica failed and re-raises -- never promoted."""
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    source = _RecordingSource(verify_raises=True)

    with pytest.raises(ValueError, match=r"missing config\.json"):
        service.acquire(
            source,
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev1",
            credential=None,
        )

    replica = service.get_replica("test:org/model")
    assert replica is not None
    assert replica.state == "failed", (
        "a tree that failed verification must be failed, not available"
    )
    assert replica.verified_at is None, "verified_at must not be stamped on a failed tree"
    # The staging directory must not have been promoted to the final location.
    assert not service.local_path("test:org/model").exists()


def test_acquire_promotes_when_source_has_no_verify(tmp_path: Path) -> None:
    """A source without verify is left alone (the hasattr guard).

    Not every source implements verify; the call is guarded the same way
    ``resolve`` is. A source with no ``verify`` keeps the previous promote
    behaviour, so this test pins the guard against a silent regression.
    """

    class _BareSource:
        supports_revision_pinning = True
        requires_credential = False

        def source_id(self) -> str:
            return "bare"

        def resolve(self, model_id: str, revision: str | None) -> str:
            return revision or "rev1"

        def acquire(
            self,
            model_id: str,
            revision: str,
            destination_dir: str,
            credential: str | None,
            progress: AcquireProgress | None = None,
            file_selector: tuple[str, ...] = (),
        ) -> object:
            Path(destination_dir).mkdir(parents=True, exist_ok=True)
            (Path(destination_dir) / "config.json").write_text(json.dumps({}))
            return type("R", (), {"size_bytes": 1, "content_digest": None})()

    bare = _BareSource()
    assert not hasattr(bare, "verify")

    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")
    replica = service.acquire(
        bare,
        model_id="bare:org/m",
        source_model_id="org/m",
        revision="rev1",
        credential=None,
    )
    assert replica.state == "available"


def test_verify_accepts_a_diffusion_single_file_tree(tmp_path: Path) -> None:
    """A ComfyUI-style tree has no config.json and never will.

    Comfy-Org repositories ship bare ``.safetensors`` sorted into category
    directories; safetensors embeds its own header and the workflow graph
    supplies the rest. Demanding config.json failed the MiniMax H3 acquisition
    at the last step before promotion, after 59 GB had been fetched.
    """
    (tmp_path / "diffusion_models").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "diffusion_models" / "minimax_h3_fl2va.safetensors").write_bytes(b"\x00")
    (tmp_path / "vae" / "minimax_h3_video_vae.safetensors").write_bytes(b"\x00")
    _source().verify(tmp_path)  # no raise


def test_verify_rejects_an_empty_weight_in_a_diffusion_tree(tmp_path: Path) -> None:
    """The structural question is still asked, just of the format that is present."""
    (tmp_path / "diffusion_models").mkdir()
    (tmp_path / "diffusion_models" / "truncated.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="safetensors file is empty"):
        _source().verify(tmp_path)


def test_verify_still_rejects_a_flat_tree_that_lost_its_config(tmp_path: Path) -> None:
    """The diffusion exception must not weaken the check for transformers trees.

    A transformers tree keeps its weights at the root, so an interrupted
    download that lost config.json looks nothing like a category layout and
    still fails -- which is the guarantee the exception was written not to cost.
    """
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    with pytest.raises(ValueError, match=r"missing config\.json"):
        _source().verify(tmp_path)


def test_verify_failure_leaves_no_staging_tree(tmp_path: Path) -> None:
    """An unpromotable tree is disposed of like a failed transfer.

    The two failure paths on either side of the promote had drifted: a failed
    transfer removed its staging directory while a failed verification left
    one behind, for the same reason -- bytes that will never be promoted and
    that no retry reads, because each acquire stages under a fresh UUID.
    """
    service = ModelAcquisitionService(store_dir=tmp_path / "models", marker_dir=tmp_path / "state")

    with pytest.raises(ValueError, match=r"missing config\.json"):
        service.acquire(
            _RecordingSource(verify_raises=True),
            model_id="test:org/model",
            source_model_id="org/model",
            revision="rev1",
            credential=None,
        )

    assert not list((tmp_path / "models").glob("*.staging.*"))
