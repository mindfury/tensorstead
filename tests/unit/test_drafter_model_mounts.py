"""A drafter can be an acquired model, and only an acquired model.

Found on hardware. The agent bind-mounts exactly one model
directory, so a drafter acquired through ``model.acquire`` -- recorded, pinned,
replicated onto the node -- was still invisible inside the container. vLLM
rejected its on-node path with "Invalid repository ID or local directory
specified", and the only form that worked was a HuggingFace repo id the runtime
fetched from the network at *every* container start. That is the mechanism
behind an earlier advisory, which stated the symptom without explaining
it.

Closing that gap means letting a configuration string select a bind-mount, which
is the thing ``host_config`` refuses and the design exists to prevent. So the
capability is split: the adapter names where a drafter *may* be (runtime-specific
knowledge), and the agent decides whether the named thing is
something it owns. Most of what follows tests the second half, because that is
where the security property lives -- an adapter naming a path must not be enough
to mount it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.acquisition import ModelAcquisitionService

pytestmark = pytest.mark.unit


def _store(tmp_path: Path) -> ModelAcquisitionService:
    store = tmp_path / "models"
    store.mkdir()
    return ModelAcquisitionService(store_dir=store, marker_dir=tmp_path / "markers")


# ── the agent's half: what may be mounted ─────────────────────────────────


def test_an_acquired_model_directory_resolves(tmp_path: Path) -> None:
    """The whole point: a drafter the operator acquired is mountable."""
    service = _store(tmp_path)
    drafter = tmp_path / "models" / "aHVnZ2luZ2ZhY2U6RHJhZnRlcg"
    drafter.mkdir()

    assert service.managed_model_path(str(drafter)) == str(drafter.resolve())


def test_a_huggingface_repo_id_resolves_to_nothing(tmp_path: Path) -> None:
    """The form that worked before this existed must keep working unchanged.

    It is not a path, nothing is mounted, and the runtime fetches it itself --
    exactly the behaviour the earlier advisory describes.
    """
    service = _store(tmp_path)

    assert service.managed_model_path("Doopeworld/Qwen3.8-27B-DSpark-vLLM") is None


def test_a_host_path_outside_the_store_is_refused(tmp_path: Path) -> None:
    """The security property. Configuration must not name host access.

    Without this, ``speculative_config.model`` would be a general-purpose
    channel for mounting any directory on the host into a container, and a
    deployment record would stop being a true account of what its container can
    reach.
    """
    service = _store(tmp_path)
    elsewhere = tmp_path / "etc"
    elsewhere.mkdir()

    assert service.managed_model_path(str(elsewhere)) is None
    assert service.managed_model_path("/etc") is None


def test_traversal_out_of_the_store_is_refused(tmp_path: Path) -> None:
    """A path that starts inside the store and climbs out of it."""
    service = _store(tmp_path)
    outside = tmp_path / "secrets"
    outside.mkdir()

    escape = tmp_path / "models" / ".." / "secrets"
    assert service.managed_model_path(str(escape)) is None


def test_a_symlink_pointing_out_of_the_store_is_refused(tmp_path: Path) -> None:
    """Resolution follows links, so appearing to live in the store is not enough.

    This is why the check resolves before comparing rather than testing the
    string prefix: a prefix test passes for a symlink whose target is anywhere.
    """
    service = _store(tmp_path)
    outside = tmp_path / "secrets"
    outside.mkdir()
    link = tmp_path / "models" / "looks-legitimate"
    link.symlink_to(outside, target_is_directory=True)

    assert service.managed_model_path(str(link)) is None


def test_the_store_root_itself_is_refused(tmp_path: Path) -> None:
    """Mounting the root would hand the container every model on the node."""
    service = _store(tmp_path)

    assert service.managed_model_path(str(tmp_path / "models")) is None


def test_a_path_that_does_not_exist_resolves_to_nothing(tmp_path: Path) -> None:
    """A drafter named but never acquired is not an error here.

    The runtime is left to do what it did before -- fetch it. Refusing the
    deployment would turn a working configuration into an outage on upgrade.
    """
    service = _store(tmp_path)

    assert service.managed_model_path(str(tmp_path / "models" / "never-acquired")) is None


def test_a_file_is_not_a_model_directory(tmp_path: Path) -> None:
    service = _store(tmp_path)
    stray = tmp_path / "models" / "notes.txt"
    stray.write_text("x")

    assert service.managed_model_path(str(stray)) is None


# ── the adapter's half: where a drafter may be named ──────────────────────


def _requirements(config: dict[str, Any]) -> Any:
    return VLLMAdapter().container_requirements(config)


def test_the_adapter_names_a_drafter_it_was_given() -> None:
    requirements = _requirements(
        {
            "speculative_config": {
                "method": "dspark",
                "model": "/var/lib/tensorstead/models/aHVnZ2luZ2ZhY2U6RHJhZnRlcg",
                "num_speculative_tokens": 7,
            }
        }
    )

    assert requirements.model_references == [
        "/var/lib/tensorstead/models/aHVnZ2luZ2ZhY2U6RHJhZnRlcg"
    ]


def test_a_repo_id_is_still_named() -> None:
    """The adapter reports the value verbatim and judges nothing.

    Deciding whether a string is an acquired model needs the agent's store
    root, which the adapter cannot see -- so it must not try.
    """
    requirements = _requirements(
        {
            "speculative_config": {
                "method": "dspark",
                "model": "Org/Drafter",
                "num_speculative_tokens": 7,
            }
        }
    )

    assert requirements.model_references == ["Org/Drafter"]


def test_mtp_names_no_drafter() -> None:
    """The in-checkpoint head is not a separate set of weights.

    vLLM shares the target model's embeddings and lm_head with it, so there is
    nothing extra to mount -- which is most of why MTP costs no memory.
    """
    requirements = _requirements(
        {"speculative_config": {"method": "mtp", "num_speculative_tokens": 5}}
    )

    assert requirements.model_references == []


def test_no_speculative_config_names_no_drafter() -> None:
    assert _requirements({"max_model_len": 4096}).model_references == []


# ── the primary path: same containment, no fallback ────────────────────────
#
# require_model_path shares its containment logic with managed_model_path
# above (_resolve_within_store) -- traversal, symlink escape, and "does not
# exist" are already covered there and are not re-derived here. What is
# distinct about require_model_path, and worth its own tests: it raises
# instead of returning None, and it returns a Path rather than a str.


def test_require_model_path_accepts_a_real_replica(tmp_path: Path) -> None:
    service = _store(tmp_path)
    replica_dir = service.local_path("m1")
    replica_dir.mkdir(parents=True)

    resolved = service.require_model_path(str(replica_dir))

    assert resolved == replica_dir.resolve()


def test_require_model_path_refuses_a_path_outside_the_store(tmp_path: Path) -> None:
    service = _store(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    with pytest.raises(ValueError, match="managed store"):
        service.require_model_path(str(outside))


def test_require_model_path_refuses_a_symlink_escape(tmp_path: Path) -> None:
    service = _store(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = service.local_path("m1")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        service.require_model_path(str(link))


def test_require_model_path_refuses_a_path_that_does_not_exist(tmp_path: Path) -> None:
    service = _store(tmp_path)

    with pytest.raises(ValueError):
        service.require_model_path(str(service.local_path("never-acquired")))
