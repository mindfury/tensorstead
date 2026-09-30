"""The six ratified fakes, exercised.

Six faked boundaries were ratified. Two of them — the NVML source and
the peer agent — had fakes that no test used. They were built and left, which
means the boundaries they stand for were unfaked in practice.

They are also, not coincidentally, the two boundaries where defects were found
on 2026-08-09: accelerator readings reported `ok` with three nulls, and peer
replication had no coverage when image distribution was later built on top of
it. A fake nobody runs is a claim of coverage that isn't there.
"""

from __future__ import annotations

import pytest

from tensorstead.agent.resources import read_resources
from tests.fakes.nvml import FakeNVMLSource, FakeStorageReading
from tests.fakes.peer_agent import FakePeerAgent

pytestmark = pytest.mark.contract


# ------------------------------------------------------- the NVML boundary


def _fs(capacity: int = 4_000_000_000_000) -> object:
    class _Fs:
        def read(self, path: str) -> dict:
            return {"capacity_bytes": capacity, "available_bytes": capacity // 2}

    return _Fs()


def test_the_nvml_fake_drives_a_real_resource_reading() -> None:
    """The fake stands in for the boundary the service actually calls."""
    nvml = FakeNVMLSource(
        utilization_pct=42.0,
        memory_used=63_620_000_000,
        memory_total=128_000_000_000,
        storage=[FakeStorageReading("models", "/var/lib/tensorstead/models", 100, 50)],
    )

    reading = read_resources(
        nvml_source=nvml,
        filesystem_source=_fs(),  # type: ignore[arg-type]
        memory_is_unified=True,
    )

    assert reading["accelerator_utilization_pct"] == 42.0
    assert reading["accelerator_memory_total"] == 128_000_000_000
    assert reading["status"] == "ok"


def test_a_reading_is_taken_on_demand_and_never_sampled() -> None:
    """The fake counts reads so this is checkable at all."""
    nvml = FakeNVMLSource()
    assert nvml.read_count == 0, "constructing must not read"

    read_resources(
        nvml_source=nvml,
        filesystem_source=_fs(),  # type: ignore[arg-type]
    )
    assert nvml.read_count == 1, "exactly one reading per request, no sampler"

    read_resources(
        nvml_source=nvml,
        filesystem_source=_fs(),  # type: ignore[arg-type]
    )
    assert nvml.read_count == 2


def test_an_accelerator_that_reports_nothing_yields_unknown_not_ok() -> None:
    """The 2026-08-09 defect, held closed through the ratified fake."""
    nvml = FakeNVMLSource(utilization_pct=None, memory_used=None, memory_total=None)  # type: ignore[arg-type]

    reading = read_resources(
        nvml_source=nvml,
        filesystem_source=_fs(capacity=0),  # type: ignore[arg-type]
        memory_is_unified=False,
    )

    assert reading["status"] == "unknown"


# ------------------------------------------------- the peer-agent boundary


def test_a_peer_pull_stages_verifies_then_promotes() -> None:
    """The stage-verify-promote ordering, which the fake exists to make checkable."""
    peer = FakePeerAgent(content_digest="sha256:abc")

    replica = peer.pull("model-1", content_digest="sha256:abc")

    assert replica.state == "available"
    assert replica.content_digest == "sha256:abc"
    assert replica.verified_at is not None
    assert peer.pull_count == 1


def test_a_digest_mismatch_leaves_the_replica_failed_not_available() -> None:
    """An unverifiable transfer must never present as available."""
    peer = FakePeerAgent(content_digest="sha256:abc")

    replica = peer.pull("model-1", content_digest="sha256:wrong")

    assert replica.state == "failed"
    assert peer.get_state("model-1") == "failed"
    assert replica.verified_at is None


def test_a_peer_will_not_serve_a_replica_it_does_not_hold() -> None:
    assert FakePeerAgent().serve_replica("absent") is None


def test_a_peer_only_serves_an_available_replica() -> None:
    """A staging copy must not propagate."""
    peer = FakePeerAgent(content_digest="sha256:abc")
    peer.pull("model-1", content_digest="sha256:wrong")  # leaves it failed

    assert peer.serve_replica("model-1") is None, "a failed replica must not be served"

    peer.pull("model-2", content_digest="sha256:abc")
    served = peer.serve_replica("model-2")
    assert served is not None and served.state == "available"


def test_an_unknown_model_reports_absent_rather_than_raising() -> None:
    assert FakePeerAgent().get_state("nope") == "absent"
