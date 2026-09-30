"""Agent resource reads on a unified-memory platform.

These tests exist because of a real gap found on DGX Spark: `node_resources`
returned ``status: "ok"`` with every accelerator field null, so an idle node and
an unreadable one looked identical, and NVIDIA's own dashboard could display a
GPU utilization figure that Tensorstead was discarding.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.agent.resources import read_resources

pytestmark = pytest.mark.unit


class _Nvml:
    def __init__(self, reading: dict[str, Any]) -> None:
        self.reading = reading

    def read(self) -> dict[str, Any]:
        return dict(self.reading)


class _Fs:
    def __init__(self, capacity: int = 4_000_000_000_000) -> None:
        self.capacity = capacity

    def read(self, path: str) -> dict[str, Any]:
        return {"capacity_bytes": self.capacity, "available_bytes": self.capacity // 2}


class _Host:
    def __init__(self, reading: dict[str, int] | None) -> None:
        self.reading = reading
        self.calls = 0

    def read(self) -> dict[str, int]:
        self.calls += 1
        return dict(self.reading) if self.reading else {}


NOTHING = {
    "accelerator_utilization_pct": None,
    "accelerator_memory_used": None,
    "accelerator_memory_total": None,
}


def test_utilization_survives_a_failed_memory_read() -> None:
    """The GB10 case: NVML answers utilization but not discrete VRAM.

    Reading both under one ``try`` discarded a figure that had already been
    obtained. The dashboard showed 0% while the product reported nothing.
    """
    host = _Host(None)
    reading = read_resources(
        nvml_source=_Nvml({**NOTHING, "accelerator_utilization_pct": 0.0}),
        filesystem_source=_Fs(),
        host_memory_source=host,
        memory_is_unified=True,
    )

    assert reading["accelerator_utilization_pct"] == 0.0
    assert reading["status"] == "ok"
    assert "accelerator_utilization_pct" not in reading["unreadable_fields"]


def test_unified_memory_is_read_from_the_host_when_nvml_cannot_supply_it() -> None:
    """On unified memory the host's accounting *is* the accelerator's."""
    host = _Host({"used": 63_620_000_000, "total": 128_000_000_000})
    reading = read_resources(
        nvml_source=_Nvml({**NOTHING, "accelerator_utilization_pct": 0.0}),
        filesystem_source=_Fs(),
        host_memory_source=host,
        memory_is_unified=True,
    )

    assert host.calls == 1
    assert reading["accelerator_memory_total"] == 128_000_000_000
    assert reading["accelerator_memory_used"] == 63_620_000_000
    assert reading["unreadable_fields"] == []


def test_the_host_is_not_consulted_when_nvml_supplied_memory() -> None:
    """A discrete-VRAM platform must keep NVML's own figure."""
    host = _Host({"used": 1, "total": 2})
    reading = read_resources(
        nvml_source=_Nvml(
            {
                "accelerator_utilization_pct": 42.0,
                "accelerator_memory_used": 8,
                "accelerator_memory_total": 16,
            }
        ),
        filesystem_source=_Fs(),
        host_memory_source=host,
        memory_is_unified=True,
    )

    assert host.calls == 0
    assert reading["accelerator_memory_total"] == 16


def test_the_host_is_not_consulted_on_a_non_unified_platform() -> None:
    host = _Host({"used": 1, "total": 2})
    reading = read_resources(
        nvml_source=_Nvml(NOTHING),
        filesystem_source=_Fs(),
        host_memory_source=host,
        memory_is_unified=False,
    )

    assert host.calls == 0
    assert reading["accelerator_memory_total"] is None


def test_a_reading_that_obtained_nothing_reports_unknown_not_ok() -> None:
    """An unreadable node must not look like an idle one."""
    reading = read_resources(
        nvml_source=_Nvml(NOTHING),
        filesystem_source=_Fs(capacity=0),
        host_memory_source=_Host(None),
        memory_is_unified=True,
    )

    assert reading["status"] == "unknown"
    assert set(reading["unreadable_fields"]) == {
        "accelerator_utilization_pct",
        "accelerator_memory_used",
        "accelerator_memory_total",
        "storage",
    }


def test_status_stays_within_the_agent_contract_vocabulary() -> None:
    """Widening `status` would be a contract change, not a repair."""
    partial = read_resources(
        nvml_source=_Nvml({**NOTHING, "accelerator_utilization_pct": 7.0}),
        filesystem_source=_Fs(),
        host_memory_source=_Host(None),
        memory_is_unified=True,
    )

    assert partial["status"] in ("ok", "unknown", "unreachable")
    # The gap is still reported, just not as a new status value.
    assert "accelerator_memory_total" in partial["unreadable_fields"]


# --------------------------------------------------------- platform topology


def test_apple_silicon_is_detected_as_unified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mac Studio is the named example; Apple Silicon is unified by construction."""
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app.sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    assert agent_app._detect_unified_memory() is True


def test_intel_mac_is_not_reported_as_unified(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app.sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    assert agent_app._detect_unified_memory() is False


def test_an_unknown_platform_states_nothing_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Not stated' must stay distinct from 'denied'."""
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app.sys, "platform", "win32")

    assert agent_app._detect_unified_memory() is None


def test_a_linux_host_without_a_device_tree_states_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An x86 server with a discrete accelerator has no device tree either."""
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app.sys, "platform", "linux")
    monkeypatch.setattr(
        agent_app.Path, "read_bytes", lambda _self: (_ for _ in ()).throw(OSError())
    )

    assert agent_app._detect_unified_memory() is None


def test_a_grace_device_tree_is_detected_as_unified(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app.sys, "platform", "linux")
    monkeypatch.setattr(agent_app.Path, "read_bytes", lambda _self: b"NVIDIA DGX Spark GB10\x00")

    assert agent_app._detect_unified_memory() is True


def test_an_explicit_override_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator on hardware we cannot detect must be able to say so."""
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app, "_detect_unified_memory", lambda: None)
    monkeypatch.setenv("TENSORSTEAD_MEMORY_IS_UNIFIED", "true")
    assert agent_app._platform_facts_from_env()["memory_is_unified"] is True

    monkeypatch.setenv("TENSORSTEAD_MEMORY_IS_UNIFIED", "false")
    assert agent_app._platform_facts_from_env()["memory_is_unified"] is False


def test_undetected_topology_is_absent_rather_than_asserted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorstead.agent import app as agent_app

    monkeypatch.delenv("TENSORSTEAD_MEMORY_IS_UNIFIED", raising=False)
    monkeypatch.setattr(agent_app, "_detect_unified_memory", lambda: None)

    assert "memory_is_unified" not in agent_app._platform_facts_from_env()


# ------------------------------------------- unified memory


def test_the_driver_is_asked_before_the_board_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fact beats a guess that happens to be right.

    The device-tree read is a name match and can only recognise hardware
    someone already taught the function about. The CUDA driver reports device
    topology directly, so it goes first — and when it answers, the board name is
    not consulted at all.
    """
    from tensorstead.agent import app as agent_app

    consulted: list[str] = []
    monkeypatch.setattr(agent_app, "_detect_unified_memory_cuda", lambda: False)

    def record_and_answer(path: object) -> bytes:
        consulted.append(str(path))
        return b"NVIDIA Grace"

    monkeypatch.setattr(agent_app.Path, "read_bytes", record_and_answer)

    assert agent_app._detect_unified_memory_linux() is False
    assert consulted == [], "the board name was consulted despite the driver answering"


def test_the_board_name_is_the_fallback_not_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the driver cannot answer, a recognised board still counts."""
    from tensorstead.agent import app as agent_app

    monkeypatch.setattr(agent_app, "_detect_unified_memory_cuda", lambda: None)
    monkeypatch.setattr(agent_app.Path, "read_bytes", lambda self: b"NVIDIA GB10 Superchip")

    assert agent_app._detect_unified_memory_linux() is True


def test_an_unrecognised_platform_says_nothing_rather_than_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` is the whole point.

    A node whose topology could not be established must not be recorded as
    having denied unified memory — which is exactly what both DGX Sparks were
    reporting.
    """
    from tensorstead.agent import app as agent_app

    def unreadable(self: object) -> bytes:
        raise OSError("no device tree")

    monkeypatch.setattr(agent_app, "_detect_unified_memory_cuda", lambda: None)
    monkeypatch.setattr(agent_app.Path, "read_bytes", unreadable)

    assert agent_app._detect_unified_memory_linux() is None


def test_dgx_is_not_a_unified_memory_token() -> None:
    """The token that looks most obviously right for this estate is wrong.

    DGX Station and DGX A100 are discrete-GPU machines. Matching "dgx" would
    make the fallback confidently wrong about most of the product line it names,
    which is worse than not recognising the board at all.
    """
    from tensorstead.agent.app import _SOC_UNIFIED_TOKENS

    assert "dgx" not in _SOC_UNIFIED_TOKENS


def test_a_driverless_host_probe_returns_none() -> None:
    """No libcuda, no answer — and no exception escaping into agent startup."""
    from tensorstead.agent.app import _detect_unified_memory_cuda

    # This host has no NVIDIA driver; the probe must degrade, not raise.
    assert _detect_unified_memory_cuda() is None


def test_an_unknown_topology_is_not_reported_as_a_discrete_gpu() -> None:
    """The endpoint that was lying.

    Detection returned None, the key was absent from platform facts, and
    `.get(..., False)` published a confident "no" about hardware that is
    unified. The internal reading still treats unknown as not-unified — letting
    the host figure stand in for accelerator memory on a discrete node reports
    the wrong pool — but the *report* now says unknown.
    """
    from fastapi.testclient import TestClient

    from tensorstead.agent.app import build_agent_app

    app = build_agent_app(management_token="t", replication_token="r")
    app.state.platform_facts = {"cpu_arch": "aarch64"}  # no topology declared
    body = TestClient(app).get("/agent/v1/resources", headers={"Authorization": "Bearer t"}).json()

    assert body["memory_is_unified"] is None, (
        "an undeclared topology was published as a discrete-GPU node"
    )


# ---------------------------------------------------- app.state wiring


def test_computed_platform_facts_actually_reach_app_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug every test above could not have caught.

    ``build_agent_app`` computes ``facts`` correctly -- override honoured,
    detection run -- and hands it to ``/agent/v1/info``, built from the local
    variable directly. No line ever put the same dict on ``app.state``, so
    ``/agent/v1/resources`` and the insufficient_memory start-time guard (both
    read ``state.platform_facts``) saw an empty dict forever, regardless of
    what detection or ``TENSORSTEAD_MEMORY_IS_UNIFIED`` said -- live, on both
    Sparks, since the change that introduced the computation this was supposed to
    feed.

    Every existing test in this file either builds a bare fake ``state`` with
    ``platform_facts`` already on it, or calls ``build_agent_app`` and then
    overwrites ``app.state.platform_facts`` by hand before asserting anything
    (see the test directly above this one). Both routes around the real
    factory, which is exactly how this survived. This test takes neither
    shortcut: real app, real env var, nothing hand-set afterward.
    """
    from tensorstead.agent.app import build_agent_app

    monkeypatch.setenv("TENSORSTEAD_MEMORY_IS_UNIFIED", "true")

    app = build_agent_app(management_token="t", replication_token="r")

    assert app.state.platform_facts.get("memory_is_unified") is True, (
        "build_agent_app computed the fact but never attached it to app.state"
    )
