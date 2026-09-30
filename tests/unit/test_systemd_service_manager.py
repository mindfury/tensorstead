"""Idempotence checks for deployment systemd cleanup."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.agent.service_manager.systemd import SystemdServiceManager


def test_missing_unit_can_be_disabled_and_removed_without_calling_systemd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SystemdServiceManager(systemd_dir=tmp_path)
    calls: list[tuple[str, ...]] = []

    def fake_systemctl(*args: str) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(manager, "_systemctl", fake_systemctl)

    manager.disable("tensorstead-deployment")
    manager.remove("tensorstead-deployment")

    assert calls == []
