"""Static checks that readiness remains read-only and TLS-verified."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]


def test_readiness_checks_prerequisites_and_services() -> None:
    readiness = (_ROOT / "ansible/playbooks/readiness.yml").read_text()

    for expected in ("docker info", "nvidia-container-cli --version", "service_facts"):
        assert expected in readiness
    assert "/health" in readiness
    assert "/agent/v1/info" in readiness


def test_readiness_uses_tls_verification_and_no_workload_operations() -> None:
    readiness = (_ROOT / "ansible/playbooks/readiness.yml").read_text()

    assert "validate_certs: true" in readiness
    assert "ca_path:" in readiness
    for forbidden in ("node register", "models:acquire", "deployments", "image pull"):
        assert forbidden not in readiness
