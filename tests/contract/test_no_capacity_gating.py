"""GUARDRAIL: no capacity gating.

No code path compares a storage or accelerator observation against a
requirement to permit, defer, or refuse an operation. The acquisition or
runtime operation stays the final authority. No operation is
refused on resource-capacity grounds.

This is verified by asserting that:

- No product code references both a resource observation field
  (``accelerator_memory_*``, ``storage.*available``, ``capacity_bytes``) and
  a comparison/gate pattern (``<``, ``>``, ``if.*capacity``, ``if.*fit``,
  ``insufficient``, ``not_enough``).
- The deployment creation and model acquisition services do not call
  ``get_resources`` or ``node.resources`` before proceeding.
- No exception or error code related to capacity refusal exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"

# Resource observation field names that, when compared against a threshold,
# would constitute capacity gating.
_RESOURCE_FIELDS = [
    "accelerator_memory_total",
    "accelerator_memory_used",
    "available_bytes",
    "capacity_bytes",
    "accelerator_utilization_pct",
]

# Patterns that, when near a resource field, indicate a capacity comparison.
_GATE_PATTERNS = [
    r"if\s+.*capacity",
    r"if\s+.*fit",
    r"if\s+.*available",
    r"if\s+.*enough",
    r"insufficient",
    r"not_enough",
    r"too_(large|big|small)",
    r"exceeds_capacity",
    r"capacity_exceeded",
    r"resource_exhausted",
]

# Error codes that would indicate a capacity refusal.
_CAPACITY_ERROR_CODES = [
    "insufficient_storage",
    "insufficient_memory",
    "insufficient_accelerator",
    "capacity_exceeded",
    "resource_exhausted",
    "not_enough_space",
    "not_enough_memory",
]


def _product_service_files() -> list[Path]:
    """Service-layer and coordinator files that make operation decisions."""
    return sorted((_SRC / "service").glob("*.py")) + sorted(
        (_SRC / "coordinator" / "routes").glob("*.py")
    )


def test_no_capacity_comparison_in_service_layer() -> None:
    """No service-layer code compares a resource observation against a threshold."""
    offenders: list[str] = []
    for py_file in _product_service_files():
        source = py_file.read_text()
        for field in _RESOURCE_FIELDS:
            if field not in source:
                continue
            # Check if a gate pattern appears near the resource field
            for pattern in _GATE_PATTERNS:
                # Look for the pattern within 5 lines of the resource field
                lines = source.splitlines()
                for i, line in enumerate(lines):
                    if field in line:
                        context = "\n".join(lines[max(0, i - 5) : i + 6])
                        if re.search(pattern, context, re.IGNORECASE):
                            offenders.append(
                                f"{py_file.relative_to(_SRC)}:{i + 1} ({field} near {pattern!r})"
                            )
    assert not offenders, "capacity comparison found in service layer:\n" + "\n".join(offenders)


def test_no_capacity_error_codes_in_domain() -> None:
    """No domain error code refuses on capacity grounds."""
    errors_path = _SRC / "domain" / "errors.py"
    source = errors_path.read_text()

    for code in _CAPACITY_ERROR_CODES:
        assert code not in source, (
            f"domain/errors.py defines {code!r} — the design forbids refusing an "
            "operation on resource-capacity grounds"
        )


def test_deployment_create_does_not_check_resources() -> None:
    """Deployment creation does not call get_resources before proceeding."""
    deployments_path = _SRC / "service" / "deployments.py"
    source = deployments_path.read_text()

    assert "get_resources" not in source, (
        "deployment creation calls get_resources — the design makes the "
        "acquisition/runtime operation the final authority, not a capacity check"
    )
    assert "node.resources" not in source, (
        "deployment creation calls node.resources — capacity must not gate deployment creation"
    )


def test_model_acquisition_does_not_check_resources() -> None:
    """Model acquisition does not call get_resources before proceeding."""
    models_path = _SRC / "service" / "models_.py"
    source = models_path.read_text()

    assert "get_resources" not in source, (
        "model acquisition calls get_resources — the design makes the "
        "acquisition's own failure authoritative, not a capacity pre-check"
    )


def test_no_admission_control_concept() -> None:
    """No admission-control, reservation, or quota concept exists."""
    offenders: list[str] = []
    for py_file in sorted(_SRC.rglob("*.py")):
        source = py_file.read_text()
        for concept in ["admission_control", "reserve_storage", "quota", "reservation"]:
            if concept in source.lower():
                offenders.append(f"{py_file.relative_to(_SRC)}: {concept!r}")
    assert not offenders, "admission-control concept found:\n" + "\n".join(offenders)
