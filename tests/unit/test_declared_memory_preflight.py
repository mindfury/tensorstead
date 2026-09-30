"""A start whose declared memory the node cannot satisfy is refused.

The agent knew the node's free memory. It knew the deployment's declared
fraction. It compared them nowhere.

Starting SGLang at ``mem_fraction_static: 0.90`` on a node with 13% free asked
for roughly 109 GB of a 121 GB device and left the host swap-thrashing until it
needed a power cycle — kernel alive, no shell, no service, no BMC.

The trap is that the fraction is a share of the device's **total** memory, not
of what is free. On an idle node 0.90 is reasonable and is what the vendor
recipe prescribes; on a node already serving something else it is a demand for
memory that does not exist.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.routes.deployments import _refuse_if_memory_cannot_hold

pytestmark = pytest.mark.unit

_GIB = 1024**3
_TOTAL = 121 * _GIB


class _Payload:
    runtime_type = "sglang"


def _state(used: int | None, total: int | None) -> Any:
    """An agent state whose accelerator reading is what the test says."""

    class _Nvml:
        def read(self) -> dict[str, Any]:
            return {
                "accelerator_utilization_pct": 0,
                "accelerator_memory_used": used,
                "accelerator_memory_total": total,
            }

    return type("S", (), {"nvml_source": _Nvml(), "fs_source": None, "platform_facts": {}})()


# ------------------------------------------------------- what each runtime says
def test_each_adapter_reports_its_own_spelling() -> None:
    """The number is the adapter's; nobody else may guess it."""
    assert VLLMAdapter().declared_memory_fraction({"gpu_memory_utilization": 0.6}) == 0.6
    assert SGLangAdapter().declared_memory_fraction({"mem_fraction_static": 0.9}) == 0.9


def test_a_runtime_with_no_budget_says_so_rather_than_guessing() -> None:
    """llama.cpp sizes from the model and ctx_size; None is the truthful answer."""
    assert LlamaCppAdapter().declared_memory_fraction({"ctx_size": 4096}) is None
    assert VLLMAdapter().declared_memory_fraction({}) is None


# ---------------------------------------------------------------- the pre-flight
def test_the_start_that_took_a_node_down_is_now_refused() -> None:
    """0.90 declared against a node with ~13% free."""
    state = _state(used=105 * _GIB, total=_TOTAL)
    with pytest.raises(HTTPException) as caught:
        _refuse_if_memory_cannot_hold(
            state, SGLangAdapter(), {"mem_fraction_static": 0.90}, _Payload()
        )
    detail: dict[str, Any] = caught.value.detail  # type: ignore[assignment]
    assert caught.value.status_code == 422
    assert detail["code"] == "insufficient_memory"
    assert detail["detail"]["declared_fraction"] == 0.90
    # The message must say why the fraction is not what it looks like.
    assert "not of what is free" in detail["message"]


def test_the_same_fraction_on_an_idle_node_is_allowed() -> None:
    """0.90 is what the vendor recipe prescribes, and it is fine when it fits."""
    state = _state(used=2 * _GIB, total=_TOTAL)
    _refuse_if_memory_cannot_hold(state, SGLangAdapter(), {"mem_fraction_static": 0.90}, _Payload())


def test_a_budget_that_exactly_fits_is_allowed() -> None:
    """The boundary is not a refusal: declared == available must pass."""
    state = _state(used=int(_TOTAL * 0.10), total=_TOTAL)
    _refuse_if_memory_cannot_hold(state, SGLangAdapter(), {"mem_fraction_static": 0.90}, _Payload())


def test_a_runtime_declaring_nothing_is_never_refused() -> None:
    """llama.cpp has no fraction, so there is nothing to compare."""
    state = _state(used=120 * _GIB, total=_TOTAL)
    _refuse_if_memory_cannot_hold(state, LlamaCppAdapter(), {"ctx_size": 4096}, _Payload())


def test_unreadable_memory_is_not_a_refusal() -> None:
    """Observation must never be the reason a start fails.

    A node whose accelerator the agent cannot read reports unknown. Refusing on
    an unread number would ground the estate on a monitoring gap -- the same
    mistake as reporting a healthy node broken.
    """
    for used, total in ((None, _TOTAL), (105 * _GIB, None), (None, None)):
        _refuse_if_memory_cannot_hold(
            _state(used, total), SGLangAdapter(), {"mem_fraction_static": 0.90}, _Payload()
        )


def test_an_adapter_predating_the_check_is_not_refused() -> None:
    """Silence from an older adapter is not a claim that it needs nothing."""

    class _Old:
        pass

    _refuse_if_memory_cannot_hold(
        _state(105 * _GIB, _TOTAL), _Old(), {"mem_fraction_static": 0.9}, _Payload()
    )
