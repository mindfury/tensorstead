"""``_endpoint_authenticated`` reports the deployment's own runtime, not any.

Discovered live: `deployment status` on a freshly-started, genuinely
unauthenticated ComfyUI deployment reported "inference endpoint requires a
key" (ComfyUI itself was later removed from the estate -- but the
bug it exposed was general, not ComfyUI-specific). A direct, header-free
request to the same endpoint returned 200. The function was computing
``any(adapter has a mechanism for adapter in every-adapter-this-agent-knows-
about)`` instead of asking the one adapter this deployment actually runs — so
vLLM's credential mechanism, present on every agent, made every *other*
runtime on that agent report authenticated regardless of what its own adapter
says. llama.cpp and SGLang both carry the same `{}`
credential mechanism ComfyUI did, so they hold the regression covered.

Reads the container's own recorded environment rather than the agent's —
these tests construct a ``ContainerState``
directly instead of setting ``TENSORSTEAD_INFERENCE_API_KEY`` in the process
environment, which is what changed and why.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from starlette.requests import Request

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.container_engine.base import ContainerState
from tensorstead.agent.routes.observation import _endpoint_authenticated

pytestmark = pytest.mark.unit

_ADAPTERS = {
    "vllm": VLLMAdapter(),
    "llamacpp": LlamaCppAdapter(),
    "sglang": SGLangAdapter(),
}


def _request(adapters: dict = _ADAPTERS) -> Request:
    """A minimal stand-in carrying only what ``_endpoint_authenticated`` reads.

    The function touches ``request.app.state.runtime_adapters`` and nothing
    else, so a duck-typed ``SimpleNamespace`` is a faithful test double; the
    cast tells mypy what pytest already knows at runtime.
    """
    stub = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_adapters=adapters)))
    return cast(Request, stub)


def _container(environment: dict[str, str] | None = None) -> ContainerState:
    return ContainerState(running=True, environment=environment or {})


def test_no_key_installed_is_false_regardless_of_runtime() -> None:
    container = _container()
    assert _endpoint_authenticated(container, _request(), "vllm") is False
    assert _endpoint_authenticated(container, _request(), "llamacpp") is False


def test_unknown_runtime_type_is_none_not_false() -> None:
    """None means 'we cannot say'; it must not collapse into an availability answer."""
    container = _container({"VLLM_API_KEY": "secret"})
    assert _endpoint_authenticated(container, _request(), None) is None
    assert _endpoint_authenticated(container, _request(), "some-future-runtime") is None


def test_vllm_reports_authenticated_when_its_key_is_actually_installed() -> None:
    container = _container({"VLLM_API_KEY": "secret"})
    assert _endpoint_authenticated(container, _request(), "vllm") is True


def test_vllm_reports_unauthenticated_when_the_key_was_never_installed_on_this_container() -> None:
    """The core proof of the recorded-environment requirement.

    A key existing *somewhere* (the agent's own environment, another
    deployment's container) must not be read as this deployment's own
    authentication -- only what is actually in *this* container's recorded
    environment counts.
    """
    container = _container()  # nothing installed on this specific container
    assert _endpoint_authenticated(container, _request(), "vllm") is False


def test_a_mechanism_less_runtime_ignores_a_key_installed_for_a_different_runtime() -> None:
    """The regression this file exists to hold fixed.

    The container happens to carry ``VLLM_API_KEY`` -- vllm's mechanism is
    right there in its environment -- and llamacpp/sglang must still report
    their own adapter's answer (no mechanism at all), not vllm's.
    """
    container = _container({"VLLM_API_KEY": "secret"})
    assert _endpoint_authenticated(container, _request(), "llamacpp") is False
    assert _endpoint_authenticated(container, _request(), "sglang") is False


def test_no_container_is_false_for_a_known_runtime_and_none_for_an_unknown_one() -> None:
    """No container to read from -- the not-running-at-all shape."""
    assert _endpoint_authenticated(None, _request(), "vllm") is False
    assert _endpoint_authenticated(None, _request(), None) is None
