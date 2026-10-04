"""A start must not silently discard an inference credential.

SGLang and llama.cpp both implement ``inference_credential_env`` as an
unconditional ``{}`` -- deliberately, since neither has a verified
authentication mechanism, and each adapter's docstring explains why guessing
one would be worse than declaring none. The bug this closes was never that
decision: it was the generic start path turning that empty mapping into "no
environment" and continuing anyway. An operator who bound a credential and
watched the start succeed had every reason to believe it applied; instead the
endpoint published unauthenticated.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tensorstead.adapters.runtimes.exllama import ExLlamaAdapter
from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.routes.deployments import _refuse_if_credential_cannot_be_enforced

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("adapter", [SGLangAdapter(), ExLlamaAdapter()])
def test_a_supplied_credential_the_runtime_cannot_enforce_refuses_the_start(
    adapter: object,
) -> None:
    with pytest.raises(HTTPException) as caught:
        _refuse_if_credential_cannot_be_enforced(adapter, "some-secret")

    detail: dict[str, object] = caught.value.detail  # type: ignore[assignment]
    assert caught.value.status_code == 422
    assert detail["code"] == "credential_not_enforceable"
    assert adapter.runtime_type in detail["message"]  # type: ignore[attr-defined,operator]


@pytest.mark.parametrize("adapter", [SGLangAdapter(), LlamaCppAdapter(), ExLlamaAdapter()])
def test_no_credential_at_all_is_still_a_legitimate_deployment(adapter: object) -> None:
    """An operator who never asked for authentication must still be able to
    deploy SGLang/llama.cpp -- the guard fires only when a key was supplied
    and discarded, never merely because the runtime has no mechanism at all.
    This is the property a helper that only saw the resulting (empty either
    way) environment mapping could not have told apart.
    """
    assert _refuse_if_credential_cannot_be_enforced(adapter, None) is None
    assert _refuse_if_credential_cannot_be_enforced(adapter, "") is None


def test_a_runtime_that_can_enforce_the_credential_returns_its_environment() -> None:
    """vLLM's own mechanism must not be caught by this guard, and its actual
    environment mapping must still reach the caller.
    """
    environment = _refuse_if_credential_cannot_be_enforced(VLLMAdapter(), "some-secret")
    assert environment == VLLMAdapter().inference_credential_env("some-secret")
    assert environment  # non-empty: vLLM does have a mechanism


def test_vllm_with_no_credential_returns_none() -> None:
    assert _refuse_if_credential_cannot_be_enforced(VLLMAdapter(), None) is None


def test_llamacpp_now_passes_the_guard_it_used_to_fail() -> None:
    """llama.cpp gained a verified mechanism (``LLAMA_API_KEY``), so the guard
    must let it through and hand back the real environment.

    This is the guard working, not weakening. It refuses an adapter that cannot
    enforce what was supplied; llama.cpp can now, verified against the running
    image's own ``--help``. SGLang is still refused above, so the case the
    guard exists for is still covered by a shipped runtime rather than only by
    a hypothetical one.
    """
    environment = _refuse_if_credential_cannot_be_enforced(LlamaCppAdapter(), "some-secret")

    assert environment == {"LLAMA_API_KEY": "some-secret"}


@pytest.mark.parametrize("adapter", [SGLangAdapter(), ExLlamaAdapter()])
def test_a_node_default_key_never_stops_a_runtime_that_cannot_use_it(adapter: object) -> None:
    """A node-wide key is a default, not a requirement.

    It used to be treated like a binding, so one key provisioned on every node
    by the installer made every runtime without a mechanism unstartable.
    """
    assert _refuse_if_credential_cannot_be_enforced(adapter, None, node_default="node-key") is None


def test_a_node_default_key_still_applies_where_the_runtime_can_enforce_it() -> None:
    environment = _refuse_if_credential_cannot_be_enforced(
        VLLMAdapter(), None, node_default="node-key"
    )
    assert environment == VLLMAdapter().inference_credential_env("node-key")


def test_a_bound_key_wins_over_the_node_default() -> None:
    environment = _refuse_if_credential_cannot_be_enforced(
        LlamaCppAdapter(), "bound-key", node_default="node-key"
    )
    assert environment == {"LLAMA_API_KEY": "bound-key"}


def test_a_bound_key_is_still_refused_where_it_cannot_be_enforced() -> None:
    """The operator asked for this deployment to be keyed; that is not optional."""
    with pytest.raises(HTTPException) as caught:
        _refuse_if_credential_cannot_be_enforced(
            SGLangAdapter(), "bound-key", node_default="node-key"
        )
    assert caught.value.detail["code"] == "credential_not_enforceable"  # type: ignore[index]
