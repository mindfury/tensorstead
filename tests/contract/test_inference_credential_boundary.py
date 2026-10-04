"""The inference credential's boundary.

The key that authenticates inference clients to vLLM is provisioned by Ansible
and injected by the agent. That ownership is deliberate: it is a *data plane*
credential, and this product stays out of the data plane.

Two things went wrong around it, and these tests hold both fixed:

1. The mechanism lived in the agent's runtime-agnostic deployment route as
   ``if runtime_type == "vllm"``, which is where the next runtime's variant
   would have been added. Runtime-specific behaviour belongs in the
   adapter.
2. Nothing reported *that* an endpoint was authenticated, so the vLLM/llama.cpp
   asymmetry was invisible and an unauthenticated endpoint looked identical to
   an authenticated one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"


def test_the_runtime_specific_variable_lives_in_the_adapter() -> None:
    """The mechanism is the adapter's knowledge, not the route's."""
    assert VLLMAdapter().inference_credential_env("SECRET") == {"VLLM_API_KEY": "SECRET"}


def test_an_adapter_without_a_mechanism_supplies_nothing() -> None:
    """An empty mapping means the runtime serves unauthenticated.

    SGLang now carries this case. llama.cpp used to, and the move is the point:
    "no mechanism" was always a statement about what had been *verified*, not a
    permanent property of the runtime, and llama.cpp's was verified (below).
    """
    assert SGLangAdapter().inference_credential_env("SECRET") == {}


def test_llamacpp_supplies_its_verified_environment_variable() -> None:
    """llama.cpp reads ``LLAMA_API_KEY``, so it can enforce a credential.

    Verified against the image this estate actually runs
    (``local/llamacpp-qwen38-flash-next:pr27742-eaf9376``) rather than assumed,
    which is the bar for adding a mechanism at all::

        --api-key KEY   API key to use for authentication, ...
                        (env: LLAMA_API_KEY)

    The environment, never ``--api-key`` on argv: llama-server accepts both,
    and the adapter's own ``_FORBIDDEN_EXTRA_ARGS`` has always refused the argv
    spelling because it puts the secret where any local user reads it from
    ``ps``. That entry was written before this mechanism existed; it is now
    true rather than merely aspirational.
    """
    assert LlamaCppAdapter().inference_credential_env("SECRET") == {"LLAMA_API_KEY": "SECRET"}


def test_the_credential_never_reaches_the_command_line(tmp_path: Path) -> None:
    """The whole reason the mechanism is environment-based."""
    (tmp_path / "model.gguf").write_bytes(b"GGUF\x00")
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))

    assert "--api-key" not in args
    assert not any("SECRET" in arg for arg in args)


def test_the_agent_route_does_not_name_a_runtime_or_its_variable() -> None:
    """The generic path must not regain runtime-specific branches."""
    route = (_SRC / "agent/routes/deployments.py").read_text(encoding="utf-8")

    assert "VLLM_API_KEY" not in route, "the variable name belongs to the adapter"
    assert 'runtime_type == "vllm"' not in route, (
        "a runtime-specific branch here is where the next runtime gets bolted on"
    )
    assert "inference_credential_env" in route, "the route must ask the adapter"
    # Matched without the argument list on purpose. The rule is that the
    # generic route *asks* the adapter rather than deciding itself; pinning the
    # exact call shape made a legitimate signature change look like a violation
    # (the adapter now needs the config to know whether the image is
    # self-starting).
    assert "entrypoint=runtime_adapter.container_entrypoint(" in route, (
        "the container entrypoint is runtime-specific and belongs to the adapter too"
    )


def test_the_credential_value_is_never_persisted_or_exported() -> None:
    """The product carries the fact, never the secret.

    Checked structurally: no persistence, contract, or export module may name
    the credential's environment variable or a field to hold it.
    """
    offenders: list[str] = []
    for area in ("adapters/sqlite", "service/export.py", "domain/models.py"):
        target = _SRC / area
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for token in ("VLLM_API_KEY", "TENSORSTEAD_INFERENCE_API_KEY", "inference_api_key"):
                if token in text:
                    offenders.append(f"{path.name}:{token}")

    assert offenders == [], f"the inference credential must not reach: {offenders}"


def test_observed_state_reports_authentication_as_a_boolean() -> None:
    """The fact is reportable; the value is not present to report."""
    from tensorstead.contracts.agent import ObservedStateResponse

    fields = ObservedStateResponse.model_fields
    assert "endpoint_authenticated" in fields
    # None is a real answer: an agent too old to say is not "unauthenticated".
    assert fields["endpoint_authenticated"].default is None

    from tensorstead.contracts.api import DeploymentObservedPerNode

    assert "endpoint_authenticated" in DeploymentObservedPerNode.model_fields


def test_adding_the_field_was_a_minor_contract_bump() -> None:
    """The compatibility policy — an additive optional field is MINOR, and must be
    recorded."""
    from tensorstead.contracts.version import CONTRACT_VERSION, CONTRACT_VERSION_MAJOR

    assert CONTRACT_VERSION_MAJOR == 1
    assert CONTRACT_VERSION != "1.0", (
        "the observed-state contract gained a field; the minor version must move"
    )


# --------------------------------------------------------------- ownership
# The design was revised: the product owns this setting because it owns every
# other setting a deployment has. What must not change is where the *value*
# may appear.


def test_the_schema_cannot_hold_a_credential_value() -> None:
    """The store holds a reference; no column can carry a secret."""
    schema = (_SRC / "adapters/sqlite/migrations/0002_inference_credentials.sql").read_text(
        encoding="utf-8"
    )

    assert "secret_ref" in schema
    for forbidden in ("secret_value", "api_key TEXT", "value TEXT"):
        assert forbidden not in schema, f"schema must not be able to hold a value: {forbidden}"


def test_listing_credentials_returns_names_never_values() -> None:
    """There is no read path, for either kind of credential."""
    import inspect

    from tensorstead.service.credentials import InferenceCredentialService

    source = inspect.getsource(InferenceCredentialService.list)
    assert "secret_ref" not in source
    assert "resolve" not in source, "listing must not resolve a value"


def test_the_credential_travels_per_request_and_is_never_stored_agent_side() -> None:
    """Resolved late, carried once, written nowhere."""
    lifecycle = (_SRC / "service/lifecycle.py").read_text(encoding="utf-8")
    agent_route = (_SRC / "agent/routes/deployments.py").read_text(encoding="utf-8")

    assert '"inference_credential"' in lifecycle, "the coordinator must pass it per request"
    assert "if credential is not None" in lifecycle, (
        "absent must mean the key is omitted entirely, not sent as null"
    )
    # The agent uses it and does not persist it.
    assert "payload.inference_credential" in agent_route
    for persistence in ("save_", "write_text", "open(", "json.dump"):
        assert f"{persistence}inference" not in agent_route


def test_the_node_provisioned_key_remains_a_fallback() -> None:
    """Deployments predating product ownership must keep working.

    The bound value and the node's key travel separately, because they are
    not the same request: a bound key that cannot be enforced refuses the
    start, while the node's key is only a default for runtimes that can use it.
    """
    agent_route = (_SRC / "agent/routes/deployments.py").read_text(encoding="utf-8")

    assert "TENSORSTEAD_INFERENCE_API_KEY" in agent_route
    assert 'node_default=os.environ.get("TENSORSTEAD_INFERENCE_API_KEY")' in agent_route, (
        "the supplied value must win, with the node's provisioning as fallback"
    )


def test_deleting_a_bound_credential_is_refused() -> None:
    """Same referential-integrity rule as for models and images."""
    import inspect

    from tensorstead.service.credentials import InferenceCredentialService

    source = inspect.getsource(InferenceCredentialService.delete)
    assert "StillReferencedError" in source
    assert "referrers" in source
