from __future__ import annotations

import json
import ssl
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from tests.smoke.runner import (
    SmokeCheckError,
    SmokeSettings,
    _check_mcp_http_tls_and_auth,
    _expected_tool_names,
    _mcp_transport_items,
    _require_ca,
    run,
)

pytestmark = pytest.mark.unit


def _settings(tmp_path: Path) -> SmokeSettings:
    management = tmp_path / "management-token"
    inference = tmp_path / "inference-key"
    management.write_text("management-secret\n")
    inference.write_text("inference-secret\n")
    return SmokeSettings(
        api_url="http://coordinator.test",
        management_token_file=management,
        deployment_name="qwen36-27b",
        expected_model="qwen36-27b",
        expected_source_model="nvidia/Qwen3.6-27B-NVFP4",
        expected_nodes=("spark-alpha.internal", "spark-beta.internal"),
        inference_url="http://runtime.test",
        inference_api_key_file=inference,
        expected_reply="Tensorstead smoke test passed.",
        mcp_url="https://coordinator.test:8090/mcp",
        mcp_ca_file=tmp_path / "ca.pem",
    )


def _api_response(request: httpx.Request) -> httpx.Response:
    assert request.headers["authorization"] == "Bearer management-secret"
    if request.url.path == "/v1/nodes":
        return httpx.Response(
            200,
            json=[
                {"id": "node-a", "name": "spark-alpha.internal"},
                {"id": "node-b", "name": "spark-beta.internal"},
            ],
        )
    if request.url.path == "/v1/deployments":
        return httpx.Response(
            200,
            json=[{"declared": {"id": "deployment-a", "name": "qwen36-27b"}}],
        )
    if request.url.path == "/v1/models":
        return httpx.Response(
            200,
            json=[
                {
                    "source_model_id": "nvidia/Qwen3.6-27B-NVFP4",
                    "resolved_revision": "e1f2a3b4c5d6e7f8090a1b2c3d4e5f60718293ab",
                    "revision_pinned": True,
                }
            ],
        )
    if request.url.path == "/v1/deployments/deployment-a/status":
        return httpx.Response(200, json={"observed": {"status": "running"}})
    raise AssertionError(f"unexpected API request: {request.method} {request.url}")


def _inference_response(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/models":
        if "authorization" not in request.headers:
            return httpx.Response(401)
        assert request.headers["authorization"] == "Bearer inference-secret"
        return httpx.Response(200, json={"data": [{"id": "qwen36-27b"}]})
    if request.url.path == "/v1/chat/completions":
        assert request.headers["authorization"] == "Bearer inference-secret"
        body: dict[str, Any] = json.loads(request.content)
        assert body["model"] == "qwen36-27b"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Tensorstead smoke test passed."}}]},
        )
    raise AssertionError(f"unexpected inference request: {request.method} {request.url}")


def test_smoke_checks_api_mcp_and_authenticated_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = httpx.Client(
        base_url="http://coordinator.test", transport=httpx.MockTransport(_api_response)
    )
    inference = httpx.Client(
        base_url="http://runtime.test", transport=httpx.MockTransport(_inference_response)
    )
    monkeypatch.setattr("tests.smoke.runner.httpx.Client", lambda **_kwargs: inference)

    run(_settings(tmp_path), client=api, check_mcp_transport=False)


def test_smoke_requires_an_inference_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_client = httpx.Client
    api = httpx.Client(
        base_url="http://coordinator.test", transport=httpx.MockTransport(_api_response)
    )

    def runtime_without_auth(**_kwargs: Any) -> httpx.Client:
        return real_client(
            base_url="http://runtime.test",
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})),
        )

    monkeypatch.setattr("tests.smoke.runner.httpx.Client", runtime_without_auth)

    with pytest.raises(SmokeCheckError, match="without an API key"):
        run(_settings(tmp_path), client=api, check_mcp_transport=False)


def _hosted_mcp_clients(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public: Callable[[httpx.Request], httpx.Response],
    private: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Route the hosted-MCP probe's two clients to separate mock transports.

    ``_check_mcp_http_tls_and_auth`` distinguishes them by the ``verify``
    keyword: the private-CA client passes one, the public-trust-store client
    does not.
    """
    real_client = httpx.Client

    def factory(**kwargs: Any) -> httpx.Client:
        handler = private if "verify" in kwargs else public
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("tests.smoke.runner.httpx.Client", factory)


def _tls_rejected(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("certificate verify failed: unable to get local issuer certificate")


def test_hosted_mcp_must_not_verify_against_the_public_trust_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publicly trusted certificate means the private CA is not load-bearing."""
    _hosted_mcp_clients(
        monkeypatch,
        public=lambda _request: httpx.Response(401),
        private=lambda _request: httpx.Response(401),
    )

    with pytest.raises(SmokeCheckError, match="public trust store"):
        _check_mcp_http_tls_and_auth(ssl.create_default_context(), _settings(tmp_path))


@pytest.mark.parametrize("accepted_status", [200, 202])
def test_hosted_mcp_must_refuse_an_unauthenticated_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted_status: int
) -> None:
    """Anything but a refusal hands management authority to a network caller."""
    _hosted_mcp_clients(
        monkeypatch,
        public=_tls_rejected,
        private=lambda _request: httpx.Response(accepted_status),
    )

    with pytest.raises(SmokeCheckError, match="refuse an unauthenticated caller"):
        _check_mcp_http_tls_and_auth(ssl.create_default_context(), _settings(tmp_path))


def test_hosted_mcp_passes_when_tls_is_private_and_bad_credentials_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str | None] = []

    def private(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(401)

    _hosted_mcp_clients(monkeypatch, public=_tls_rejected, private=private)

    _check_mcp_http_tls_and_auth(ssl.create_default_context(), _settings(tmp_path))

    # Both the missing-token and wrong-token cases must actually be exercised.
    assert seen == [None, "Bearer tensorstead-smoke-invalid-token"]


def test_hosted_mcp_tool_names_match_the_operation_catalogue() -> None:
    """The expected tool set is derived, never hand-maintained."""
    from tensorstead.service.registry import operation_ids

    assert _expected_tool_names() == [op.replace(".", "_") for op in operation_ids()]


def test_missing_private_ca_bundle_is_reported_before_any_request(tmp_path: Path) -> None:
    with pytest.raises(SmokeCheckError, match="private CA bundle"):
        _require_ca(tmp_path / "absent.pem")


def test_mcp_stdio_list_items_are_collected_from_multiple_content_blocks() -> None:
    """The SDK serializes a list-tool result as one content block per item."""
    result = SimpleNamespace(
        isError=False,
        content=[
            SimpleNamespace(text=json.dumps({"id": "node-a", "name": "spark-a"})),
            SimpleNamespace(text=json.dumps({"id": "node-b", "name": "spark-b"})),
        ],
        structuredContent=None,
    )

    assert _mcp_transport_items(result) == [
        {"id": "node-a", "name": "spark-a"},
        {"id": "node-b", "name": "spark-b"},
    ]
