"""Read-only release smoke checks for a configured Tensorstead appliance.

This module deliberately contains no lifecycle operation.  It proves that the
three management surfaces agree about an existing deployment and that the
runtime itself accepts an authenticated inference request.  Configuration is
read from files named by environment variables so secret values never need to
appear in a shell history, Makefile, or repository.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from tensorstead.mcp.client import CoordinatorClient
from tensorstead.mcp.server import build_server

# Content negotiation and protocol version for a raw Streamable HTTP probe.
# These are only needed for the unauthenticated checks, which deliberately run
# below the SDK so that a refusal is observed as an HTTP status rather than as
# an opaque client-side error.
MCP_ACCEPT = "application/json, text/event-stream"
MCP_PROTOCOL_VERSION = "2025-06-18"


class SmokeCheckError(RuntimeError):
    """A configured appliance did not satisfy a read-only release check."""


@dataclass(frozen=True)
class SmokeSettings:
    """Local-only configuration for one existing deployment."""

    api_url: str
    management_token_file: Path
    deployment_name: str
    expected_model: str
    expected_source_model: str
    expected_nodes: tuple[str, ...]
    inference_url: str
    # Optional: a deployment need not require a key. When one is named, the
    # smoke test also proves the endpoint refuses requests without it.
    inference_api_key_file: Path | None
    expected_reply: str
    mcp_url: str
    mcp_ca_file: Path

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> SmokeSettings:
        values = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = values.get(name, "").strip()
            if not value:
                raise SmokeCheckError(f"{name} must be set for make test-smoke")
            return value

        nodes = tuple(
            node.strip()
            for node in values.get("TENSORSTEAD_SMOKE_EXPECTED_NODES", "").split(",")
            if node.strip()
        )
        return cls(
            api_url=required("TENSORSTEAD_SMOKE_API").rstrip("/"),
            management_token_file=Path(required("TENSORSTEAD_SMOKE_MGMT_TOKEN_FILE")).expanduser(),
            deployment_name=values.get("TENSORSTEAD_SMOKE_DEPLOYMENT", "qwen36-27b"),
            expected_model=values.get("TENSORSTEAD_SMOKE_MODEL", "qwen36-27b"),
            expected_source_model=values.get(
                "TENSORSTEAD_SMOKE_SOURCE_MODEL", "nvidia/Qwen3.6-27B-NVFP4"
            ),
            expected_nodes=nodes,
            inference_url=required("TENSORSTEAD_SMOKE_INFERENCE_URL").rstrip("/"),
            inference_api_key_file=(
                Path(key_file).expanduser()
                if (key_file := values.get("TENSORSTEAD_SMOKE_INFERENCE_API_KEY_FILE", "").strip())
                else None
            ),
            expected_reply=values.get(
                "TENSORSTEAD_SMOKE_EXPECTED_REPLY", "Tensorstead smoke test passed."
            ),
            mcp_url=required("TENSORSTEAD_SMOKE_MCP_URL").rstrip("/"),
            mcp_ca_file=Path(required("TENSORSTEAD_SMOKE_MCP_CA_FILE")).expanduser(),
        )


def run(
    settings: SmokeSettings,
    *,
    client: httpx.Client | None = None,
    check_mcp_transport: bool = True,
) -> None:
    """Run API, MCP, and direct-inference checks without changing appliance state."""
    management_token = _read_secret(settings.management_token_file, "management token")
    inference_key = (
        _read_secret(settings.inference_api_key_file, "inference API key")
        if settings.inference_api_key_file is not None
        else None
    )
    owns_client = client is None
    # An https coordinator presents the managed private CA's certificate, which
    # is in no public trust store, so the API client needs the same anchor the
    # MCP check uses. Plain http installations are unaffected.
    api = client or httpx.Client(
        base_url=settings.api_url,
        timeout=30.0,
        verify=(
            str(settings.mcp_ca_file)
            if settings.api_url.startswith("https://") and settings.mcp_ca_file.is_file()
            else True
        ),
    )
    try:
        _check_api(api, management_token, settings)
        _check_mcp(api, management_token, settings)
        if check_mcp_transport:
            asyncio.run(_check_mcp_stdio(settings.api_url, management_token, settings))
            _check_mcp_streamable_http(management_token, settings)
        _check_inference(inference_key, settings)
    finally:
        if owns_client:
            api.close()


def _read_secret(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SmokeCheckError(f"cannot read {label} file {path}") from exc
    if not value:
        raise SmokeCheckError(f"{label} file {path} is empty")
    return value


def _check_api(api: httpx.Client, token: str, settings: SmokeSettings) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    nodes = _get_json(api, "/v1/nodes", headers=headers)
    if not isinstance(nodes, list):
        raise SmokeCheckError("coordinator returned an invalid node inventory")
    registered = {str(node.get("name")) for node in nodes if isinstance(node, dict)}
    missing = set(settings.expected_nodes) - registered
    if missing:
        raise SmokeCheckError(f"expected registered nodes are missing: {sorted(missing)}")

    models = _get_json(api, "/v1/models", headers=headers)
    _find_pinned_model(models, settings.expected_source_model)

    deployments = _get_json(api, "/v1/deployments", headers=headers)
    deployment = _find_deployment(deployments, settings.deployment_name)
    deployment_id = deployment["declared"]["id"]
    status = _get_json(api, f"/v1/deployments/{deployment_id}/status", headers=headers)
    observed = status.get("observed") if isinstance(status, dict) else None
    if not isinstance(observed, dict) or observed.get("status") != "running":
        raise SmokeCheckError(f"deployment {settings.deployment_name!r} is not running")


def _check_mcp(api: httpx.Client, token: str, settings: SmokeSettings) -> None:
    server = build_server(client=CoordinatorClient(client=api, token=token))
    mcp_nodes = _mcp_call(server, "node_list")
    if not isinstance(mcp_nodes, list):
        raise SmokeCheckError("MCP node_list returned an invalid inventory")
    registered = {str(node.get("name")) for node in mcp_nodes if isinstance(node, dict)}
    missing = set(settings.expected_nodes) - registered
    if missing:
        raise SmokeCheckError(f"MCP cannot see expected nodes: {sorted(missing)}")

    deployments = _mcp_call(server, "deployment_list")
    _find_deployment(deployments, settings.deployment_name)
    models = _mcp_call(server, "model_list")
    _find_pinned_model(models, settings.expected_source_model)


async def _check_mcp_stdio(api_url: str, token: str, settings: SmokeSettings) -> None:
    """Exercise the real MCP stdio protocol against the configured coordinator.

    The in-process check above proves tool behavior.  This check additionally
    proves that the packaged MCP server starts, completes its protocol
    handshake, and carries read-only tool calls over its standard client
    transport.  The token exists only in the child process environment and is
    never included in a command line or output.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    # A locally launched MCP server is an ordinary HTTP client of the
    # coordinator, so against a TLS coordinator it needs the managed private CA
    # exactly as any other client does. Without this the child
    # process fails verification and reports an empty inventory, which reads as
    # "the coordinator knows of no nodes" rather than "this client could not
    # verify it".
    child_env = {"TENSORSTEAD_API": api_url, "TENSORSTEAD_MGMT_TOKEN": token}
    if api_url.startswith("https://") and settings.mcp_ca_file.is_file():
        child_env["TENSORSTEAD_CA_BUNDLE"] = str(settings.mcp_ca_file)

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tensorstead.mcp.server"],
        env=child_env,
    )
    async with (
        stdio_client(server) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        nodes = _mcp_transport_items(await session.call_tool("node_list", {}))
        registered = {str(node.get("name")) for node in nodes if isinstance(node, dict)}
        missing = set(settings.expected_nodes) - registered
        if missing:
            raise SmokeCheckError(f"MCP stdio cannot see expected nodes: {sorted(missing)}")

        deployments = _mcp_transport_items(await session.call_tool("deployment_list", {}))
        _find_deployment(deployments, settings.deployment_name)
        models = _mcp_transport_items(await session.call_tool("model_list", {}))
        _find_pinned_model(models, settings.expected_source_model)


def _check_mcp_streamable_http(token: str, settings: SmokeSettings) -> None:
    """Exercise the hosted Streamable HTTP MCP endpoint without changing anything.

    The stdio check above proves the packaged server speaks MCP when a local
    client starts it.  This check proves the *deployed network service* is
    sound, which is a different claim: it is reached over TLS from another
    host, so its trust and authentication boundaries are what stand between a
    network caller and full management authority.

    Four properties are asserted, and the negative ones matter most:

    1. TLS verifies against the managed private CA.
    2. The endpoint fails closed against the public trust store — proving the
       private CA is genuinely required rather than incidentally satisfied.
    3. A missing token and a wrong token are both refused.
    4. An authenticated client completes a real ``initialize`` handshake, and
       ``tools/list`` returns exactly the supported management operation set.

    Property 4's tool count is compared against ``service.registry`` rather
    than a hard-coded number, which makes this a live parity check: a
    tool added to the catalogue but missing from the deployed endpoint fails
    here rather than being discovered by a person.
    """
    ca = _require_ca(settings.mcp_ca_file)
    _check_mcp_http_tls_and_auth(ca, settings)
    asyncio.run(_check_mcp_http_session(ca, token, settings))


def _require_ca(path: Path) -> ssl.SSLContext:
    if not path.is_file():
        raise SmokeCheckError(f"MCP private CA bundle {path} was not found")
    try:
        return ssl.create_default_context(cafile=str(path))
    except OSError as exc:
        raise SmokeCheckError(f"MCP private CA bundle {path} is not a usable certificate") from exc


def _check_mcp_http_tls_and_auth(ca: ssl.SSLContext, settings: SmokeSettings) -> None:
    """Assert the endpoint's TLS trust and its refusal of bad credentials."""
    handshake = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "tensorstead-smoke", "version": "0"},
        },
    }
    headers = {"Accept": MCP_ACCEPT, "Content-Type": "application/json"}

    # Public trust store must not be enough; the CA is private by design.
    try:
        with httpx.Client(timeout=30.0) as public:
            public.post(settings.mcp_url, headers=headers, json=handshake)
    except httpx.HTTPError:
        pass
    else:
        raise SmokeCheckError(
            "hosted MCP endpoint verified against the public trust store; "
            "its certificate should chain only to the managed private CA"
        )

    with httpx.Client(verify=ca, timeout=30.0) as client:
        for label, extra in (
            ("without a token", {}),
            (
                "with an incorrect token",
                {"Authorization": "Bearer tensorstead-smoke-invalid-token"},
            ),
        ):
            response = client.post(settings.mcp_url, headers={**headers, **extra}, json=handshake)
            if response.status_code not in (401, 403):
                raise SmokeCheckError(
                    f"hosted MCP endpoint answered HTTP {response.status_code} {label}; "
                    "it must refuse an unauthenticated caller"
                )


async def _check_mcp_http_session(ca: ssl.SSLContext, token: str, settings: SmokeSettings) -> None:
    """Complete a real authenticated MCP session over Streamable HTTP."""
    from mcp import ClientSession
    from mcp.client.streamable_http import httpx2, streamable_http_client

    expected_tools = set(_expected_tool_names())
    async with (
        httpx2.AsyncClient(
            verify=ca,
            timeout=60.0,
            headers={"Authorization": f"Bearer {token}"},
            follow_redirects=True,
        ) as http_client,
        streamable_http_client(settings.mcp_url, http_client=http_client) as streams,
    ):
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            published = {tool.name for tool in listed.tools}
            missing = expected_tools - published
            extra = published - expected_tools
            if missing or extra:
                # Naming the likely cause matters: the usual reason for missing
                # tools is that this build is newer than the estate, which
                # reads as a defect in the endpoint unless it is said plainly.
                cause = (
                    " — this build knows operations the deployed coordinator does not, "
                    "so the estate is probably running an older release; deploy it, "
                    "or compare `stead status`"
                    if missing and not extra
                    else ""
                )
                raise SmokeCheckError(
                    "hosted MCP tools do not match the supported management operation set "
                    f"(missing: {sorted(missing)}; unexpected: {sorted(extra)}){cause}"
                )

            nodes = _mcp_transport_items(await session.call_tool("node_list", {}))
            registered = {str(node.get("name")) for node in nodes if isinstance(node, dict)}
            absent = set(settings.expected_nodes) - registered
            if absent:
                raise SmokeCheckError(f"hosted MCP cannot see expected nodes: {sorted(absent)}")


def _expected_tool_names() -> list[str]:
    """Return the MCP tool name for every catalogued management operation."""
    from tensorstead.service.registry import all_operations

    return [f"{operation.noun}_{operation.verb}" for operation in all_operations()]


def _check_inference(inference_key: str | None, settings: SmokeSettings) -> None:
    """Prove the runtime is ready, allowing bounded post-restart warm-up.

    A listening vLLM socket can briefly reset requests while it loads weights.
    The coordinator's transport-level health observation is useful but does not
    mean that inference is ready yet.  Retrying the complete read-only probe
    prevents a just-restarted, otherwise healthy model from producing a flaky
    release result.
    """
    deadline = time.monotonic() + 120.0
    last_error: Exception | None = None
    while True:
        try:
            _check_inference_once(inference_key, settings)
            return
        except httpx.HTTPError as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                raise SmokeCheckError(
                    "inference runtime did not become ready within 120 seconds"
                ) from last_error
            time.sleep(2.0)


def _check_inference_once(inference_key: str | None, settings: SmokeSettings) -> None:
    with httpx.Client(base_url=settings.inference_url, timeout=60.0) as runtime:
        headers: dict[str, str] = {}
        if inference_key is not None:
            unauthenticated = runtime.get("/v1/models")
            if unauthenticated.status_code not in (401, 403):
                raise SmokeCheckError("inference endpoint accepted a request without an API key")
            headers = {"Authorization": f"Bearer {inference_key}"}
        models = _get_json(runtime, "/v1/models", headers=headers)
        available = {
            str(model.get("id")) for model in models.get("data", []) if isinstance(model, dict)
        }
        if settings.expected_model not in available:
            raise SmokeCheckError(
                f"inference endpoint does not advertise model {settings.expected_model!r}"
            )

        response = _post_json(
            runtime,
            "/v1/chat/completions",
            headers=headers,
            body={
                "model": settings.expected_model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reply with exactly: {settings.expected_reply}",
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        content = _chat_content(response)
        if content != settings.expected_reply:
            raise SmokeCheckError(f"inference reply was {content!r}, not the expected smoke reply")


def _get_json(api: httpx.Client, path: str, *, headers: dict[str, str]) -> Any:
    response = api.get(path, headers=headers)
    _require_success(response, path)
    return response.json()


def _post_json(
    api: httpx.Client, path: str, *, headers: dict[str, str], body: dict[str, Any]
) -> Any:
    response = api.post(path, headers=headers, json=body)
    _require_success(response, path)
    return response.json()


def _require_success(response: httpx.Response, path: str) -> None:
    if response.is_success:
        return
    raise SmokeCheckError(f"{path} returned HTTP {response.status_code}")


def _find_deployment(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise SmokeCheckError("coordinator returned an invalid deployment inventory")
    for deployment in payload:
        if isinstance(deployment, dict) and deployment.get("declared", {}).get("name") == name:
            return deployment
    raise SmokeCheckError(f"deployment {name!r} was not found")


def _find_pinned_model(payload: Any, source_model_id: str) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise SmokeCheckError("coordinator returned an invalid model inventory")
    for model in payload:
        if isinstance(model, dict) and model.get("source_model_id") == source_model_id:
            if model.get("revision_pinned") is not True or model.get("resolved_revision") == "main":
                raise SmokeCheckError(
                    f"model {source_model_id!r} is not recorded at an immutable revision"
                )
            return model
    raise SmokeCheckError(f"model {source_model_id!r} was not found")


def _mcp_call(server: Any, tool: str) -> Any:
    result = asyncio.run(server.call_tool(tool, {}))
    if result.is_error:
        raise SmokeCheckError(f"MCP tool {tool} returned an error")
    structured = result.structured_content
    if structured is None:
        return json.loads(result.content[0].text)
    return structured.get("result", structured)


def _mcp_transport_items(result: Any) -> list[dict[str, Any]]:
    """Normalise an MCP list-tool result across SDK serialisation shapes."""
    if getattr(result, "isError", False):
        raise SmokeCheckError("MCP stdio tool returned an error")
    content = getattr(result, "content", [])
    if content:
        items: list[dict[str, Any]] = []
        for block in content:
            parsed = json.loads(block.text)
            value = parsed.get("result", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                items.append(value)
            else:
                raise SmokeCheckError("MCP stdio list tool returned an invalid item")
        return items
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        value = structured.get("result", structured)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
        if isinstance(value, dict):
            return [value]
    raise SmokeCheckError("MCP stdio tool returned no content")


def _chat_content(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SmokeCheckError("inference response did not contain assistant content") from exc
    if not isinstance(content, str):
        raise SmokeCheckError("inference response did not contain text content")
    return content
