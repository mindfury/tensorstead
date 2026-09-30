"""MCP server.

Its own process, and an HTTP client of the coordinator API. It imports no
coordinator business logic, which is what makes the rule that "MCP is an
integration surface, not the architectural foundation" a structural fact: the
coordinator has no dependency on this module and runs unchanged with MCP
entirely absent.

**Both standard transports come from one implementation**. stdio and
Streamable HTTP are selected by configuration and expose the same tools built
by the same ``register_tools`` call. Transport is a deployment choice, never a
difference in capability — and because a single implementation backs both, they
cannot drift apart, so the same-surface guarantee holds by construction rather
than by test.

Run it:

    tensorstead-mcp                                  # stdio (default)
    TENSORSTEAD_MCP_TRANSPORT=streamable-http tensorstead-mcp
"""

from __future__ import annotations

import hmac
import os
from typing import Any

from tensorstead.mcp.client import CoordinatorClient
from tensorstead.mcp.tools import register_tools

SERVER_NAME = "tensorstead"

# The transports the official SDK offers and this server supports. Both expose
# the identical tool set.
TRANSPORTS = ("stdio", "streamable-http")

DEFAULT_TRANSPORT = "stdio"


def _http_settings() -> tuple[str, int, str, str, str]:
    """Read managed Streamable-HTTP listener settings from the environment."""
    host = os.environ.get("TENSORSTEAD_MCP_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("TENSORSTEAD_MCP_PORT", "8090"))
    path = os.environ.get("TENSORSTEAD_MCP_PATH", "/mcp")
    if not path.startswith("/"):
        raise ValueError("TENSORSTEAD_MCP_PATH must start with '/'")
    cert = os.environ.get("TENSORSTEAD_MCP_TLS_CERT", "")
    key = os.environ.get("TENSORSTEAD_MCP_TLS_KEY", "")
    if not cert or not key:
        raise RuntimeError(
            "Streamable HTTP requires TENSORSTEAD_MCP_TLS_CERT and TENSORSTEAD_MCP_TLS_KEY"
        )
    return host, port, path, cert, key


class _BearerAuth:
    """Network MCP callers must hold the management token."""

    def __init__(self, app: Any, token: str) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            supplied = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin-1")
            if not hmac.compare_digest(supplied, f"Bearer {self.token}"):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"detail":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def transport_from_env() -> str:
    """Select the transport from ``TENSORSTEAD_MCP_TRANSPORT`` (default stdio)."""
    transport = os.environ.get("TENSORSTEAD_MCP_TRANSPORT", DEFAULT_TRANSPORT)
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unsupported MCP transport {transport!r}; choose one of {', '.join(TRANSPORTS)}"
        )
    return transport


def build_server(*, client: CoordinatorClient | None = None) -> Any:
    """Build the MCP server with every management tool registered.

    ``client`` is injectable so tests drive a real coordinator app in process
    The returned object is the SDK's ``MCPServer``.
    """
    from mcp.server import MCPServer

    server = MCPServer(
        name=SERVER_NAME,
        instructions=(
            "Manage inference deployments: register nodes, acquire models, "
            "define and run deployments, and inspect declared versus observed "
            "state. Declared and observed are always reported separately and "
            "never merged. 'unknown' and 'unreachable' are ordinary values, not "
            "errors — report them rather than retrying. Long-running tools "
            "return an operation_id; poll operation_get for the outcome. There "
            "is no shell, exec, file, or SSH tool, by design."
        ),
    )
    register_tools(server, client or CoordinatorClient())
    return server


def main() -> None:
    """Console entry point: build the server and run the selected transport."""
    server = build_server()
    transport = transport_from_env()
    if transport != "streamable-http":
        server.run(transport=transport)
        return
    token = os.environ.get("TENSORSTEAD_MGMT_TOKEN")
    if not token:
        raise RuntimeError("Streamable HTTP requires TENSORSTEAD_MGMT_TOKEN")
    host, port, path, cert, key = _http_settings()
    import uvicorn

    app = _BearerAuth(server.streamable_http_app(streamable_http_path=path, host=host), token)
    uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key)


if __name__ == "__main__":
    main()
