"""Coordinator HTTP client for the MCP server.

The MCP server is an **HTTP client of the coordinator API**, exactly as the CLI
is. It imports no coordinator business logic, and this module is the only place
it talks to anything.

That indirection is the whole point of that arrangement. Because MCP reaches the
product the same way any other client does, it cannot become the architectural
foundation by accident: the coordinator has no dependency on it and runs
unchanged with MCP entirely absent.

Errors are returned as the coordinator's structured failure shape rather than
raised. An agent that receives ``{"code": "agent_unreachable", ...}`` can report
it; an agent that receives a transport exception tends to retry, which is the
stale-state behaviour this design exists to prevent.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_API = "http://127.0.0.1:8080"

# What a coordinator call can hand back: a record, a list of records, or the
# structured failure shape (which is itself a record). Declaring it as a union
# rather than ``Any`` is what lets the MCP SDK derive a schema for each tool,
# and that schema is what makes a one-element list distinguishable from a
# single record on the wire.
ApiResult = dict[str, Any] | list[dict[str, Any]]


def api_url() -> str:
    """The coordinator base URL (``TENSORSTEAD_API``)."""
    return os.environ.get("TENSORSTEAD_API", DEFAULT_API)


def management_token() -> str | None:
    """The management token, if one is configured (``TENSORSTEAD_MGMT_TOKEN``)."""
    return os.environ.get("TENSORSTEAD_MGMT_TOKEN")


def tls_verify() -> str | bool:
    """Trust anchor for the coordinator hop (``TENSORSTEAD_CA_BUNDLE``).

    The MCP service reaches the coordinator as any other HTTP client does, so
    when the coordinator serves TLS from the managed private CA this process
    needs that CA too — including over loopback, where the certificate is still
    presented and still verified.
    """
    value = os.environ.get("TENSORSTEAD_CA_BUNDLE", "").strip()
    return value or True


class CoordinatorClient:
    """A narrow HTTP client over the coordinator's management API.

    ``transport`` is injectable so tests drive the real coordinator app in
    process without a socket.
    """

    def __init__(
        self, *, base_url: str | None = None, token: str | None = None, client: Any = None
    ):
        self._base_url = base_url or api_url()
        self._token = token if token is not None else management_token()
        self._client = client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> ApiResult:
        """Call the coordinator and return parsed JSON, or a failure shape.

        Never raises for an API-level failure: the structured reason is the
        useful result for an agent, and turning it into an exception would only
        invite a retry loop around a condition that will not change.
        """
        try:
            response = self._http().request(
                method,
                path,
                json=json,
                params=params,
                headers=self._headers(),
            )
        except Exception as exc:  # transport-level only
            return {
                "code": "coordinator_unreachable",
                "message": f"could not reach the coordinator at {self._base_url}: {exc}",
                "detail": {},
            }
        if not response.content:
            return {"status": "ok"} if response.status_code < 400 else {}
        try:
            parsed: ApiResult = response.json()
        except ValueError:
            return {"code": "invalid_response", "message": response.text, "detail": {}}
        return parsed

    def _http(self) -> Any:
        if self._client is not None:
            return self._client
        import httpx

        return httpx.Client(base_url=self._base_url, timeout=60.0, verify=tls_verify())
