"""Role-distinct token authentication on the agent.

An agent holds substantial host authority (Docker socket, systemd units), so a
compromised agent is a plausible outcome. Role-distinct tokens mean such an
agent can still pull artifacts (replication token) but cannot issue management
instructions (management token), which a single shared token would have granted
it over every other node.

Two token roles:
- **management** — gates the management operation set (deploy, start, stop…).
- **replication** — gates the agent-to-agent artifact replication hop. A
  compromised agent holding only the replication token cannot manage anything.

When a role's token is unconfigured the corresponding surface is open, which is
the default for a loopback or externally-terminated deployment. TLS on the agent
hops is a separate, mandatory concern.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)

ROLE_MANAGEMENT = "management"
ROLE_REPLICATION = "replication"

AuthDependency = Callable[[HTTPAuthorizationCredentials | None], None]


class AgentAuth:
    """Carries the two role tokens and exposes the two FastAPI dependencies."""

    def __init__(self, management_token: str | None, replication_token: str | None) -> None:
        self._tokens = {
            ROLE_MANAGEMENT: management_token,
            ROLE_REPLICATION: replication_token,
        }

    def _require(self, role: str) -> AuthDependency:
        def dependency(
            credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
        ) -> None:
            token = self._tokens.get(role)
            if token is None:
                return  # this role is open (loopback permitted)
            if credentials is None or credentials.scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "unauthorized", "message": f"{role} token required"},
                )
            if not hmac.compare_digest(credentials.credentials, token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={"code": "unauthorized", "message": f"invalid {role} token"},
                )

        return dependency

    @property
    def require_management(self) -> AuthDependency:
        """FastAPI dependency gating management operations."""
        return self._require(ROLE_MANAGEMENT)

    @property
    def require_replication(self) -> AuthDependency:
        """FastAPI dependency gating the replication hop."""
        return self._require(ROLE_REPLICATION)
