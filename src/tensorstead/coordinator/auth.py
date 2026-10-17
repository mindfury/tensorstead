"""Coordinator client authentication and TLS configuration.

A management token gates every coordinator route. All
authenticated clients are equally trusted — there is no multi-tenancy and no
per-user permission model in v1.

The token is supplied via the ``Authorization: Bearer <token>`` header. When no
token is configured the coordinator runs open (loopback is permitted plain);
the FastAPI dependency simply passes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tensorstead.domain.tokens import tokens_match

_bearer = HTTPBearer(auto_error=False)


class CoordinatorAuth:
    """Carries the configured management token and exposes the FastAPI dependency."""

    def __init__(self, token: str | None) -> None:
        self._token = token

    def require_auth(
        self,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
        *,
        authorization_headers: int = 1,
    ) -> None:
        """FastAPI dependency: reject when a configured token is not matched.

        Uses constant-time comparison so a timing attack cannot recover the
        token from response latencies.
        """
        if self._token is None:
            return  # open coordinator (loopback plain is permitted)
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            # Two Authorization headers are ambiguous: refused, not resolved
            # by whichever one the parser happened to keep (CodeGuard CG-19).
            or authorization_headers != 1
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "unauthorized", "message": "management token required"},
            )
        # Bytes, so a non-ASCII value is a 401 rather than a TypeError.
        if not tokens_match(credentials.credentials, self._token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "unauthorized", "message": "invalid management token"},
            )
