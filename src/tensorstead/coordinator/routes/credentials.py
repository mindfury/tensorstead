"""Credential routes.

A thin projection of ``service/credentials.py``, holding no management
logic of its own. Three routes, and the shape of the set is the point:

- ``PUT /v1/credentials/{source_id}/{name}`` accepts a value **or** a reference,
  routing either to the provider's protected store.
- ``GET /v1/credentials`` returns references — source, name, default flag, and
  when it was set.
- ``DELETE /v1/credentials/{source_id}/{name}`` reports what it did to the
  source's default.

**There is no fourth route.** No endpoint returns a secret value, because none
exists to write — that absence is what makes "0 occurrences of secret
material in any output" a property of the surface rather than something a
reviewer has to keep checking.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status

from tensorstead.contracts.api import (
    CredentialDeleteResponse,
    CredentialRef,
    CredentialSetRequest,
)
from tensorstead.coordinator.routes import require_auth_dep

router = APIRouter(prefix="/v1", tags=["credentials"])


def _credentials_service(request: Request) -> Any:
    return request.app.state.credentials_service


@router.put(
    "/credentials/{source_id}/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_auth_dep)],
)
def set_credential(
    source_id: str,
    name: str,
    payload: CredentialSetRequest,
    request: Request,
) -> None:
    """Set a named credential from a value or a reference.

    Returns ``204``: there is nothing to render back, and rendering the record
    would invite a future field that carries the value.
    """
    _credentials_service(request).set(
        source_id,
        name,
        secret=payload.secret,
        from_env=payload.from_env,
        from_file=payload.from_file,
        default=payload.default,
    )


@router.get(
    "/credentials",
    response_model=list[CredentialRef],
    dependencies=[Depends(require_auth_dep)],
)
def list_credentials(request: Request) -> list[CredentialRef]:
    """List credential references — names and default status only."""
    return [
        CredentialRef(
            source_id=credential.source_id,
            name=credential.name,
            is_default=credential.is_default,
            set_at=credential.set_at,
        )
        for credential in _credentials_service(request).list()
    ]


@router.delete(
    "/credentials/{source_id}/{name}",
    response_model=CredentialDeleteResponse,
    dependencies=[Depends(require_auth_dep)],
)
def delete_credential(
    source_id: str,
    name: str,
    request: Request,
    promote: str | None = None,
) -> CredentialDeleteResponse:
    """Delete a credential; report the consequence for the default.

    ``promote`` names a successor to take the default. Without it, deleting the
    default leaves the source with none — reported, never silently reassigned.
    """
    result = _credentials_service(request).delete(source_id, name, promote=promote)
    return CredentialDeleteResponse(**result)
