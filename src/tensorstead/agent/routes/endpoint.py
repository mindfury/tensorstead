"""Agent endpoint-check route — ``GET /agent/v1/endpoint-check``.

A read-only **advisory** pre-flight check of whether a port is already bound on
the node. It catches conflicts the managed-vs-managed
check cannot see — including a port taken by a process the product does not
manage.

It is explicitly **not a guarantee**: a port free at check time can be
taken before start, so the real enforcement remains the runtime's own bind
failure surfacing through the failed start operation. Read-only and
on-demand; introduces no polling.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from tensorstead.agent.app import require_management

router = APIRouter(prefix="/agent/v1", tags=["endpoint"])


class EndpointCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int
    bound: bool
    observed_at: datetime


@router.get(
    "/endpoint-check",
    response_model=EndpointCheckResponse,
    dependencies=[Depends(require_management)],
)
def endpoint_check(request: Request, port: int) -> EndpointCheckResponse:
    """Advisory read-only check of whether ``port`` is already bound."""
    bound = _port_bound(port)
    return EndpointCheckResponse(
        port=port,
        bound=bound,
        observed_at=datetime.now().astimezone(),
    )


def _port_bound(port: int) -> bool:
    """Return whether ``port`` is currently bound on the node.

    Uses a non-blocking socket connect to 127.0.0.1 — a pure read with no
    side effect. A refused connection means the port is free; a success or a
    timeout means something is listening (or the probe itself failed, which we
    treat as bound-unavailable so the pre-flight errs toward caution).
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
