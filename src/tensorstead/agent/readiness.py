"""Endpoint readiness probing.

Three separate facts, deliberately not collapsed into one boolean:

- **container running** — the process exists (``ContainerState``, not here);
- **endpoint reachable** — something accepts a TCP connection on the port;
- **inference ready** — the runtime answers its own API as a serving model.

The 2026-08-10 incident sat in the gap between the second and the third. A
single ``healthy`` boolean cannot express "the listener is up but the model is
not serving", and an operator who cannot see that distinction cannot act on it.

Probing is a management-plane read *about* the data plane,
not participation in it: inference clients still address the runtime directly
and nothing here proxies, routes, or translates their traffic. What this module
must never become is a path that client requests traverse.

Every probe is on-demand, issued only while serving an
observation request. Nothing here samples, caches, or runs on a timer.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

from tensorstead.ports.runtime_adapter import ReadinessProbe

__all__ = ["EndpointReadiness", "ReadinessProbe", "endpoint_port", "probe_endpoint"]

# A hung endpoint must not hang the coordinator's observation request. These are
# deliberately short: the question is "is it serving right now", and a runtime
# that needs more than two seconds to list its own models is not serving.
_CONNECT_TIMEOUT_SECONDS = 0.5
_HTTP_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class EndpointReadiness:
    """The result of probing one deployment's endpoint.

    ``None`` never means "no". It means the probe could not establish the fact,
    which is a different thing an operator must be able to tell apart —
    the same rule applies to unreachable nodes.
    """

    reachable: bool | None = None
    inference_ready: bool | None = None
    detail: str | None = None


def endpoint_port(endpoint: str) -> int | None:
    """Extract the TCP port from a declared ``host:port`` endpoint."""
    try:
        parsed = urlsplit(f"//{endpoint}")
        return parsed.port
    except ValueError:
        return None


def probe_endpoint(
    endpoint: str | None,
    *,
    probe: ReadinessProbe | None = None,
    api_key: str | None = None,
) -> EndpointReadiness:
    """Probe ``endpoint`` for transport reachability and inference readiness.

    Probes ``127.0.0.1`` rather than the declared host. A deployment's endpoint
    is routinely declared as ``0.0.0.0:8000`` — a bind address, not a
    destination — and the agent runs on the node that would serve it, so the
    loopback answer is both the correct one and the one that cannot be confused
    by DNS or routing between nodes.
    """
    if not endpoint:
        return EndpointReadiness(detail="no endpoint recorded for this deployment")

    port = endpoint_port(endpoint)
    if port is None:
        return EndpointReadiness(detail=f"endpoint is not host:port: {endpoint!r}")

    if not _tcp_connectable(port):
        return EndpointReadiness(
            reachable=False,
            inference_ready=False,
            detail=f"nothing accepting connections on port {port}",
        )

    if probe is None:
        # The listener is up and the adapter declares no way to ask the runtime
        # whether it is serving. Reporting readiness unknown is the honest
        # answer; reporting it ready would recreate the defect one layer up.
        return EndpointReadiness(
            reachable=True,
            inference_ready=None,
            detail="runtime declares no readiness probe",
        )

    return _http_probe(port, probe, api_key)


def _tcp_connectable(port: int) -> bool:
    """Whether a TCP connection to ``port`` on loopback is accepted."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _http_probe(port: int, probe: ReadinessProbe, api_key: str | None) -> EndpointReadiness:
    """Ask the runtime's own API whether it is serving."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = f"http://127.0.0.1:{port}{probe.path}"

    try:
        response = httpx.get(url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS)
    except Exception as exc:
        # A connection accepted and then reset is exactly what the incident
        # showed. It is emphatically not "ready", and naming it beats a bare
        # false: the operator learns the listener is present but broken.
        return EndpointReadiness(
            reachable=True,
            inference_ready=False,
            detail=f"{probe.path} did not complete: {type(exc).__name__}",
        )

    if response.status_code == probe.expect_status:
        return EndpointReadiness(reachable=True, inference_ready=True, detail=None)

    if response.status_code in (401, 403):
        # We cannot see past the runtime's own auth. "Serving but we may not
        # ask" is not the same as "not serving", and collapsing the two would
        # report a healthy deployment as broken every time the agent lacks the
        # inference credential.
        return EndpointReadiness(
            reachable=True,
            inference_ready=None,
            detail=(
                f"{probe.path} returned {response.status_code}; "
                "the agent holds no credential this runtime accepts"
            ),
        )

    return EndpointReadiness(
        reachable=True,
        inference_ready=False,
        detail=f"{probe.path} returned {response.status_code}",
    )
