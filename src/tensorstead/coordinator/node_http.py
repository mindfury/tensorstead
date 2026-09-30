"""httpx node client.

The coordinator's implementation of the node-client port over httpx. Every
coordinator→agent call:

- is TLS-protected and carries a **management token**;
- verifies the agent's certificate against the **managed private CA** that
  Ansible operates, via the ``verify`` argument;

  **Certificate pinning is not implemented.** The design specifies pinning the
  agent's fingerprint at registration as defence in depth against a
  mis-issued certificate from that CA. ``Node.agent_cert_fingerprint`` exists
  and is persisted, but nothing captures a value for it and nothing compares
  one. A method named ``_pin_fingerprint`` previously stood here, was called on
  every request, and only read the field and discarded it — so the control read
  as present while being absent. It has been removed rather than left to imply
  otherwise. See the design notes for the recorded gap;
- performs **version negotiation on every call** via the
  ``X-Contract-Version`` header the agent returns in-band, raising
  ``CompatibilityRefusal`` on a mismatch before the call is treated as valid;
- surfaces the structured failure shape.

``httpx`` is used so the calls are first-class HTTP with streaming and TLS
verification, not a hand-rolled socket client.
"""

from __future__ import annotations

from typing import Any

import httpx

from tensorstead.agent.app import AGENT_VERSION_HEADER
from tensorstead.contracts.version import (
    CODE_AGENT_VERSION_INCOMPATIBLE,
    CONTRACT_VERSION,
    CompatibilityRefusal,
    parse,
)
from tensorstead.domain.models import Node
from tensorstead.ports.node_client import AgentCallError

# How long the coordinator waits on one agent hop before giving up.
#
# **A build hop is not a request.** It holds the connection open for the whole
# compile, so this bound is a claim about the slowest recorded spec anyone will
# build -- and 1800s was a judgement rather than a measurement, as was
# the archive client's own 1800s.
#
# It was measured on 2026-09-05 and found wrong. The official Flash-Next runtime
# (vLLM 0.28.1rc1 from source, MAX_JOBS=2 on a GB10) died at **1800.129s** --
# the bound, to a tenth of a second -- and the operation recorded only "The read
# operation timed out" with an empty detail. The image then had to
# be built out of band and imported, so the provenance of the estate's own
# candidate runs through a tar file instead of the recorded spec the coordinator
# holds. That is the failure this product exists to prevent, caused by a number.
#
# Four hours, and it is worth being exact about what that is and is not. The
# only *measured* fact is a lower bound: this compile ran past 1800s without
# finishing. Nothing establishes how long it actually needs -- the operation died
# at the hop and recorded no per-node outcome, so even whether the node-side
# build survived is unknown. An earlier revision explained a fast
# 2026-09-02 build as having had a prebuilt wheel and then withdrew it as
# unevidenced; the same caution applies here, and I had written "comfortably
# above a full from-source compile" before noticing I was making the same
# unsupported claim one layer down.
#
# So: four hours is generous headroom over the one number anybody measured, not a
# fitted bound. It is still a judgement. Which is precisely why the *other* half
# of this change matters more -- a compile that outruns even this now fails with
# the bound named and the node's state described, instead of as a bare string.
# The number can be wrong and the record will still be usable.
#
# The real fix is to make the agent-side build accepted-then-polled, the way
# the *coordinator's* own submission path already is. That changes the agent
# contract, so it is not this change.
_BUILD_TIMEOUT_SECONDS = 14400.0

# Moving a multi-gigabyte archive between peers. Unchanged; it was already
# measured against a real transfer.
_DISTRIBUTE_TIMEOUT_SECONDS = 3600.0


class NodeHTTPClient:
    """Drive one node agent over httpx with TLS, pinning, and negotiation.

    Every agent has historically held the same fleet-wide ``management_token``
    -- a compromised node discloses the credential that controls every other
    node too. ``Node.agent_management_token_ref``
    is the escape hatch: when a node has been issued its own credential (via
    ``NodeService.rotate_agent_management_token``), this resolves and
    presents *that* value to *that* node instead of the fleet-wide one.
    Resolved fresh on every call and held nowhere (the same pattern
    ``LifecycleService`` uses for the inference credential), not cached on
    the client -- a rotation must take effect on the very next call, not
    whenever this client happens to be reconstructed.
    """

    def __init__(
        self,
        *,
        management_token: str,
        verify: bool | str = True,
        timeout: float = 30.0,
        credential_provider: Any = None,
    ) -> None:
        self._token = management_token
        self._verify = verify
        self._timeout = timeout
        self._credential_provider = credential_provider

    def _headers(self, node: Node) -> dict[str, str]:
        token = self._token
        if node.agent_management_token_ref and self._credential_provider is not None:
            token = self._credential_provider.resolve(node.agent_management_token_ref)
        return {"Authorization": f"Bearer {token}"}

    def _call(
        self,
        node: Node,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{node.agent_endpoint.rstrip('/')}{path}"
        bound = timeout or self._timeout
        try:
            with httpx.Client(verify=self._verify, timeout=bound) as client:
                response = client.request(method, url, headers=self._headers(node), json=json)
        except httpx.TimeoutException as exc:
            raise self._timeout_error(node, path, bound, exc) from exc

        # Version negotiation on every call.
        agent_version = response.headers.get(AGENT_VERSION_HEADER)
        if agent_version is not None:
            actual = parse(agent_version)
            if actual[0] != parse(CONTRACT_VERSION)[0]:
                raise CompatibilityRefusal(
                    code=CODE_AGENT_VERSION_INCOMPATIBLE,
                    operation=path,
                    required_version=parse(CONTRACT_VERSION),
                    actual_version=actual,
                )

        if response.status_code >= 400:
            raise self._error_from_response(node, response)
        return response.json() if response.content else {}

    @staticmethod
    def _timeout_error(
        node: Node, path: str, bound: float, exc: httpx.TimeoutException
    ) -> AgentCallError:
        """Name the bound that was hit, and say whether the node is still working.

        Before this, an ``httpx.TimeoutException`` escaped ``_call`` raw. It is
        not an ``AgentCallError``, so nothing on the way out classified it: the
        build route's catch-all recorded ``str(exc)`` -- the string "The read
        operation timed out" -- with an empty detail and no per-node outcome.
        That is the same defect one layer up, on the caller's side of the
        same hop, and it cost the estate a managed build.

        The distinction that actually matters to an operator is not the timeout;
        it is **whether work is still happening on the node.** A connect timeout
        means the request never landed, so nothing started and a retry is clean.
        A read timeout means the agent accepted it and this side stopped waiting
        -- the compile is very likely still running, and may yet produce the
        image minutes after the operation says it failed. Retrying *that*
        races a build that is still going, which is how one spec becomes two
        concurrent builds of the same reference.

        So the failure says which, and says what to check before retrying.
        """
        started = not isinstance(exc, httpx.ConnectTimeout)
        if started:
            consequence = (
                "the agent accepted the request and this side stopped waiting, so the work is "
                "probably still running on the node. Check `image_list` before retrying -- the "
                "image may appear after this failure was recorded, and a retry would race a "
                "build that is still going"
            )
        else:
            consequence = (
                "the request never reached the agent, so nothing was started there and a retry "
                "is safe"
            )
        return AgentCallError(
            code="agent_timeout",
            message=(
                f"{path} on node {node.name!r} exceeded the {bound:.0f}s bound this client "
                f"places on the hop ({type(exc).__name__}); {consequence}"
            ),
            node_id=node.id,
            detail={
                "path": path,
                "timeout_seconds": bound,
                "timeout_kind": type(exc).__name__,
                # The operator-facing distinction, as a field rather than only
                # as prose, so a caller can branch on it without parsing English.
                "work_may_still_be_running": started,
            },
        )

    def _error_from_response(self, node: Node, response: httpx.Response) -> AgentCallError:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        # FastAPI's HTTPException nests the structured error under "detail",
        # while the agent's own handlers put it at the top level. Reading only
        # the top level reduced every nested error to code "http_error" with
        # the whole JSON blob as its message, so an agent that said exactly
        # what went wrong was reported as if it had said nothing.
        nested = body.get("detail")
        if isinstance(nested, dict) and "code" in nested and "code" not in body:
            body = {**nested, "detail": nested.get("detail", {})}
        return AgentCallError(
            code=str(body.get("code", "http_error")),
            message=str(body.get("message", response.text)),
            node_id=node.id,
            detail=body.get("detail", {}) if isinstance(body.get("detail", {}), dict) else {},
        )

    # ---------------------------------------------------------------- contract
    def get_info(self, node: Node) -> dict[str, Any]:
        return self._call(node, "GET", "/agent/v1/info")

    def get_resources(self, node: Node) -> dict[str, Any]:
        return self._call(node, "GET", "/agent/v1/resources")

    def get_observed(self, node: Node, deployment_id: str) -> dict[str, Any]:
        return self._call(node, "GET", f"/agent/v1/deployments/{deployment_id}/observed")

    def get_runtime(self, node: Node, deployment_id: str, *, tail: int = 500) -> dict[str, Any]:
        return self._call(node, "GET", f"/agent/v1/deployments/{deployment_id}/runtime?tail={tail}")

    def check_endpoint(self, node: Node, port: int) -> dict[str, Any]:
        return self._call(node, "GET", f"/agent/v1/endpoint-check?port={port}")

    def acquire_model(
        self,
        node: Node,
        *,
        source_id: str,
        source_model_id: str,
        revision: str | None,
        credential: str | None,
        file_selector: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Drive one node's acquisition, carrying the credential per-request.

        ``credential`` is a **resolved secret value**, not a name: the
        coordinator resolved the reference from its provider immediately before
        this call. It travels in the request body of this one
        call, over TLS, and is held nowhere — the agent uses it for the upstream
        fetch and never writes it, and nothing on this side
        retains it once the call returns.

        Omitted entirely when there is none, rather than sent as null, so an
        agent log of the request body has no credential key to render at all.
        """
        payload: dict[str, Any] = {
            "source_id": source_id,
            "source_model_id": source_model_id,
            "revision": revision,
        }
        if credential is not None:
            payload["credential"] = credential
        # Omitted when empty for the same reason the credential is, plus one
        # more: the agent's request model forbids unknown keys, so an agent
        # predating file selection would refuse a request carrying the field
        # even to say "all files". A whole-repository acquisition therefore
        # sends exactly the body it always sent, and only a deployment that
        # actually selects requires an agent that understands selection.
        if file_selector:
            payload["file_selector"] = list(file_selector)
        # A model download can take far longer than an ordinary management
        # request.  Keep the coordinator-to-agent request alive rather than
        # incorrectly recording a healthy long download as unreachable.
        return self._call(
            node,
            "POST",
            "/agent/v1/models:acquire",
            json=payload,
            timeout=3600.0,
        )

    def replicate_model(
        self,
        node: Node,
        *,
        model: dict[str, Any],
        source_node: Node,
    ) -> dict[str, Any]:
        """Instruct ``node`` to pull ``model`` from ``source_node``.

        The source's pinned certificate fingerprint travels with the
        instruction, because only the coordinator holds it — that is what lets
        the destination verify the peer it is about to trust bytes from,
        without a certificate authority existing anywhere.

        The response is the destination's outcome. No model byte passes through
        this process.
        """
        payload = {
            "model": model,
            "source_replica": {
                "node_id": source_node.id,
                "agent_endpoint": source_node.agent_endpoint,
                "agent_cert_fingerprint": source_node.agent_cert_fingerprint,
            },
        }
        return self._call(node, "POST", "/agent/v1/models:replicate", json=payload)

    def build_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a recorded spec on ``node``."""
        return self._call(
            node, "POST", "/agent/v1/images:build", json=payload, timeout=_BUILD_TIMEOUT_SECONDS
        )

    def distribute_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Have ``node`` pull an image from a peer."""
        return self._call(
            node,
            "POST",
            "/agent/v1/images:distribute",
            json=payload,
            timeout=_DISTRIBUTE_TIMEOUT_SECONDS,
        )

    def import_image(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        """Import a prebuilt archive on ``node``."""
        return self._call(node, "POST", "/agent/v1/images:import", json=payload, timeout=1800.0)

    def remove_image(self, node: Node, *, reference: str) -> dict[str, Any]:
        """Remove an image from ``node``."""
        return self._call(
            node, "POST", "/agent/v1/images:remove", json={"reference": reference}, timeout=300.0
        )

    def list_images(self, node: Node) -> list[dict[str, Any]]:
        """What ``node``'s daemon holds, for comparison against the records."""
        result = self._call(node, "GET", "/agent/v1/images")
        return result if isinstance(result, list) else []

    def create_deployment(self, node: Node, payload: dict[str, Any]) -> dict[str, Any]:
        # The first start may need to pull a multi-gigabyte runtime image.
        return self._call(
            node,
            "POST",
            "/agent/v1/deployments",
            json=payload,
            timeout=3600.0,
        )

    def stop_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        return self._call(node, "POST", f"/agent/v1/deployments/{deployment_id}:stop")

    def reconcile_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        return self._call(
            node,
            "POST",
            f"/agent/v1/deployments/{deployment_id}:reconcile",
            timeout=3600.0,
        )

    def remove_deployment(self, node: Node, deployment_id: str) -> dict[str, Any]:
        return self._call(node, "DELETE", f"/agent/v1/deployments/{deployment_id}")

    def delete_model(self, node: Node, model_id: str) -> dict[str, Any]:
        return self._call(node, "DELETE", f"/agent/v1/models/{model_id}")
