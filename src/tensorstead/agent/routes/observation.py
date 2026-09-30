"""Agent observation route — ``GET /agent/v1/deployments/{id}/observed``.

A strictly read-only endpoint with **no write path**. An
observed-state request issues zero mutations — which is what makes that boundary
structural rather than a discipline: divergence is detected here and repaired
only when the coordinator later calls ``reconcile``.

Returns ``running_image_digest``, ``running_revision``, and
``endpoint_reachable`` so the coordinator can compute ``image_digest_mismatch``
and ``revision_mismatch`` divergences.

Reports an ``unexpected_instance`` when a container in the managed
namespace exists that the coordinator did not record — **reported, never
killed**. An operator's manual change must not be silently
overwritten.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict

from tensorstead.agent.app import require_management
from tensorstead.agent.container_engine.base import ContainerState
from tensorstead.agent.readiness import EndpointReadiness, ReadinessProbe, probe_endpoint

router = APIRouter(prefix="/agent/v1", tags=["observation"])


class ManagedContainerReport(BaseModel):
    """One container the agent can see in the ``tensorstead-`` namespace.

    Reported so the *coordinator* can decide what is unexpected. The agent
    knows what exists on its node; only the coordinator knows what it recorded,
    and merging those two questions into one boolean on the agent side is what
    made the previous check undecidable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    # From the ``tensorstead.deployment_id`` label. ``None`` for a container
    # created before the product labelled containers, which is an unknown
    # identity and not evidence of anything being wrong.
    deployment_id: str | None = None
    running: bool = False


class ObservedStateReport(BaseModel):
    """Extended observation carrying the ``unexpected_instance`` flag."""

    model_config = ConfigDict(extra="forbid")

    status: str
    observed_at: datetime
    running_image_digest: str | None = None
    running_revision: int | None = None
    # Transport: something accepts a connection on the deployment's port.
    endpoint_reachable: bool | None = None
    # Inference: the runtime answers its own API as a serving model. Separate
    # from reachability because the incident this field exists for lived exactly
    # between the two — a listener was present while every request failed
    # ``None`` means the probe could not establish the fact.
    inference_ready: bool | None = None
    # Whether this container was ever meant to serve. A vLLM
    # rank other than the head runs headless and starts no API server, so the
    # two fields above are not merely unknown for it -- there is nothing to
    # know. Reported so the coordinator can leave such a rank out of the
    # deployment's serving verdict instead of letting it veto one.
    #
    # ``True`` for every single-node deployment and for every container started
    # before this existed, which is what they always were.
    serves_inference: bool = True
    # A boolean, never the credential. See _endpoint_authenticated below.
    endpoint_authenticated: bool | None = None
    detail: str | None = None
    unexpected_instance: bool = False
    # The managed namespace as this node sees it. Enumeration, not judgment:
    # the coordinator compares it against the deployments it recorded.
    managed_containers: list[ManagedContainerReport] = []


@router.get(
    "/deployments/{deployment_id}/observed",
    response_model=ObservedStateReport,
    dependencies=[Depends(require_management)],
)
def get_observed(deployment_id: str, request: Request) -> ObservedStateReport:
    """Read-only observation of one deployment; mutates nothing.

    The agent inspects its local container engine and probes the deployment's
    endpoint to report what is actually serving right now. Every field is
    measured or ``None``; nothing is inferred from configuration. No sampler, no
    history, no write path.
    """
    state = request.app.state
    container_engine = getattr(state, "container_engine", None)

    container_name = f"tensorstead-{deployment_id}"
    now = datetime.now().astimezone()

    # If the container engine has no record of this deployment, it is not
    # running from the product's perspective. An unexpected instance is one
    # where the container engine sees a managed container the coordinator
    # did not ask for — reported, never killed.
    container: ContainerState | None = None
    managed: list[ManagedContainerReport] = []

    if container_engine is not None:
        container = _inspect(container_engine, container_name)
        # Enumerated on every observation rather than only when this
        # deployment's container is missing. The previous code asked only in
        # the absent case, so a hand-started container alongside a healthy one
        # was never looked for -- and that is the likelier arrangement, since
        # an operator working around the product leaves the original running.
        managed = _managed_containers(container_engine)

    unexpected = _has_mislabelled_container(managed)

    if container is None:
        # Reachability is left unknown rather than false: no probe was issued,
        # and a port this deployment does not hold may still be bound by
        # something else. `status` carries the alarm, and stating an unmeasured
        # fact as measured is the habit this whole change exists to break.
        return ObservedStateReport(
            status="not_running",
            observed_at=now,
            # No container, so no labels -- the runtime that would have run
            # here is as unknowable as its revision, and None says so.
            endpoint_authenticated=_endpoint_authenticated(None, request, None),
            detail="no container exists for this deployment",
            unexpected_instance=unexpected,
            managed_containers=managed,
        )

    labels = container.labels
    # A container the product created before labels existed carries no labels. Its
    # revision is genuinely unknown, and `None` says so; inventing one would
    # reintroduce the false confidence this change removes.
    running_revision = _int_or_none(labels.get("tensorstead.revision"))
    endpoint = labels.get("tensorstead.endpoint")
    # Read before the not-running branch, not after it. Computed below the early
    # return, this fact was truthful for a running worker and reverted to the
    # model default of `true` the moment one stopped -- so it stopped being true
    # exactly when an operator is looking at a stopped or failed rank, which is
    # when they most need it.
    #
    # Absent on containers started before this label existed, and "true" is what
    # they were: every deployment predating multi-node served from every node it
    # ran on.
    serves_inference = labels.get("tensorstead.serves_inference") != "false"

    if not container.running:
        # Not probed, therefore not claimed either way. What an operator needs
        # here is the engine's own account -- "exited (1)" -- and `status`,
        # which is measured. A port check for the "something else took 8000"
        # case has its own advisory route and belongs before a start,
        # not on every observation of a stopped deployment.
        return ObservedStateReport(
            status="not_running",
            observed_at=now,
            running_image_digest=container.image_digest,
            running_revision=running_revision,
            serves_inference=serves_inference,
            endpoint_authenticated=_endpoint_authenticated(
                container, request, labels.get("tensorstead.runtime_type")
            ),
            detail=_stopped_detail(container),
            unexpected_instance=unexpected,
            managed_containers=managed,
        )

    if not serves_inference:
        # Not probed, because there is nothing to probe: this rank runs headless
        # and starts no API server. Probing anyway would
        # report `inference_ready: false` for a rank that is doing exactly what
        # it was asked to, and one such rank is enough to make a whole healthy
        # group read as broken.
        #
        # The endpoint fields stay `None` -- not `False`, which would be a claim
        # about a listener nobody looked for, and not `True`, which would be a
        # claim about a service that does not exist here.
        readiness = EndpointReadiness(
            detail="this rank runs headless and serves no API; the group's head serves"
        )
    elif endpoint is None:
        # The coordinator has an endpoint for this deployment; this *container*
        # does not carry one, because it was created before the product recorded
        # them. Saying "no endpoint recorded" reads as a contradiction to an
        # operator looking at the declared endpoint two lines above it, and
        # names no way out. Say which thing is missing and what fixes it.
        readiness = EndpointReadiness(
            detail=(
                "container predates endpoint labelling; re-create the "
                "deployment to enable readiness probing"
            )
        )
    else:
        runtime_type = labels.get("tensorstead.runtime_type")
        readiness = probe_endpoint(
            endpoint,
            probe=_readiness_probe(request, runtime_type),
            api_key=_installed_credential(container, _adapter_for(request, runtime_type)),
        )

    return ObservedStateReport(
        status="running",
        observed_at=now,
        running_image_digest=container.image_digest,
        running_revision=running_revision,
        endpoint_reachable=readiness.reachable,
        inference_ready=readiness.inference_ready,
        serves_inference=serves_inference,
        endpoint_authenticated=_endpoint_authenticated(
            container, request, labels.get("tensorstead.runtime_type")
        ),
        detail=readiness.detail,
        unexpected_instance=unexpected,
        managed_containers=managed,
    )


def _stopped_detail(container: ContainerState) -> str:
    """Describe why a container is not running, in the engine's own words."""
    parts = [container.detail or "not running"]
    if container.exit_code is not None:
        parts.append(f"exit code {container.exit_code}")
    return "; ".join(parts)


def _int_or_none(value: str | None) -> int | None:
    """Parse a label value that should be an integer, tolerating anything else."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _adapter_for(request: Request, runtime_type: str | None) -> Any:
    """The registered runtime adapter for ``runtime_type``, or None."""
    if runtime_type is None:
        return None
    adapters = getattr(request.app.state, "runtime_adapters", None) or {}
    return adapters.get(runtime_type)


def _installed_credential(container: ContainerState, adapter: Any) -> str | None:
    """The inference credential this container was actually launched with.

    Reads the container's own recorded environment, not the agent's, so
    this reports what the runtime was actually given -- a per-deployment
    credential, the node's fallback, or nothing -- rather than assuming the
    node's fallback always applies, which could differ from what this
    specific container is running with.

    The adapter's ``inference_credential_env`` maps a value to a fixed set
    of environment variable *names* (vLLM always uses ``VLLM_API_KEY``
    regardless of what value is passed), so a throwaway value is enough to
    learn which name(s) to look up in the real environment. Returned only to
    drive a probe request -- never rendered in an API response or log line.
    """
    if adapter is None or not hasattr(adapter, "inference_credential_env"):
        return None
    for name in adapter.inference_credential_env("placeholder"):
        value = container.environment.get(name)
        if value:
            return str(value)
    return None


def _readiness_probe(request: Request, runtime_type: str | None) -> ReadinessProbe | None:
    """The declared readiness probe for this deployment's runtime."""
    adapter = _adapter_for(request, runtime_type)
    probe = getattr(adapter, "readiness_probe", None)
    if probe is None:
        return None
    result: ReadinessProbe | None = probe()
    return result


def _endpoint_authenticated(
    container: ContainerState | None, request: Request, runtime_type: str | None
) -> bool | None:
    """Whether *this deployment's* runtime gives it an inference credential.

    Reports a *fact about the endpoint*, never the credential. Tensorstead does
    not own that value: it authenticates inference clients to the runtime,
    which is the data plane this product stays out of, and it is
    provisioned onto the host by external tooling.

    Without this, an operator could not learn from the product that an endpoint
    needs credentials at all — and could not see that a runtime whose adapter
    offers no mechanism is serving unauthenticated. Both were previously
    discoverable only by calling the endpoint and reading the status code.

    Takes ``runtime_type`` and looks up exactly one adapter, the same pattern
    ``_readiness_probe`` already uses. It did not always: this checked
    ``any(...)`` across every adapter this agent knows about, so one runtime
    with a credential mechanism (vLLM, registered on every agent) made every
    *other* runtime on that agent report "requires a key" regardless of what
    that runtime's own adapter says. ComfyUI's `inference_credential_env`
    correctly returns ``{}`` and ran genuinely unauthenticated the whole time
    — confirmed by a direct, unauthenticated request returning 200 — while
    this function reported the opposite. llama.cpp and SGLang carried the same
    defect silently, since both also return ``{}`` and both share an agent
    with vLLM.

    Returns ``None`` when the runtime is unknown to this agent, because "we
    cannot say" and "no authentication" must not be the same answer. That
    includes the no-container case (``runtime_type=None``): no labels means
    no runtime is knowable either, so this reports ``None`` uniformly there
    now rather than the previous ``False``-if-no-node-wide-key,
    ``None``-otherwise split, which depended on agent-global state for a
    question about one specific, possibly nonexistent, container.
    """
    if runtime_type is None:
        return None
    adapter = _adapter_for(request, runtime_type)
    if adapter is None or not hasattr(adapter, "inference_credential_env"):
        return None
    if container is None:
        return False
    return _installed_credential(container, adapter) is not None


def _inspect(engine: Any, name: str) -> ContainerState | None:
    """Return the container's actual state, or None if it does not exist.

    Uses the narrow container-engine seam; never parses CLI output.

    This previously asked ``get_digest`` and treated any answer as proof the
    container was running. A digest is a fact about an image: the engine returns
    one just as readily for a container that exited hours ago, which is how a
    dead runtime came to report itself healthy for an entire incident window.
    """
    inspect = getattr(engine, "inspect_container", None)
    if inspect is None:
        return None
    result: ContainerState | None = inspect(name)
    return result


def _managed_containers(engine: Any) -> list[ManagedContainerReport]:
    """Enumerate the managed namespace for the coordinator to judge.

    This replaced a stub that asked the engine for a ``containers`` dict. Only
    the test fake has one, so on real hardware the expression was
    ``isinstance(None, dict)`` and ``unexpected_instance`` was **always false**
    — the guarantee reported nothing at all in production, and the tests
    passed because they ran against the one object that answered.

    The stub was also wrong where it did run. It flagged any *other*
    ``tensorstead-`` container as unexpected, so a node legitimately hosting two
    deployments accused itself the moment either one's container went away.
    Deciding this needs the coordinator's records, which the agent does not
    have, so the agent no longer tries: it reports what it sees.
    """
    enumerate_containers = getattr(engine, "list_managed_containers", None)
    if enumerate_containers is None:
        return []
    try:
        states: list[ContainerState] = enumerate_containers()
    except Exception:
        # An engine that cannot answer is not an empty namespace. Returning
        # nothing here says "we saw none", which the coordinator would read as
        # a clean node; the list is left empty and no divergence is claimed
        # from it either way.
        return []
    reports = []
    for state in states:
        if not state.name:
            continue
        reports.append(
            ManagedContainerReport(
                name=state.name,
                deployment_id=state.labels.get("tensorstead.deployment_id"),
                running=state.running,
            )
        )
    return reports


def _has_mislabelled_container(managed: list[ManagedContainerReport]) -> bool:
    """The one unexpectedness the agent *can* decide by itself.

    A container whose ``tensorstead.deployment_id`` label disagrees with its own
    name was not created by this product in its current form — a rename or a
    re-run from a committed image. That needs no coordinator records to spot.

    An *absent* label is deliberately not a disagreement: every container
    created before labels existed has none, and treating that as tampering would
    accuse the live estate of a fault it does not have.
    """
    return any(
        container.deployment_id is not None
        and container.name != f"tensorstead-{container.deployment_id}"
        for container in managed
    )


# ``_endpoint_reachable`` was removed. It returned
# ``service_manager.is_enabled(container_name)`` — whether systemd would start
# the unit at boot, which stays true across a crash and was never a statement
# about the endpoint at all. Reachability is now measured by connecting to the
# port (agent/readiness.py), and the endpoint it connects to comes from the
# container's own label rather than from the unit file.
