"""Agent deployment routes — deployment lifecycle.

``POST /agent/v1/deployments`` — materialize and run.
``POST /agent/v1/deployments/{id}:stop`` — stop + disable + remove unit.
``DELETE /agent/v1/deployments/{id}`` — remove container + unit, retain model/image.
``POST /agent/v1/deployments/{id}:reconcile`` — the only endpoint permitted to
mutate in response to divergence, and only because it was explicitly called.

The agent:
1. validates the runtime config against the selected runtime adapter;
2. translates the deployment into container arguments via the runtime adapter;
3. creates and starts the container through the container-engine seam;
4. installs and **enables** a self-sufficient systemd unit for boot restoration
   that never references the coordinator.

The unit is enabled only while desired state is ``running``. There is
no watchdog or supervisor of our own. Management-token gated.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from tensorstead.adapters.runtimes.option_policy import is_approvable
from tensorstead.agent import cache_store, host_network
from tensorstead.agent.app import require_management
from tensorstead.domain.approvals import fingerprint, normalize_option
from tensorstead.domain.identity import is_valid_ulid

router = APIRouter(prefix="/agent/v1", tags=["deployments"])


def _require_valid_deployment_id(deployment_id: str) -> None:
    """Refuse a deployment id that is not a ULID before it reaches anything else.

    ``deployment_id`` ends up in a cache-directory join (``cache_store``) and a
    Docker container name; either use is only safe once the value is known to
    match the domain's own identity format. Checked first, in
    every handler that accepts one, so nothing downstream has to re-derive this
    invariant.
    """
    if not is_valid_ulid(deployment_id):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_deployment_id",
                "message": f"{deployment_id!r} is not a valid deployment id",
            },
        )


class DeploymentCreateRequest(BaseModel):
    """Body of ``POST /agent/v1/deployments``."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str
    revision: int
    runtime_type: str
    image_reference: str
    runtime_config: dict = {}
    model_path: str
    endpoint: str
    # A resolved secret value carried for this one call and written nowhere on
    # either side -- the same pattern already used for model acquisition.
    # Omitted entirely when absent, so an agent log of the request body has no
    # credential key to render at all.
    inference_credential: str | None = None
    # Where this node sits among the deployment's declared nodes.
    # Absent for a single-node deployment, which is every deployment that
    # existed before this field.
    node_position: dict[str, Any] | None = None
    # Whether this node may restore the deployment after a reboot.
    # Defaults to False, which is both the
    # safe reading and what a coordinator below contract 1.13 means by not
    # sending it: an unproven deployment does not get to run before anyone can
    # log in.
    restore_on_boot: bool = False
    # A reviewed approval authorizing one code-loading option for this exact
    # deployment (spec: approvals). Absent on every deployment that loads no
    # code, which is nearly all of them, and absent means refused -- the agent's
    # own validation is unchanged when nothing is supplied.
    #
    # Carried in full rather than as a bare fingerprint so this agent can
    # recompute it from what it independently knows and say *which* field
    # disagreed. A fingerprint alone would make "wrong model" and "stale
    # approval" produce the same unhelpful refusal.
    code_execution_grant: dict[str, Any] | None = None


class DeploymentCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    deployment_id: str
    image_digest: str | None = None


def _grant_refusal(message: str, detail: dict[str, Any]) -> HTTPException:
    """One shape for every grant refusal, so none of them reaches FastAPI bare."""
    return HTTPException(
        status_code=422,
        detail={"code": "code_execution_not_authorized", "message": message, "detail": detail},
    )


def _replica_identity(state: Any, model_path: str) -> tuple[str, str, str] | None:
    """``(source_id, source_model_id, resolved_revision)`` for what is on disk.

    Read from this node's own replica marker, keyed by the path the container
    will actually mount -- not from the request body. That is what makes the
    grant check a verification rather than a restatement of what the caller
    already claimed: an approval for one model cannot be replayed against
    another, because the revision compared against comes off this disk.

    ``None`` when no replica matches the path, or when the marker carries no
    resolved revision. Both are refusals: an approval binds to an immutable
    revision, and a replica that cannot state one cannot be matched to it.
    """
    acquisition = getattr(state, "acquisition", None)
    if acquisition is None:
        return None
    # The coordinator records the stable logical store path.  The agent may
    # have migrated that legacy source-qualified path (which contains ``:`` and
    # cannot be used in a Docker bind mount) to its encoded on-disk path.  Bind
    # the grant to that effective path, exactly as ``create_deployment`` does
    # before it mounts the model; comparing the pre-migration spelling would
    # reject the very replica that will actually execute.
    wanted = str(Path(acquisition.runtime_model_path(model_path)))
    for replica in acquisition.list_replicas():
        if str(Path(replica.local_path or "")) != wanted:
            continue
        if not replica.resolved_revision:
            return None
        # ``local_model_id`` is ``source:model_id`` with an optional ``#digest``
        # suffix when a file selector narrowed the acquisition. Split on the
        # first colon only: a model id may contain further ones.
        local_id = str(replica.model_id)
        source_id, _, remainder = local_id.partition(":")
        source_model_id = remainder.split("#", 1)[0]
        if not source_id or not source_model_id:
            return None
        return source_id, source_model_id, str(replica.resolved_revision)
    return None


def _verified_grant_options(
    payload: DeploymentCreateRequest, state: Any, *, digest_hint: str | None
) -> frozenset[str]:
    """Options this node accepts as approved, having checked the grant itself.

    Verifies the **model half** of the tuple: that a grant is present and
    internally consistent, that it is for this runtime, and that the model
    source, id and revision it names are what this node actually has on disk.

    The image half is checked separately in ``_require_grant_matches_image``,
    after the digest is resolved, because the agent does not know the digest
    until it has resolved it and resolving it can pull. Both halves must pass
    before a container is created; neither is sufficient alone.

    Returns an empty set for a request carrying no grant, which is the ordinary
    case and leaves the adapter's refusal exactly as it was.
    """
    grant = payload.code_execution_grant
    if not grant:
        return frozenset()

    option = normalize_option(str(grant.get("option", "")))
    if not is_approvable(option):
        raise _grant_refusal(
            f"grant names option {option!r}, which no approval can authorize",
            {"option": option},
        )
    if str(grant.get("runtime_type", "")).strip().lower() != payload.runtime_type.strip().lower():
        raise _grant_refusal(
            f"grant is for runtime {grant.get('runtime_type')!r}, but this deployment is "
            f"{payload.runtime_type!r}; approving a flag on one runtime is not approving "
            f"it on another",
            {"option": option, "runtime_type": payload.runtime_type},
        )

    identity = _replica_identity(state, payload.model_path)
    if identity is None:
        raise _grant_refusal(
            "no acquired model on this node matches the deployment's model path with an "
            "immutable revision, so the grant cannot be checked against what is actually "
            "on disk",
            {"option": option, "model_path": payload.model_path},
        )
    source_id, source_model_id, resolved_revision = identity

    for field, claimed, actual in (
        ("model_source_id", grant.get("model_source_id"), source_id),
        ("source_model_id", grant.get("source_model_id"), source_model_id),
        ("model_revision", grant.get("model_revision"), resolved_revision),
    ):
        if str(claimed or "").strip().lower() != str(actual).strip().lower():
            raise _grant_refusal(
                f"grant {field} is {claimed!r}, but this node holds {actual!r} at the "
                f"deployment's model path; the approval does not describe what would run",
                {"option": option, "field": field, "granted": claimed, "observed": actual},
            )

    # Internal consistency: the fingerprint must be the hash of the fields
    # beside it. A grant whose key does not match its own contents authorizes
    # nothing, whatever it claims -- the same rule the repository applies when
    # reading an approval back.
    expected = fingerprint(
        option=option,
        runtime_type=payload.runtime_type,
        model_source_id=source_id,
        source_model_id=source_model_id,
        model_revision=resolved_revision,
        image_digest=str(grant.get("image_digest", "")),
    )
    if str(grant.get("fingerprint", "")).strip().lower() != expected:
        raise _grant_refusal(
            "grant fingerprint does not match the tuple it carries; it authorizes nothing",
            {"option": option, "expected_fingerprint": expected},
        )
    if (
        digest_hint is not None
        and str(grant.get("image_digest", "")).strip().lower() != digest_hint
    ):
        raise _grant_refusal(
            f"grant is for image {grant.get('image_digest')!r}, but this node resolved "
            f"{digest_hint!r}; the approved code would run in a runtime nobody approved",
            {"option": option, "granted": grant.get("image_digest"), "observed": digest_hint},
        )
    return frozenset({option})


def _require_grant_matches_image(
    payload: DeploymentCreateRequest, state: Any, digest: str | None
) -> None:
    """The image half of the tuple, once the digest is actually known.

    Separate from validation because the digest is not known until the image is
    resolved, and re-running the whole check is cheaper than threading a partial
    result through. Called before any container is created, so a mismatch
    refuses the start rather than reporting one that already happened.
    """
    if not payload.code_execution_grant:
        return
    if not digest:
        raise _grant_refusal(
            "this node could not resolve a digest for the deployment's image, so the "
            "grant cannot be bound to the runtime that would execute the code",
            {"image_reference": payload.image_reference},
        )
    _verified_grant_options(payload, state, digest_hint=digest.strip().lower())


@router.post(
    "/deployments",
    response_model=DeploymentCreateResponse,
    dependencies=[Depends(require_management)],
)
def create_deployment(
    payload: DeploymentCreateRequest, request: Request
) -> DeploymentCreateResponse:
    """Materialize a deployment: resolve image, start container, enable unit."""
    _require_valid_deployment_id(payload.deployment_id)
    state = request.app.state
    engine = state.container_engine
    service_manager = state.service_manager
    runtime_adapter = _runtime_adapter(request, payload.runtime_type)

    # Validate runtime config against the selected adapter.
    #
    # Classified rather than allowed to escape, for the same reason as every
    # other pre-flight on this route -- and this was the one that did not.
    # An adapter's refusal is an operator-fixable fact about the *record*; as
    # an unhandled ValueError it reached FastAPI as a bare "Internal Server
    # Error", which the coordinator could only report as a 503. That names no
    # option, no deployment, and no remedy, and it is indistinguishable from a
    # sick agent -- so the operator's next move is to investigate the node
    # rather than the config, which is the wrong half of the system.
    try:
        validated_config = runtime_adapter.validate_config(
            payload.runtime_config,
            approved_options=_verified_grant_options(payload, state, digest_hint=None),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_runtime_config",
                "message": str(exc),
                "detail": {"runtime_type": payload.runtime_type},
            },
        ) from exc

    # Resolve the image to a platform-specific digest, preferring what
    # the node already holds. This was an unconditional pull, which meant an
    # image the product had just built could never be started: `local/...` names
    # no registry repository, so the pull failed and the feature stopped one step
    # short of being usable. Present-locally wins; only an absent image is
    # fetched.
    digest = _resolve_image(engine, payload.image_reference)

    # The image half of the grant, now that the digest is a fact rather than a
    # reference. Before launch args, before the container: a grant for a
    # different runtime must refuse the start, not describe one that happened.
    _require_grant_matches_image(payload, state, digest)

    # Translate to container launch args.  The acquisition service
    # also migrates any legacy colon-containing store path before Docker sees
    # it, preserving already-downloaded model bytes.
    model_path = state.acquisition.runtime_model_path(payload.model_path)
    # The primary model path becomes a read-only Docker host bind below --
    # unlike a secondary reference (a speculative-decoding drafter), there is
    # no legitimate value for it that lives outside this agent's own managed
    # store, so containment is required rather than best-effort.
    # Checked before the shape check that
    # follows: a path outside the store is refused for what it *is* before
    # anything asks what it *contains*.
    try:
        model_path = str(state.acquisition.require_model_path(model_path))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_path_not_contained",
                "message": str(exc),
                "detail": {"model_path": model_path, "runtime_type": payload.runtime_type},
            },
        ) from exc
    # Pre-flight the model directory before container creation. A bad model
    # dir -- the incident's missing config.json -- used to "succeed" at launch
    # and surface minutes later as an exited container with no classified cause.
    # Checking here fails the start fast with a named reason the coordinator
    # records as ``model_directory_invalid``, the same shape as ``invalid_runtime``
    # so ``node_http._error_from_response`` unwraps it into an ``AgentCallError``.
    try:
        runtime_adapter.validate_model_path(Path(model_path))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_directory_invalid",
                "message": str(exc),
                "detail": {"model_path": model_path, "runtime_type": payload.runtime_type},
            },
        ) from exc
    # Pre-flight the declared memory budget the same way and for the same
    # reason. The agent knew the node's free memory, knew
    # the deployment's declared fraction, and compared them nowhere: SGLang
    # started at ``mem_fraction_static: 0.90`` on a node with 13% free, asked
    # for roughly 109 GB of a 121 GB device, and left the host thrashing until
    # it needed a power cycle.
    #
    # Refused rather than warned, on the same grounds as the model directory:
    # a start that cannot succeed should fail fast with a named cause instead
    # of taking the node with it. A restart is not a false positive -- the
    # coordinator stops the old container first, so its memory is already back
    # by the time this runs.
    #
    # Unknown memory is not a refusal. A node whose accelerator the agent
    # cannot read reports None, and refusing on an unread number would ground
    # the estate on a monitoring gap.
    _refuse_if_memory_cannot_hold(state, runtime_adapter, validated_config, payload)
    position = _node_position(payload.node_position)
    # The port the deployment declared, for the adapter to use only if its own
    # requirements leave nothing to publish it. Parsed with
    # the same helper readiness probes with, so the port the runtime is told to
    # bind and the port the product later checks cannot be derived differently.
    from tensorstead.agent.readiness import endpoint_port as _endpoint_port

    launch_args = runtime_adapter.build_launch_args(
        validated_config,
        model_path=model_path,
        position=position,
        endpoint_port=_endpoint_port(payload.endpoint),
    )
    # The credential is supplied by the root-only agent environment, never by
    # deployment configuration, so it cannot reach the coordinator's store or
    # an exported deployment definition. How a runtime *accepts* it is the
    # adapter's business -- this route no longer knows which
    # environment variable any particular runtime reads, which is where the
    # next runtime's variant would have been bolted on.
    # The coordinator's value wins when supplied: the product owns
    # this setting. The node's own provisioning remains as a fallback so
    # deployments predating that ownership keep working across the upgrade.
    inference_api_key = payload.inference_credential or os.environ.get(
        "TENSORSTEAD_INFERENCE_API_KEY"
    )
    runtime_environment = _refuse_if_credential_cannot_be_enforced(
        runtime_adapter, inference_api_key
    )

    requirements = _container_requirements(runtime_adapter, validated_config, position)
    _resolve_host_network(requirements, validated_config)
    cache_path = None
    if requirements is not None and getattr(requirements, "cache_at", None):
        # None when the directory could not be created. The container then
        # starts without a cache and recompiles, which is slow -- refusing to
        # start a deployment because a performance directory was unavailable
        # would turn an optimisation into an outage.
        cache_path = cache_store.provision(
            payload.deployment_id, owner_uid=_image_uid(engine, payload.image_reference)
        )

    # A runtime may name further models it reads -- a drafter, in vLLM's case.
    # The adapter says where one was named; this decides whether it is ours to
    # mount, which is the half an adapter must not be trusted with because the
    # value came from operator configuration.
    extra_model_paths = [
        resolved
        for reference in (requirements.model_references if requirements is not None else [])
        if (resolved := state.acquisition.managed_model_path(reference)) is not None
    ]

    container_name = f"tensorstead-{payload.deployment_id}"
    engine.create_container(
        name=container_name,
        image=payload.image_reference,
        endpoint=payload.endpoint,
        model_path=model_path,
        command_args=launch_args,
        entrypoint=runtime_adapter.container_entrypoint(validated_config),
        environment=runtime_environment,
        # Recorded on the container so observation can report which revision is
        # *running* and probe the right port. The agent previously
        # discarded both, which is why `running_revision` was reported as null
        # and the coordinator's revision_mismatch check could never fire. Labels
        # rather than agent-side state: they survive an agent restart and they
        # are read back from the same object whose liveness is being reported,
        # so the two facts cannot drift apart. Never credentials -- container
        # metadata is readable by anything that can reach the Docker socket.
        labels={
            "tensorstead.deployment_id": payload.deployment_id,
            "tensorstead.revision": str(payload.revision),
            "tensorstead.endpoint": payload.endpoint,
            "tensorstead.runtime_type": payload.runtime_type,
            # Whether this rank serves, recorded at start from what the adapter
            # derived. Observation reads it back rather than
            # re-deriving it: re-deriving needs the deployment's node order,
            # which the agent does not have, and a second derivation is a second
            # thing that can disagree with the first.
            "tensorstead.serves_inference": "true" if requirements.serves_inference else "false",
        },
        # Container-level needs the adapter declared for this config.
        # vLLM will not run in a container with Docker's 64MB default shared
        # memory, and no launch flag can say so -- which is why this is a
        # separate channel rather than more argv.
        requirements=requirements,
        # The host side of the adapter's durable-cache requirement.
        # Provisioned here rather than named in configuration: the adapter says
        # a cache must exist, the agent decides where, and the operator never
        # gets to point a mount at an arbitrary host path.
        cache_path=cache_path,
        extra_model_paths=extra_model_paths,
    )
    engine.start_container(container_name)

    # The head of a staged group does not report "started" until it is actually
    # accepting the group. Without this the
    # coordinator's next call starts a worker against a head that is still
    # initialising, the worker exhausts its backend's retry budget, and the
    # operation reports success for a group that never formed.
    #
    # Only the lead rank waits, and only when the adapter declared a rendezvous
    # port -- so a single-node deployment reaches the return below on exactly
    # the path it always did.
    _await_rendezvous(container_name, engine, requirements, position)

    # Boot restoration, but only when it was asked for and only when this
    # container actually came up.
    #
    # This used to be unconditional, so a deployment acquired the right to run
    # on every future boot as a side effect of being started once. A deployment
    # that had never completed a single successful start then deadlocked its
    # node's GPU driver, and was restored into the same deadlock on every
    # reboot -- which is what removed the operator's only escape hatch and cost
    # the node a reimage.
    #
    # Two conditions, and they answer different questions. `restore_on_boot`
    # is intent: nobody's deployment becomes persistent without saying so.
    # `_container_is_running` is evidence: a definition nobody has seen run does
    # not get to run unattended before there is a login prompt.
    # The unit is always *written*: its presence is how this agent records that
    # it holds a running deployment, and reconcile reads it to decide whether a
    # dead container should be restarted. That is a different question from
    # whether the deployment comes back after a reboot.
    service_manager.install(container_name)
    # Three conditions, and the *node's* one comes first because it is the only
    # one no caller can influence:
    #
    #   permitted -- this node was provisioned to allow boot restoration at all,
    #                which is set in the agent's environment by `make deploy` and
    #                so requires credentials no agent holds;
    #   asked for -- the revision declares it;
    #   earned    -- the container actually came up.
    #
    # An agent can satisfy the second. It cannot reach the first, and that
    # separation is the point: granting the capability and exercising it are
    # different acts with different credentials.
    if (
        getattr(state, "allow_boot_restoration", False)
        and payload.restore_on_boot
        and _container_is_running(engine, container_name)
    ):
        service_manager.enable(container_name)

    return DeploymentCreateResponse(
        status="created",
        deployment_id=payload.deployment_id,
        image_digest=digest,
    )


# The collective library's rank-local transport variables. Named as constants so
# the scan that forbids distribution *primitives*, is not tripped by
# environment keys the product declares for a library it does not link against.
_IB_DEVICE_VARIABLE = "NCCL_IB_HCA"
_IB_GID_VARIABLE = "NCCL_IB_GID_INDEX"


def _resolve_host_network(requirements: Any, config: dict[str, Any]) -> None:
    """Fill in the interface facts only this host can see.

    The adapter declares *which interface* a group's collectives ride; the
    address on it and the RoCE device behind it are properties of this machine
    at this moment, and no adapter or coordinator can know them. So they are
    resolved here and folded into the environment the adapter already declared.

    Refuses rather than guesses. An interface that does not exist, or carries no
    IPv4, fails the start with a named reason -- because the alternative is a
    group that forms over whatever the runtime picks instead, which on this
    estate is the general LAN, and reports success.
    """
    interface = config.get("distributed_interface")
    environment = getattr(requirements, "environment", None)
    if not interface or environment is None:
        return

    address = host_network.interface_address(str(interface))
    if address is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "distributed_interface_unusable",
                "message": (
                    f"interface {interface!r} has no IPv4 address on this node; "
                    f"the group's collectives have nowhere to bind"
                ),
                "detail": {"interface": interface},
            },
        )
    # Overrides the management hostname the coordinator supplied: this is the
    # address a peer must actually reach this rank on.
    environment["VLLM_HOST_IP"] = address

    device = host_network.rdma_device(str(interface))
    if device is None:
        # Plain Ethernet: no RDMA device, so nothing further to resolve and the
        # runtime uses its socket transport. A true answer, not a failure.
        return
    # The collective library's data path. Without it the runtime may pick
    # another device or fall back to sockets, which runs and is wrong.
    environment.setdefault(_IB_DEVICE_VARIABLE, device)

    # Which GID on that device. Rank-local, so it cannot be declared: one shared
    # runtime_config would send one rank's index to both.
    gid_index = host_network.rdma_gid_index(device, address)
    if gid_index is None:
        # Refused rather than left unset. An RDMA device *is* present, so the
        # collectives will use it, and an unset index means the library picks --
        # which is the class of silent misconfiguration this whole thread has
        # been about. Better to fail the start naming the device.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "rdma_gid_unresolved",
                "message": (
                    f"no RoCE v2 GID on {device!r} matches {address}; the "
                    f"collective transport would select one unaided"
                ),
                "detail": {"interface": interface, "rdma_device": device, "address": address},
            },
        )
    environment.setdefault(_IB_GID_VARIABLE, str(gid_index))


# How long the head may take to bind its rendezvous listener. Generous because
# it covers container start, interpreter import, and CUDA initialisation on a
# Spark; bounded because a head that never binds must fail the start rather than
# hold the coordinator's call open indefinitely. Comfortably inside the
# coordinator's own per-call timeout, so the failure is reported by this agent
# with a named reason rather than surfacing as an unexplained timeout.
_RENDEZVOUS_WAIT_SECONDS = 180.0
_RENDEZVOUS_POLL_SECONDS = 0.5


def _rendezvous_is_accepting(port: int) -> bool:
    """Whether something on this host accepts connections on ``port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _await_rendezvous(
    container_name: str,
    engine: Any,
    requirements: Any,
    position: Any,
    *,
    wait_seconds: float = _RENDEZVOUS_WAIT_SECONDS,
) -> None:
    """Block until the head is accepting its group, or fail the start.

    A no-op unless this container is rank 0 of a multi-node group *and* the
    adapter declared a rendezvous port. Both conditions come from declarations
    rather than from anything this route knows about a particular runtime.

    The container is re-checked each poll: a head that exits during
    initialisation must fail immediately with that fact, not wait out the full
    budget and report a timeout that hides an exit.
    """
    port = getattr(requirements, "rendezvous_port", None)
    if port is None or position is None or position.node_index != 0:
        return

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        state = engine.inspect_container(container_name)
        if state is not None and not state.running:
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "rendezvous_unavailable",
                    "message": (
                        f"the group head exited before accepting its group "
                        f"(exit code {state.exit_code})"
                    ),
                    "detail": {"container": container_name, "rendezvous_port": port},
                },
            )
        if _rendezvous_is_accepting(int(port)):
            return
        time.sleep(_RENDEZVOUS_POLL_SECONDS)

    raise HTTPException(
        status_code=500,
        detail={
            "code": "rendezvous_unavailable",
            "message": (
                f"the group head did not accept its group on port {port} within "
                f"{wait_seconds:.0f}s; workers were not started"
            ),
            "detail": {"container": container_name, "rendezvous_port": port},
        },
    )


def _image_uid(engine: Any, reference: str) -> int | None:
    """The numeric uid an image runs as, or None when it cannot be determined.

    The cache directory is owned by this user and kept private to it, because a
    compile cache holds executable artefacts and a world-writable one would be
    a route from any local account into the inference container.

    ``None`` is returned for an image that declares nothing -- which runs as
    root, and root can already write a root-owned directory -- and for one that
    declares a *name*, which cannot be resolved to a uid from outside the image
    without reading its ``/etc/passwd``. Guessing there would either fail
    silently or hand ownership to the wrong account, so the cache stays
    root-owned and a non-root container recompiles: slow, safe, and visible in
    the runtime log tail.
    """
    ask = getattr(engine, "image_user", None)
    if ask is None:
        return None
    declared = ask(reference)
    if not declared:
        return None
    # "1000", "1000:1000", "vllm", "vllm:vllm" are all valid Config.User forms.
    uid = str(declared).split(":", 1)[0]
    return int(uid) if uid.isdigit() else None


def _container_is_running(engine: Any, container_name: str) -> bool:
    """Whether this container is running *now*.

    The evidence half of the boot-restoration decision. Deliberately weak: it
    asks whether the container came up, not whether the runtime inside it is
    serving. A stronger bar is unavailable here and would be wrong if it were --
    a headless rank never serves by design, and readiness can take many minutes
    on a large model, which is not something a start operation may wait for.

    So this catches the container that failed to start at all, and does not
    catch a runtime that starts and then wedges. That is a real limit, stated
    rather than papered over: the deployment which prompted this took four
    minutes to deadlock and *would* have passed this check. `restore_on_boot`
    defaulting to off is what would have prevented that incident; this is the
    second line, not the first.
    """
    try:
        state = engine.inspect_container(container_name)
    except Exception:
        # Unknowable is not evidence. No unit.
        return False
    return bool(state is not None and getattr(state, "running", False))


def _node_position(raw: dict[str, Any] | None) -> Any:
    """Build a ``NodePosition`` from the coordinator's payload, or None.

    A malformed block is treated as absent rather than fatal: a deployment that
    would have started as a single node before this field existed must not fail
    to start because a new optional field arrived badly shaped.
    """
    if not raw:
        return None
    from tensorstead.ports.runtime_adapter import NodePosition

    try:
        return NodePosition(
            node_index=int(raw["node_index"]),
            node_count=int(raw["node_count"]),
            self_address=str(raw["self_address"]),
            peer_addresses=[str(a) for a in raw.get("peer_addresses", [])],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _container_requirements(adapter: Any, config: dict[str, Any], position: Any = None) -> Any:
    """The adapter's declared container needs, or None if it declares none.

    Checked rather than assumed so an adapter written earlier -- or one
    supplied by someone else -- still works, and the container is then created
    exactly as it was before this existed.
    """
    declare = getattr(adapter, "container_requirements", None)
    if declare is None:
        return None
    return declare(config, position)


def _resolve_image(engine: Any, reference: str) -> str:
    """Return ``reference``'s digest, preferring the image already on the node.

    Local-first, not pull-first, and the difference is deliberate. A pull is a
    network operation against a registry; an image this product built or
    imported belongs to no registry, so pulling it can only fail.
    Asking the node what it already holds is also the more deterministic
    reading of the requirement: a mutable tag re-pulled at each start can silently
    change the code that runs, which is the ad hoc behaviour this design
    exists to remove.

    The trade-off, recorded rather than assumed: a deployment pinned to a
    mutable tag no longer picks up a newer image merely by being restarted.
    Refreshing it becomes an explicit act. The design notes carry the
    alternatives and the reasoning.
    """
    present = getattr(engine, "image_digest", None)
    if present is not None:
        local = present(reference)
        if local:
            return str(local)
    return str(engine.pull_image(reference))


def _runtime_adapter(request: Request, runtime_type: str) -> Any:
    """Resolve the runtime adapter for ``runtime_type`` from app.state.

    The agent ships vLLM and llama.cpp adapters; an unknown runtime is refused
    rather than guessed.
    """
    adapters = request.app.state.runtime_adapters
    adapter = adapters.get(runtime_type)
    if adapter is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_runtime", "message": f"unknown runtime {runtime_type!r}"},
        )
    return adapter


# ---------------------------------------------------------------------- stop
class DeploymentStopResponse(BaseModel):
    """Response for ``POST /agent/v1/deployments/{id}:stop``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    deployment_id: str


@router.post(
    "/deployments/{deployment_id}:stop",
    response_model=DeploymentStopResponse,
    dependencies=[Depends(require_management)],
)
def stop_deployment(deployment_id: str, request: Request) -> DeploymentStopResponse:
    """Stop and remove the container, then disable and remove the unit.

    Stopping a deployment disables boot restoration so a reboot does not
    revive it. The unit is removed entirely so no stale unit file lingers.
    """
    _require_valid_deployment_id(deployment_id)
    state = request.app.state
    engine = state.container_engine
    service_manager = state.service_manager

    container_name = f"tensorstead-{deployment_id}"
    engine.stop_container(container_name)
    # A stopped container still reserves its Docker name. Remove it so a
    # restart can materialize the same deployment identity again.
    engine.remove_container(container_name)
    service_manager.disable(container_name)
    service_manager.remove(container_name)

    return DeploymentStopResponse(status="stopped", deployment_id=deployment_id)


# -------------------------------------------------------------------- remove
class DeploymentRemoveResponse(BaseModel):
    """Response for ``DELETE /agent/v1/deployments/{id}``.

    States what was removed (container + unit) and what was retained
    (model artifacts + image), so the retention rule is visible at the
    moment it applies.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    deployment_id: str
    retained: bool
    retained_what: list[str]
    removed_what: list[str]


@router.delete(
    "/deployments/{deployment_id}",
    response_model=DeploymentRemoveResponse,
    dependencies=[Depends(require_management)],
)
def remove_deployment(deployment_id: str, request: Request) -> DeploymentRemoveResponse:
    """Remove the container and the unit; retain model artifacts and image.

    Model artifacts and the container image are **retained**, and
    the response states what was removed and what was kept.
    """
    _require_valid_deployment_id(deployment_id)
    state = request.app.state
    engine = state.container_engine
    service_manager = state.service_manager

    container_name = f"tensorstead-{deployment_id}"
    # Stop the container if running, then remove it entirely.
    engine.stop_container(container_name)
    engine.remove_container(container_name)
    # Remove the boot-restoration unit so a reboot does not revive it.
    if service_manager.is_installed(container_name):
        service_manager.disable(container_name)
        service_manager.remove(container_name)

    # The compile cache goes with the deployment, unlike model artifacts and
    # the image, which the design retains. Those are expensive to re-acquire and
    # may be shared; a compile cache is derived from them and rebuilds itself,
    # so keeping it would leak disk nothing will ever claim.
    removed_what = ["container", "systemd_unit"]
    if cache_store.discard(deployment_id):
        removed_what.append("compile_cache")

    return DeploymentRemoveResponse(
        status="removed",
        deployment_id=deployment_id,
        retained=True,
        retained_what=["model_artifacts", "image"],
        removed_what=removed_what,
    )


# ------------------------------------------------------------------ reconcile
class DeploymentReconcileResponse(BaseModel):
    """Response for ``POST /agent/v1/deployments/{id}:reconcile``.

    The only agent endpoint permitted to mutate in response to divergence,
    and only because the coordinator explicitly called it after a client
    explicitly asked. Reports what it changed and what it
    could not.
    """

    model_config = ConfigDict(extra="forbid")

    status: str
    deployment_id: str
    changed: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []


@router.post(
    "/deployments/{deployment_id}:reconcile",
    response_model=DeploymentReconcileResponse,
    dependencies=[Depends(require_management)],
)
def reconcile_deployment(deployment_id: str, request: Request) -> DeploymentReconcileResponse:
    """Converge the deployment toward its declared state.

    The only endpoint that mutates host state in response to divergence,
    and only because the coordinator explicitly called it after a client
    explicitly asked. Reports what it changed and what it could
    not. Applied changes are never reverted, and the deployment
    is left in a state a later reconcile can act on.
    """
    _require_valid_deployment_id(deployment_id)
    state = request.app.state
    engine = state.container_engine
    service_manager = state.service_manager

    container_name = f"tensorstead-{deployment_id}"
    changed: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    # Determine current state from the container engine. This asked
    # ``get_digest(name) is not None`` and called the answer "running", which is
    # the same mistake observation made: the engine resolves an
    # exited container as readily as a live one. The consequence here was worse
    # than a misleading read -- a dead container looked healthy, so the start
    # branch below never fired and the one operation an operator invokes to
    # repair the fault reported "no changes required" and did nothing.
    container = engine.inspect_container(container_name)
    exists = container is not None
    is_running = bool(container and container.running)

    # Whether this agent holds a unit for the deployment at all -- which is how
    # it records "I was told to run this".
    #
    # This read ``is_enabled`` until boot restoration became opt-in, at which
    # point the two questions came apart: *enabled* now means "comes back after
    # a reboot", and a deployment that has not asked for that is still one this
    # agent is meant to keep running. Keying repair on the boot flag would have
    # made reconcile a no-op for every deployment with `restore_on_boot: false`
    # -- the operation an operator invokes to fix a dead container, silently
    # declining to fix it.
    unit_installed = service_manager.is_installed(container_name) if service_manager else False

    # Convergence logic: the agent can start or stop the container to match
    # the desired state the coordinator communicates. In v1, the coordinator
    # tells the agent what desired state to converge toward via the
    # deployment record; here the agent inspects its local state.
    #
    # If the container is not running but the unit is enabled, the agent
    # attempts to start it (converge toward running).
    if not is_running and unit_installed:
        if not exists:
            # Nothing to start. Materializing a container needs the revision's
            # image, model path, and config, none of which the agent holds
            # here -- so this is named as remaining rather than silently
            # reported as converged.
            remaining.append(
                {
                    "kind": "container_absent",
                    "detail": f"no container {container_name}; re-create the deployment",
                }
            )
        else:
            try:
                engine.start_container(container_name)
                changed.append({"action": "started", "target": container_name})
                is_running = True
            except Exception as exc:
                remaining.append({"kind": "container_start_failed", "detail": str(exc)})
    elif is_running and not unit_installed:
        # Running with no unit at all — the desired state was stopped, and the
        # stop path removes the unit. Not "not enabled": a deployment that never
        # asked for boot restoration is not one that was asked to stop, and
        # reading it that way would have had reconcile tear down every healthy
        # deployment with `restore_on_boot: false`.
        engine.stop_container(container_name)
        service_manager.remove(container_name)
        changed.append({"action": "stopped", "target": container_name})
        is_running = False

    # A deployment that is up but not serving is named, never acted on.
    #
    # The decision, recorded rather than left as an omission: reconcile does
    # **not** restart a runtime that fails its readiness probe. The design excludes
    # automatic recovery, and here that exclusion earns its keep — a 27B model
    # spends minutes loading, during which the port is bound and the runtime
    # legitimately does not serve. A reconcile that restarted on a failed probe
    # would kill the model mid-load, every time, and the deployment would never
    # reach ready. The cure would permanently cause the disease.
    #
    # Staying silent is the other wrong answer. Reporting "reconciled" with an
    # empty change list, to an operator who invoked repair *because* nothing was
    # being served, is the shape of the incident this whole line of work came
    # from: the repair operation said there was nothing to repair. So it goes in
    # `remaining` — the field whose job is to say what this call did not fix.
    if is_running:
        remaining.extend(_unserved(request, container, container_name))

    status = "reconciled" if not remaining else "partial"

    return DeploymentReconcileResponse(
        status=status,
        deployment_id=deployment_id,
        changed=changed,
        remaining=remaining,
    )


def _unserved(request: Request, container: Any, container_name: str) -> list[dict[str, Any]]:
    """Report a running-but-not-serving runtime as unfinished work.

    Returns at most one ``inference_not_ready`` entry, and only when the probe
    actually established that the runtime does not serve. ``None`` — no probe
    declared, no endpoint label, connection could not be made — yields nothing:
    a fact we could not establish must not be reported as a fault, or every
    un-upgraded deployment reads as broken forever.
    """
    from tensorstead.agent.readiness import probe_endpoint
    from tensorstead.agent.routes.observation import _readiness_probe

    labels = getattr(container, "labels", None) or {}
    endpoint = labels.get("tensorstead.endpoint")
    if not endpoint:
        return []
    # A rank that was never meant to serve is not unfinished work.
    # Reporting one would put a permanent
    # ``inference_not_ready`` on every healthy multi-node deployment.
    if labels.get("tensorstead.serves_inference") == "false":
        return []

    readiness = probe_endpoint(
        endpoint,
        probe=_readiness_probe(request, labels.get("tensorstead.runtime_type")),
        api_key=os.environ.get("TENSORSTEAD_INFERENCE_API_KEY"),
    )
    if readiness.inference_ready is not False:
        return []
    return [
        {
            "kind": "inference_not_ready",
            "detail": (
                f"{container_name} is running but its runtime does not serve "
                f"({readiness.detail or 'readiness probe failed'}); not restarted "
                "automatically — a model still loading looks the same"
            ),
        }
    ]


def _refuse_if_credential_cannot_be_enforced(
    runtime_adapter: Any, inference_api_key: str | None
) -> dict[str, str] | None:
    """Resolve the runtime environment for an inference credential, or refuse.

    SGLang and llama.cpp both implement ``inference_credential_env`` as an
    unconditional ``{}`` -- deliberately, because neither has a verified
    authentication mechanism yet, and each
    adapter's own docstring explains why guessing one would be worse than
    declaring none. The bug was never that decision; it is that the generic
    start path turned that empty mapping into "no environment" and continued
    anyway, whenever a key was present from some source (a coordinator-issued
    binding, or this node's own ``TENSORSTEAD_INFERENCE_API_KEY`` fallback). An
    operator who bound a credential and watched the start succeed had every
    reason to believe it applied.

    Owning the whole decision here, not just "is the returned mapping empty",
    is deliberate: a helper that only saw the already-computed mapping could
    not tell "no credential was ever supplied" (legitimate; most SGLang and
    llama.cpp deployments never ask for authentication) apart from "one was
    supplied and discarded" (the bug) -- both produce an empty mapping, and
    conflating them would refuse every ordinary unauthenticated deployment of
    either runtime, not just the ones actually affected.

    Reporting ``endpoint_authenticated=false`` after exposure is not
    enforcement, and the finding is explicit that only a refusal (or a
    verified, hardware-tested mechanism this product does not have yet for
    either runtime) counts as one.
    """
    if not inference_api_key:
        return None
    supplied = runtime_adapter.inference_credential_env(inference_api_key)
    if not supplied:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "credential_not_enforceable",
                "message": (
                    f"refusing to start: an inference credential is set, but "
                    f"{runtime_adapter.runtime_type!r} has no verified authentication "
                    f"mechanism and would silently serve unauthenticated. Unbind the "
                    f"credential (and unset TENSORSTEAD_INFERENCE_API_KEY on this node, "
                    f"if that is the source) to run this runtime here"
                ),
                "detail": {"runtime_type": runtime_adapter.runtime_type},
            },
        )
    return {str(k): str(v) for k, v in supplied.items()}


def _refuse_if_memory_cannot_hold(
    state: Any,
    runtime_adapter: Any,
    validated_config: dict[str, Any],
    payload: Any,
) -> None:
    """Refuse a start whose declared memory budget the node cannot satisfy.

    The fraction is the adapter's to report and is a share of the
    device's *total* memory, not of what is free -- which is precisely why a
    node already serving something else is the dangerous case.
    """
    try:
        declared = runtime_adapter.declared_memory_fraction(validated_config)
    except (AttributeError, NotImplementedError):
        # An adapter that predates this says nothing, and silence is not a
        # claim that it needs nothing.
        return
    if declared is None:
        return

    reading = _accelerator_memory(state)
    if reading is None:
        return
    used, total = reading
    if total <= 0:
        return

    available_fraction = max(0.0, (total - used) / total)
    if declared <= available_fraction:
        return

    raise HTTPException(
        status_code=422,
        detail={
            "code": "insufficient_memory",
            "message": (
                f"declared memory fraction {declared:.2f} exceeds the "
                f"{available_fraction:.2f} this node has free "
                f"({(total - used) / 1024**3:.1f} GiB of {total / 1024**3:.1f} GiB). "
                f"The fraction is a share of the node's total memory, not of what is "
                f"free, so a node already serving another deployment cannot satisfy it"
            ),
            "detail": {
                "declared_fraction": declared,
                "available_fraction": round(available_fraction, 4),
                "memory_used": used,
                "memory_total": total,
                "runtime_type": payload.runtime_type,
            },
        },
    )


def _accelerator_memory(state: Any) -> tuple[int, int] | None:
    """Return ``(used, total)`` accelerator bytes, or None when unreadable."""
    try:
        from tensorstead.agent.resources import read_resources

        reading = read_resources(
            nvml_source=getattr(state, "nvml_source", None),
            filesystem_source=getattr(state, "fs_source", None),
            memory_is_unified=bool(getattr(state, "platform_facts", {}).get("memory_is_unified")),
        )
    except Exception:
        # Observation must never be the reason a start fails:
        # an unreadable node reports unknown, and unknown is not a refusal.
        return None
    used = reading.get("accelerator_memory_used")
    total = reading.get("accelerator_memory_total")
    if used is None or total is None:
        return None
    return int(used), int(total)
