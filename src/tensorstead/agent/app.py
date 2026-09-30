"""Agent FastAPI application.

The node agent exposes the closed management operation set over HTTP. This
module builds the app and provides the phase-2 concerns:

- ``GET /agent/v1/info`` returns contract version, agent version, platform
  facts, service manager, and container engine.
- **In-band contract-version reporting**: every agent response carries
  the contract version in a ``X-Contract-Version`` header via middleware, so a
  heartbeat or extra round trip is never needed.

Phase 3 wires the management routes onto this app.
Each route attaches the management-token dependency (``require_management_dep``)
so the closed operation set is gated by role. The seams those
routes need — model source, acquisition, container engine, service manager —
are injected via ``app.state`` so tests can supply the fakes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tensorstead.agent.auth import AgentAuth
from tensorstead.contracts.agent import AgentInfoResponse, ContainerEngineInfo
from tensorstead.contracts.version import AGENT_VERSION, CONTRACT_VERSION
from tensorstead.version import build_identity


def _env_flag(name: str) -> bool:
    """Read a boolean from the node's provisioned environment, default False.

    Only the affirmative spellings an operator would actually write count as
    true. Everything else -- unset, empty, "no", a typo -- is false, because
    this gates a capability whose failure mode is an unrecoverable node
    and a misread should deny rather than permit.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


AGENT_VERSION_HEADER = "X-Contract-Version"

_bearer = HTTPBearer(auto_error=False)


def require_management(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """FastAPI dependency gating agent management operations.

    Delegates to the auth object's ``require_management``.
    Route modules reference this as ``Depends(require_management)`` so a single
    token configuration gates every management route on the agent.
    """
    request.app.state.auth.require_management(credentials)


def require_replication(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """FastAPI dependency gating the agent-to-agent replication hop.

    Deliberately a *different* role from ``require_management``.
    An agent holds real host authority, so a compromised one is a plausible
    outcome; holding only the replication token it can pull artifacts but
    cannot issue management instructions to any other node.
    """
    request.app.state.auth.require_replication(credentials)


# Board names known to carry integrated memory. A fallback, not the primary
# source -- see ``_detect_unified_memory_cuda`` below.
#
# Deliberately *not* including "dgx": DGX Station and DGX A100 are discrete-GPU
# machines, so the token that looks most obviously right for this estate would
# make the function confidently wrong about most of the product line it names.
_SOC_UNIFIED_TOKENS = ("grace", "gb10", "tegra", "jetson", "orin")

# ``CU_DEVICE_ATTRIBUTE_INTEGRATED`` from the CUDA driver API. 1 means the
# device shares memory with the host.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18


def _detect_unified_memory_cuda() -> bool | None:
    """Ask the driver whether the accelerator shares host memory.

    This is the authoritative answer rather than a guess about it: the CUDA
    driver reports device topology directly, so nothing here depends on
    recognising a board name.

    Deliberately the *driver* API through ``ctypes`` rather than the runtime
    API, ``pynvml``, or torch. ``libcuda.so.1`` is present wherever an NVIDIA
    driver is, so this adds no dependency, and ``cuDeviceGetAttribute`` needs no
    CUDA context -- the agent does no GPU work and should not start doing any to
    answer a question about the hardware. NVML, which the agent already uses,
    does not expose this attribute at all.

    Returns ``None`` for every failure: no driver, no device, an unexpected
    return code. "Could not ask" must stay distinct from "asked and the answer
    was no", and a detection routine is the last place to start
    collapsing that distinction.
    """
    import ctypes

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None

    try:
        if cuda.cuInit(0) != 0:
            return None
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return None
        value = ctypes.c_int()
        status = cuda.cuDeviceGetAttribute(
            ctypes.byref(value), _CU_DEVICE_ATTRIBUTE_INTEGRATED, device
        )
    except Exception:
        return None
    if status != 0:
        return None
    return bool(value.value)


def _detect_unified_memory_linux() -> bool | None:
    """Linux: ask the driver, and fall back to what the firmware calls the board.

    The driver is asked first because it *knows*; the device tree is a name
    match and can only ever recognise hardware someone has already taught this
    function about. That ordering is the difference between a fact and a guess
    that happens to be right.

    An absent or unrecognized device tree means "not stated" rather than
    "discrete": an ordinary x86 server with a discrete accelerator has no
    device-tree model either, and so does hardware nobody has taught this
    function about yet.
    """
    from_driver = _detect_unified_memory_cuda()
    if from_driver is not None:
        return from_driver

    for candidate in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            model = Path(candidate).read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if any(token in model.lower() for token in _SOC_UNIFIED_TOKENS):
            return True
    return None


def _detect_unified_memory_darwin() -> bool | None:
    """macOS: Apple Silicon is unified by construction; Intel Macs are not.

    The architecture must permit Mac Studio as a platform,
    so this branch exists even though no macOS node ships in v1. Detecting only
    the DGX hardware would have quietly encoded the assumption that this rule
    forbids.
    """
    import platform

    return platform.machine() == "arm64"


def _detect_unified_memory() -> bool | None:
    """Detect whether the accelerator shares the host's memory.

    Dispatches on the platform rather than probing one vendor's hardware. The
    return is deliberately three-valued: ``None`` means the platform did not
    say, which is a different answer from ``False`` and must stay distinct — a
    node that has not declared its topology must not be recorded as having
    denied unified memory.

    This was previously hardcoded ``True`` for every host, so a discrete-GPU
    node was reported as unified and every export's ``origin_platform``
    recorded a fact nobody had established.
    """
    if sys.platform.startswith("linux"):
        return _detect_unified_memory_linux()
    if sys.platform == "darwin":
        return _detect_unified_memory_darwin()
    return None


def _detect_compute_capability() -> str | None:
    """The accelerator's CUDA compute capability as ``"major.minor"``, or None.

    ``None`` means the question could not be answered -- no NVML, no
    accelerator, a driver that declined -- and is deliberately not a default
    value. A capability guessed wrong is worse than one absent, because every
    judgement built on it inherits the error silently.
    """
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    except Exception:
        return None
    return f"{int(major)}.{int(minor)}"


def _platform_facts_from_env() -> dict[str, Any]:
    """Best-effort platform facts gathered from the host the agent runs on."""
    import platform

    facts: dict[str, Any] = {
        "cpu_arch": platform.machine(),
        "os_family": platform.system().lower(),
        "os_version": platform.release(),
    }
    # The accelerator's compute capability, measured rather than assumed. It
    # decides which numeric formats execute on tensor cores versus being
    # decompressed in software, which is the difference between a quantised
    # model being fast and being slower than the unquantised one.
    #
    # Reported as a fact, never interpreted here: this product does not ship a
    # table of which capability supports which format, for reasons recorded in
    # the vLLM adapter.
    capability = _detect_compute_capability()
    if capability is not None:
        facts["accelerator_compute_capability"] = capability

    override = os.environ.get("TENSORSTEAD_MEMORY_IS_UNIFIED", "").strip().lower()
    if override in ("1", "true", "yes"):
        facts["memory_is_unified"] = True
    elif override in ("0", "false", "no"):
        facts["memory_is_unified"] = False
    else:
        detected = _detect_unified_memory()
        if detected is not None:
            facts["memory_is_unified"] = detected
    return facts


def _required_environment_value(name: str) -> str:
    """Return one non-empty service setting without ever rendering its value."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for the installed agent service")
    return value


def build_agent_app_from_env() -> FastAPI:
    """Build the production agent app from its systemd environment.

    The injectable :func:`build_agent_app` remains the test and embedding seam.
    This factory exists solely for the Uvicorn process installed by external
    provisioning, where secrets must arrive in a protected environment file
    rather than process arguments or source-controlled configuration.
    """
    management_token = _required_environment_value("TENSORSTEAD_MGMT_TOKEN")
    replication_token = _required_environment_value("TENSORSTEAD_REPLICATION_TOKEN")
    if management_token == replication_token:
        raise RuntimeError("TENSORSTEAD_MGMT_TOKEN and TENSORSTEAD_REPLICATION_TOKEN must differ")

    model_store = Path(
        os.environ.get("TENSORSTEAD_MODEL_STORE_PATH", "/var/lib/tensorstead/models")
    )
    state_dir = Path(os.environ.get("TENSORSTEAD_STATE_DIR", "/var/lib/tensorstead/state"))
    image_store = os.environ.get("TENSORSTEAD_IMAGE_STORE_PATH", "/var/lib/tensorstead/images")
    return build_agent_app(
        management_token=management_token,
        replication_token=replication_token,
        store_dir=model_store,
        marker_dir=state_dir,
        model_store_path=str(model_store),
        image_store_path=image_store,
    )


def build_agent_app(
    *,
    management_token: str | None = None,
    replication_token: str | None = None,
    platform_facts: dict[str, Any] | None = None,
    model_source: Any | None = None,
    container_engine: Any | None = None,
    service_manager: Any | None = None,
    acquisition: Any | None = None,
    store_dir: Any | None = None,
    marker_dir: Any | None = None,
    nvml_source: Any | None = None,
    fs_source: Any | None = None,
    model_store_path: str | None = None,
    image_store_path: str | None = None,
    image_distribution: Any | None = None,
    peer_client: Any | None = None,
) -> FastAPI:
    """Create the node agent application.

    ``management_token`` / ``replication_token`` gate the two role-distinct
    surfaces; when unset the corresponding surface is open
    (loopback permitted). ``platform_facts`` overrides the default scaffold.

    The Phase 3 seams are injectable for tests: ``model_source``,
    ``container_engine``, ``service_manager``, and ``acquisition``. When
    omitted, production defaults are built lazily from the environment.
    """
    app = FastAPI(title="tensorstead agent", version=AGENT_VERSION)
    app.state.auth = AgentAuth(management_token, replication_token)

    @app.middleware("http")
    async def _report_contract_version(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach the contract version to every agent response."""
        response = await call_next(request)
        response.headers[AGENT_VERSION_HEADER] = CONTRACT_VERSION
        return response

    # Image failures leave the node as a structured body, not a bare 500.
    # Without these, every build, import, or pull failure reached the operator
    # as "internal_error: unexpected internal error" — the reference and the
    # engine's own explanation were raised and then discarded, which made the
    # first real build on hardware undiagnosable.
    from fastapi.responses import JSONResponse

    from tensorstead.agent.acquisition import ModelAcquireError
    from tensorstead.agent.container_engine.base import ImageBuildError, ImagePullError
    from tensorstead.agent.image_distribution import ImageDistributionError

    @app.exception_handler(ModelAcquireError)
    async def _handle_model_acquire_error(_request: Request, exc: Exception) -> JSONResponse:
        """Keep an upstream model-download failure actionable to the operator."""
        return JSONResponse(
            status_code=502,
            content={"code": "model_acquire_failed", "message": str(exc), "detail": {}},
        )

    @app.exception_handler(ImageBuildError)
    async def _handle_image_build_error(_request: Request, exc: Exception) -> JSONResponse:
        error = exc if isinstance(exc, ImageBuildError) else None
        # The error's own detail carries the build evidence -- log tail, failing
        # step -- and dropping it here would put the reason on the node and the
        # bare fact on the operator, which is the state this handler exists to
        # avoid. ``reference`` stays first so an error that sets its own cannot
        # displace the one the raise site named.
        return JSONResponse(
            status_code=500,
            content={
                "code": "image_build_failed",
                "message": error.message if error else str(exc),
                "detail": {
                    "reference": error.reference if error else "",
                    **(error.detail if error else {}),
                },
            },
        )

    @app.exception_handler(ImageDistributionError)
    async def _handle_image_distribution_error(_request: Request, exc: Exception) -> JSONResponse:
        """The destination agent's own reason for a failed peer transfer.

        Added after the first live multi-node build: ``pull_from_peer`` raises
        ``ImageDistributionError`` carrying the peer-fetch, archive-load, or
        identifier-mismatch context, and no handler was registered for it -- so
        FastAPI returned a bare 500 and the coordinator could record only
        ``Internal Server Error`` as the per-node outcome.

        Which is the same defect the two handlers beside it were written to fix,
        recurring because the fix was applied to the errors that had failed
        rather than to every error this app can raise. There is now a guardrail
        test asserting each one has a handler.

        502 rather than 500: like ``ImagePullError``, the failure is upstream of
        this agent -- it could not obtain something from a peer.
        """
        error = exc if isinstance(exc, ImageDistributionError) else None
        return JSONResponse(
            status_code=502,
            content={
                "code": "image_distribution_failed",
                "message": error.message if error else str(exc),
                "detail": {"reference": error.reference if error else ""},
            },
        )

    @app.exception_handler(ImagePullError)
    async def _handle_image_pull_error(_request: Request, exc: Exception) -> JSONResponse:
        error = exc if isinstance(exc, ImagePullError) else None
        return JSONResponse(
            status_code=502,
            content={
                "code": getattr(error, "code", "image_digest_unresolved"),
                "message": str(exc),
                "detail": {"reference": error.reference if error else ""},
            },
        )

    facts = platform_facts if platform_facts is not None else _platform_facts_from_env()

    @app.get("/agent/v1/info", response_model=AgentInfoResponse)
    async def info() -> AgentInfoResponse:
        """Return the agent's identity and platform facts."""
        return AgentInfoResponse(
            contract_version=CONTRACT_VERSION,
            agent_version=AGENT_VERSION,
            # Which build this process actually is. ``agent_version`` reports
            # the package version, unchanged across sixty builds, so it could
            # never answer "is this node current".
            build=build_identity(),
            platform_facts=facts,
            service_manager="systemd",
            container_engine=ContainerEngineInfo(name="docker", version="unknown"),
        )

    # ---- Phase 3 seams on app.state (injectable for tests) ----
    # Default store/marker dirs on the host; overridable for tests.
    store = store_dir if store_dir is not None else Path("/var/lib/tensorstead/models")
    markers = marker_dir if marker_dir is not None else Path("/var/lib/tensorstead/state")

    from tensorstead.agent.acquisition import ModelAcquisitionService

    app.state.acquisition = acquisition or ModelAcquisitionService(
        store_dir=store, marker_dir=markers
    )
    app.state.store_dir = store
    app.state.marker_dir = markers

    # Peer replication: the destination agent owns the pull, so this
    # service is present on every agent — as puller and as server.
    from tensorstead.agent.replication import HTTPPeerClient, PeerReplicationService

    app.state.replication = PeerReplicationService(
        acquisition=app.state.acquisition,
        peer_client=peer_client
        if peer_client is not None
        # Same managed-CA env var image distribution already reads;
        # this construction had never read it, so model
        # replication ran with no TLS verification at all in
        # production.
        else HTTPPeerClient(ca_bundle=os.environ.get("TENSORSTEAD_AGENT_CA_BUNDLE") or None),
    )
    app.state.replication_token = replication_token

    # Default seams when not injected: the real Hugging Face source, docker
    # engine, and systemd service manager (lazy — imported only when used).
    if model_source is None:
        from tensorstead.adapters.sources.huggingface import HuggingFaceSource

        model_source = HuggingFaceSource()
    if container_engine is None:
        from tensorstead.agent.container_engine.docker_py import DockerEngine

        container_engine = DockerEngine()
    if service_manager is None:
        from tensorstead.agent.service_manager.systemd import SystemdServiceManager

        service_manager = SystemdServiceManager()
    app.state.model_source = model_source
    app.state.container_engine = container_engine
    app.state.service_manager = service_manager

    # Whether this node permits boot restoration at all.
    # **Default false**, and read from the node's own
    # provisioning rather than from anything a caller sends.
    #
    # This is the half of the guard that holds. A deployment field can be set by
    # anyone holding a management token; this file is written by `make deploy`,
    # which needs the operator's Vault and become passwords -- credentials no
    # agent has been given, deliberately. So granting the capability and
    # exercising it are separate acts with separate credentials, and an agent
    # can only ever perform the second.
    #
    # A node that was never provisioned for it refuses to enable a unit no
    # matter what the coordinator asked for, which is what keeps a deployment
    # from arranging to run before anyone can log in.
    app.state.allow_boot_restoration = _env_flag("TENSORSTEAD_ALLOW_BOOT_RESTORATION")

    # Phase 4 resource-reading seams (injectable for tests).
    app.state.nvml_source = nvml_source
    app.state.fs_source = fs_source
    app.state.model_store_path = model_store_path
    app.state.image_store_path = image_store_path
    # `facts` (above) is computed correctly -- override honoured, CUDA-driver
    # detection run -- and reaches `/agent/v1/info`, which builds its response
    # from the local variable directly. It reached nothing else: no line here
    # ever put it on `app.state`, so `/agent/v1/resources` and the
    # insufficient_memory start-time guard (both read `state.platform_facts`)
    # saw an empty dict forever, regardless of what detection or
    # TENSORSTEAD_MEMORY_IS_UNIFIED said. Two consumers of one computed fact,
    # wired to two different sources -- found while chasing why the override
    # (also correct) still showed `memory_is_unified: null` live on both
    # Sparks.
    app.state.platform_facts = facts
    # Peer-to-peer image distribution. Injectable so tests
    # supply a double; built by default so a real agent can receive an image
    # without extra configuration.
    if image_distribution is not None:
        app.state.image_distribution = image_distribution
    elif container_engine is not None:
        from tensorstead.agent.image_distribution import ImageDistributionService

        app.state.image_distribution = ImageDistributionService(
            container_engine,
            image_store_path or "/var/lib/tensorstead/images",
            os.environ.get("TENSORSTEAD_AGENT_CA_BUNDLE") or None,
        )

    # Runtime adapters by type (no cross-runtime config union).
    from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
    from tensorstead.adapters.runtimes.sglang import SGLangAdapter
    from tensorstead.adapters.runtimes.vllm import VLLMAdapter

    app.state.runtime_adapters = {
        "vllm": VLLMAdapter(),
        "llamacpp": LlamaCppAdapter(),
        "sglang": SGLangAdapter(),
    }

    _mount_routes(app)
    return app


def _mount_routes(app: FastAPI) -> None:
    """Register the Phase 3/4 management route modules onto the app."""
    from tensorstead.agent.routes import (
        deployments,
        endpoint,
        images,
        models,
        observation,
        replication,
        resources,
        runtime_logs,
    )

    app.include_router(models.router)
    app.include_router(images.router)
    app.include_router(endpoint.router)
    app.include_router(deployments.router)
    app.include_router(observation.router)
    app.include_router(runtime_logs.router)
    app.include_router(resources.router)
    app.include_router(replication.router)
