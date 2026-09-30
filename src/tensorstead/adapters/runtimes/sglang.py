"""SGLang runtime adapter.

The **third** runtime, and the first added since the seam was reworked. Two
things about it are worth stating because they are what the seam was for.

It owns its own config schema, and that schema does not resemble vLLM's.
SGLang spells the same ideas differently — ``--model-path`` where vLLM says
``--model``, ``--mem-fraction-static`` where vLLM says
``--gpu-memory-utilization``, ``--tp-size`` where vLLM says
``--tensor-parallel-size``, ``--context-length`` where vLLM says
``--max-model-len``. There is deliberately no translation layer and no shared
"parallelism" model: a cross-runtime config union would have to pick one
vocabulary, and the one it picked would be a lie about the other runtime
An operator writes the flags their runtime documents.

And it inherits the ``extra_args`` authorization rule for free, which is what
``adapters/runtimes/extra_args.py`` was extracted for after two adapters
implemented it separately and both got it wrong the same way.
Nothing here re-states that rule.

``supports_distributed`` is True: SGLang forms its own group with ``--nnodes``,
``--node-rank`` and ``--dist-init-addr``. The product coordinates and the
runtime distributes — the rank derivation below reads the declared
``NodePosition`` and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg, normalize_flag
from tensorstead.adapters.runtimes.option_policy import (
    already_validated,
    approved_from_context,
    authorize,
    validated_context,
)
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    NodePosition,
    ReadinessProbe,
    SchemaProbe,
    SuggestedImage,
)

# SGLang's own default server port. Named here rather than in the agent for the
# reason the design notes record: the runtime-agnostic layer held one runtime's
# port (vLLM's 8000) and a llama.cpp deployment mapped its endpoint to a
# container port nothing was listening on.
_DEFAULT_API_PORT = 30000

# The port ranks rendezvous on when the group spans nodes.
_DISTRIBUTED_RENDEZVOUS_PORT = 29500

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    "model-path": "the model path comes from the acquired model, not from config",
    "api-key": "the inference credential is delivered as environment, never argv",
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
    "nnodes": "the group's shape is derived from the deployment's nodes",
    "node-rank": "a node's rank is derived from its position in the deployment",
    "dist-init-addr": "the rendezvous address is derived from the participating nodes",
}


class SGLangConfig(BaseModel):
    """SGLang's runtime-specific config schema.

    A documented slice of ``sglang.launch_server``'s surface, in SGLang's own
    vocabulary. Anything not modelled goes through ``extra_args`` and is
    forwarded verbatim — SGLang moves faster than this file, and a schema that
    cannot express what an operator needs sends them around the management
    plane rather than stopping them.
    """

    model_config = ConfigDict(extra="forbid")

    # Topology. ``tp_size`` is SGLang's name for tensor parallelism; the group's
    # node count is derived from the deployment and never declared here.
    tp_size: int | None = Field(default=None, ge=1)
    dp_size: int | None = Field(default=None, ge=1)

    # Memory and context.
    mem_fraction_static: float | None = Field(default=None, gt=0.0, le=1.0)
    context_length: int | None = Field(default=None, gt=0)
    max_running_requests: int | None = Field(default=None, ge=1)
    chunked_prefill_size: int | None = Field(default=None, ge=1)
    kv_cache_dtype: str | None = None

    # Hybrid-attention models keep a Mamba-style cache that SGLang sizes
    # separately from the KV pool.
    max_mamba_cache_size: int | None = Field(default=None, ge=1)
    mamba_radix_cache_strategy: str | None = None

    # Structured output. Both are the reason this estate pins its images by
    # digest: a mismatched grammar library breaks tool calling outright.
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None

    # Speculative decoding. SGLang selects an algorithm by name and then reads
    # algorithm-specific settings; they are modelled flat because that is how
    # SGLang's own command line spells them.
    speculative_algorithm: str | None = None
    speculative_num_draft_tokens: int | None = Field(default=None, ge=1)
    speculative_dspark_block_size: int | None = Field(default=None, ge=1)
    speculative_eagle_num_steps: int | None = Field(default=None, ge=1)
    speculative_eagle_topk: int | None = Field(default=None, ge=1)
    speculative_draft_model_path: str | None = None

    trust_remote_code: bool | None = None
    enable_metrics: bool | None = None

    # Flags this product does not model, forwarded verbatim.
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    # Container-level needs an operator may override, in the shape the vLLM
    # adapter established so an operator who has tuned one estate does not have
    # to learn a second spelling.
    host_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trust_remote_code")
    @classmethod
    def _check_trust_remote_code(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        """Refuse what the authorization policy already classifies as refused.

        The same first-class-field gap as vLLM's identical field:
        modelled config never passed through
        ``authorize_extra_arg``, so the policy's ``LOADS_CODE`` classification
        was never enforced here. ``authorize()`` is the single source for the
        refusal reason so this cannot drift from vLLM's wording or from the
        policy's own text.
        """
        # ``already_validated`` is the re-parse path, not an escape: config only
        # reaches it by having passed this same gate on the way to a revision.
        if value and not already_validated(info):
            authorized, reason = authorize(
                "trust_remote_code", approved=approved_from_context(info)
            )
            if not authorized:
                raise ValueError(f"trust_remote_code may not be set to true: {reason}")
        return value

    @field_validator("extra_args")
    @classmethod
    def _check_extra_args(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        for key in value:
            authorize_extra_arg(
                key,
                forbidden=_FORBIDDEN_EXTRA_ARGS,
                model_fields=cls.model_fields,
            )
        return value


class SGLangAdapter:
    """A ``RuntimeAdapter`` for SGLang (``supports_distributed = True``)."""

    runtime_type = "sglang"
    supports_distributed = True
    # SGLang publishes its own images. Named as a suggestion and nothing more:
    # this estate's preference is official NVIDIA releases, and no SGLang image
    # has been verified on GB10 here. The aarch64 question in particular is the
    # operator's to settle before this is deployed -- the llama.cpp CUDA images
    # are x86_64-only, and assuming otherwise for SGLang would be a guess
    # dressed as a recommendation (once pulled, prefer `image list`).
    suggested_images = (
        SuggestedImage(
            "lmsysorg/sglang:latest",
            "SGLang project's published server image; not verified on this "
            "estate, and its aarch64/GB10 availability is not established here. "
            "Pin a digest you have tested rather than a moving tag, and prefer "
            "`image list` once pulled.",
        ),
    )

    def validate_config(
        self, config: dict[str, Any], *, approved_options: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        try:
            validated = SGLangConfig.model_validate(
                config, context={"approved_options": frozenset(approved_options)}
            )
        except ValidationError as exc:
            accepted = ", ".join(sorted(SGLangConfig.model_fields))
            raise ValueError(
                f"invalid sglang config: {exc}\naccepted sglang keys: {accepted}\n"
                "an sglang flag this product does not model can be passed through "
                "extra_args, where it is forwarded unvalidated"
            ) from exc
        dumped = validated.model_dump(exclude_none=True)
        if not dumped.get("extra_args"):
            dumped.pop("extra_args", None)
        if not dumped.get("host_config"):
            dumped.pop("host_config", None)
        return dumped

    def container_entrypoint(
        self,
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str] | None:
        """SGLang is launched as a module, not as a console script.

        ``python3 -m sglang.launch_server`` is the invocation SGLang documents
        and the one its images expect. Stated here rather than relying on an
        image's own entrypoint because an image built for a different purpose
        would otherwise silently start something else.
        """
        return ["python3", "-m", "sglang.launch_server"]

    def inference_credential_env(self, secret: str) -> dict[str, Any]:  # noqa: ARG002
        """No inference-credential mechanism is implemented for SGLang.

        An SGLang deployment therefore serves **unauthenticated**, and that is
        reported through observed state rather than left for an operator to
        discover by calling the endpoint.

        SGLang documents ``--api-key`` on the command line, which puts the
        secret where any local user reads it from ``ps``. An environment-based
        path may exist -- SGLang's variables are prefixed ``SGLANG_`` -- but it
        is not documented clearly enough to rely on, and guessing an
        environment variable name would produce a deployment that *looks*
        authenticated and is not. That is a worse outcome than declaring none,
        so this returns none until someone verifies the mechanism against a
        real image. Same reasoning, same answer, as the llama.cpp adapter.
        """
        return {}

    def readiness_probe(self) -> ReadinessProbe:
        """SGLang serves an OpenAI-compatible ``/v1/models`` once loaded.

        Chosen over ``/health`` for the reason both other adapters chose it:
        it answers only when a model is actually being served, so a runtime
        whose HTTP server is up but whose weights are still loading is not
        reported ready.
        """
        return ReadinessProbe(path="/v1/models", expect_status=200)

    def config_advisories(
        self,
        config: dict[str, Any],
        platform_facts: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        """Advise where SGLang's own settings interact.

        Advisory only: never a reason a request is refused.
        """
        validated = SGLangConfig.model_validate(config, context=validated_context())
        notes: list[str] = []
        if validated.speculative_algorithm and not validated.speculative_num_draft_tokens:
            notes.append(
                f"speculative_algorithm {validated.speculative_algorithm!r} is set but "
                "speculative_num_draft_tokens is not; SGLang will use its own default, "
                "which may not match the draft length the checkpoint was built for"
            )
        if (
            validated.speculative_algorithm
            and validated.speculative_algorithm.lower() in {"dflash2", "eagle", "eagle3", "dspark"}
            and not validated.speculative_draft_model_path
        ):
            notes.append(
                f"speculative_algorithm {validated.speculative_algorithm!r} normally needs "
                "speculative_draft_model_path; without one SGLang has no drafter to load"
            )
        if validated.mem_fraction_static is not None:
            if validated.mem_fraction_static > 0.95:
                notes.append(
                    f"mem_fraction_static {validated.mem_fraction_static} leaves little headroom "
                    "on unified memory, where the KV pool and the host share one budget"
                )
            # The trap that cost this estate a node. SGLang
            # sizes against the device's *total* memory and assumes it owns the
            # device -- so on a node already serving something else, this
            # fraction is not "the share I may use", it is "the share I will
            # take". At 0.90 beside a resident model it asked for ~109 GB of a
            # 121 GB device with 16 GB free and thrashed the host until it
            # needed a power cycle.
            notes.append(
                f"mem_fraction_static {validated.mem_fraction_static} is a fraction of the "
                f"node's TOTAL memory, not of what is free, and SGLang assumes it owns the "
                f"device: on this setting it will try to claim roughly "
                f"{validated.mem_fraction_static:.0%} of the node. Confirm the node is not "
                f"already serving another deployment before starting this one"
            )
        return notes

    def container_requirements(
        self,
        config: dict[str, Any],
        position: NodePosition | None = None,
    ) -> ContainerRequirements:
        """What SGLang needs from its container beyond argv.

        Shared memory for the same reason vLLM needs it: tensor parallelism
        runs a process per rank and they communicate through ``/dev/shm``,
        where Docker's 64 MB default produces a failure naming neither shared
        memory nor Docker.

        A distributed group rendezvouses over the host's fabric, which a
        bridged container cannot reach, and the runtime's collective
        transport needs the RDMA devices exposed. Declared **only when the
        deployment actually spans nodes**, matching the vLLM adapter's
        pattern.
        """
        validated = SGLangConfig.model_validate(config, context=validated_context())
        overridden = validated.host_config or {}
        spans_nodes = position is not None and position.node_count > 1

        # RDMA: declared only for a group that spans nodes. A single-node
        # deployment must not be handed the host network stack because a
        # distributed one needed it.
        network_mode: str | None = None
        devices: list[str] = []
        capabilities: list[str] = []
        rdma_ulimits: dict[str, tuple[int, int]] = {}
        if spans_nodes:
            network_mode = "host"
            devices = ["/dev/infiniband"]
            capabilities = ["IPC_LOCK"]
            rdma_ulimits = {"memlock": (-1, -1)}

        # Merge operator-supplied ulimits with the RDMA memlock requirement.
        merged_ulimits = {
            str(name): (int(limits[0]), int(limits[1]))
            for name, limits in (overridden.get("ulimits") or {}).items()
        }
        merged_ulimits.update(rdma_ulimits)

        return ContainerRequirements(
            network_mode=network_mode,
            devices=devices,
            capabilities=capabilities,
            api_port=int(overridden.get("api_port", _DEFAULT_API_PORT)),
            # Rank 0 is the one that serves inference; the same derivation
            # ``_distributed_args`` uses, stated once so argv and observation
            # cannot disagree about which rank to probe.
            serves_inference=position is None or position.node_index == 0,
            rendezvous_port=_DISTRIBUTED_RENDEZVOUS_PORT if spans_nodes else None,
            shm_size=int(overridden.get("shm_size", 8 * 1024**3)),
            ipc_mode=overridden.get("ipc_mode"),
            extra_ports={str(k): int(v) for k, v in (overridden.get("extra_ports") or {}).items()},
            ulimits=merged_ulimits,
            environment={str(k): str(v) for k, v in (overridden.get("environment") or {}).items()},
            # A drafter is a second set of weights this container reads, and
            # only SGLang knows that ``speculative_draft_model_path`` is where
            # one is named -- so it is named here rather than pattern-matched
            # by the runtime-agnostic agent.
            model_references=(
                [validated.speculative_draft_model_path]
                if validated.speculative_draft_model_path
                else []
            ),
        )

    def schema_probe(self) -> SchemaProbe | None:
        """No probe declared for SGLang.

        ``sglang.launch_server`` reports its surface as ``--help`` text, and
        recovering structure from generated text is the drift this mechanism
        exists to avoid. ``None`` records the surface as **unknown**, which is
        true, rather than empty, which would claim SGLang accepts nothing.
        """
        return None

    def parse_schema(self, output: str) -> list[Any]:  # noqa: ARG002
        """Unreachable while ``schema_probe`` returns None; present for the port."""
        return []

    def validate_distribution(
        self,
        config: dict[str, Any],
        *,
        node_count: int,
    ) -> None:
        """Refuse a declared shape SGLang cannot form.

        ``tp_size`` must divide across the group. A shape that cannot be formed
        is refused here, before anything starts, rather than surfacing as a
        process that exits during rendezvous with a message about world size.
        """
        validated = SGLangConfig.model_validate(config, context=validated_context())
        if validated.tp_size is None or node_count <= 1:
            return
        if validated.tp_size % node_count != 0:
            raise ValueError(
                f"tp_size {validated.tp_size} does not divide across {node_count} nodes; "
                f"SGLang splits tensors evenly, so the group cannot be formed"
            )

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: NodePosition | None = None,
        endpoint_port: int | None = None,
    ) -> list[str]:
        validated = SGLangConfig.model_validate(config, context=validated_context())
        args = ["--model-path", model_path]
        # SGLang binds **127.0.0.1** by default, exactly as llama.cpp does and
        # unlike vLLM. Under bridge networking the engine maps the endpoint's
        # port to the container's, and a server listening only on loopback
        # inside its own namespace answers nobody outside it -- while logging
        # "The server is fired up and ready to roll!" and reporting healthy.
        #
        # Found the same way in both runtimes, one after the other.
        # The container-side interface is the adapter's
        # business and is not operator-configurable: setting it
        # could only break the mapping.
        args += ["--host", "0.0.0.0"]  # noqa: S104 - inside the container namespace

        for flag, value in (
            ("--tp-size", validated.tp_size),
            ("--dp-size", validated.dp_size),
            ("--mem-fraction-static", validated.mem_fraction_static),
            ("--context-length", validated.context_length),
            ("--max-running-requests", validated.max_running_requests),
            ("--chunked-prefill-size", validated.chunked_prefill_size),
            ("--kv-cache-dtype", validated.kv_cache_dtype),
            ("--max-mamba-cache-size", validated.max_mamba_cache_size),
            ("--mamba-radix-cache-strategy", validated.mamba_radix_cache_strategy),
            ("--reasoning-parser", validated.reasoning_parser),
            ("--tool-call-parser", validated.tool_call_parser),
            ("--speculative-algorithm", validated.speculative_algorithm),
            ("--speculative-num-draft-tokens", validated.speculative_num_draft_tokens),
            ("--speculative-dspark-block-size", validated.speculative_dspark_block_size),
            ("--speculative-eagle-num-steps", validated.speculative_eagle_num_steps),
            ("--speculative-eagle-topk", validated.speculative_eagle_topk),
            ("--speculative-draft-model-path", validated.speculative_draft_model_path),
        ):
            if value is not None:
                args += [flag, str(value)]

        if validated.trust_remote_code:
            args += ["--trust-remote-code"]
        if validated.enable_metrics:
            args += ["--enable-metrics"]

        args += _distributed_args(position)

        # The endpoint's port, for the same reason vLLM emits it: a group that
        # spans nodes runs on host networking, which removes the published port
        # map that made the endpoint's port an implementation detail. Without
        # this the runtime binds its own default while the record names another,
        # and readiness reports a port nothing was told to bind
        # as a result.
        spans_nodes = position is not None and position.node_count > 1
        if spans_nodes and endpoint_port is not None:
            args += ["--port", str(endpoint_port)]

        for key, value in validated.extra_args.items():
            flag = normalize_flag(key)
            if isinstance(value, bool):
                args += [f"--{flag}"] if value else [f"--no-{flag}"]
            else:
                args += [f"--{flag}", str(value)]
        return args

    def declared_memory_fraction(self, config: dict[str, Any]) -> float | None:
        """SGLang spells it ``mem_fraction_static``, and means the whole device."""
        return SGLangConfig.model_validate(config, context=validated_context()).mem_fraction_static

    def validate_model_path(self, path: Path) -> None:
        """Raise when the on-disk tree is not one SGLang can load.

        SGLang loads a transformers-format directory, so the check is the same
        one the source makes: a directory containing ``config.json``. Checked
        here as well as there because a directory can be corrupted between
        acquire and start -- a hand-edit, a partial rsync -- and the failure
        this prevents is a container that starts, reports running, and exits
        minutes later with no classified cause.
        """
        if not path.is_dir():
            raise ValueError(f"model path is not a directory: {path}")
        if not (path / "config.json").is_file():
            raise ValueError(f"model directory has no config.json: {path}")


def _distributed_args(position: NodePosition | None) -> list[str]:
    """Derive SGLang's group flags from the declared position.

    Derived, never declared: a ``per_node`` override map would make "the
    deployment" a set of related-but-different ones, and every divergence,
    export and comparison would start comparing things that were never the same.

    A single-node deployment gets nothing at all, so its argv is unchanged by
    the existence of distribution.
    """
    if position is None or position.node_count <= 1:
        return []
    return [
        "--nnodes",
        str(position.node_count),
        "--node-rank",
        str(position.node_index),
        # Rank 0 is where the group rendezvouses, and it is the first declared
        # participant -- the same ordering the record states.
        "--dist-init-addr",
        f"{position.peer_addresses[0]}:{_DISTRIBUTED_RENDEZVOUS_PORT}",
    ]
