"""vLLM runtime adapter.

Each runtime adapter owns its runtime-specific Pydantic config schema; there is
never a cross-runtime config union. vLLM ``supports_distributed =
True``: a multi-node deployment against vLLM is permitted and
the runtime itself forms its distributed group.

``validate_config`` validates the incoming ``runtime_config`` against vLLM's
own schema and returns the validated (normalized) config, rejecting unknown or
invalid keys with vLLM's reason before the deployment is treated as
valid.

``build_launch_args`` turns the validated config plus the model path into the
vLLM CLI arguments that launch the serving process inside the container.

**Modelled and forwarded are different things.** The typed fields
below are ones this product understands: it knows their shape, can explain them,
and rejects a bad value before a deployment is recorded. ``extra_args`` is the
deliberate door for everything else — forwarded to vLLM verbatim and *recorded
as unvalidated*. The distinction is kept visible rather than blurred, because
claiming to have validated a flag we merely relayed is the same species of
untruth this design removed from observed state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg, normalize_flag
from tensorstead.adapters.runtimes.option_policy import (
    POLICY_VERSION,
    already_validated,
    approved_from_context,
    authorize,
    validated_context,
)
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    NodePosition,
    OptionDefault,
    ReadinessProbe,
    RuntimeOption,
    SchemaProbe,
    SuggestedImage,
)

# Flags an operator may never set through ``extra_args``, and why.
#
# This is a safety boundary, not a style preference. Each of these is owned by
# some part of the product that would be silently overridden.
_GIB = 1024 * 1024 * 1024

# Where durable cache appears *inside* the container. The host side is the
# agent's business and is deliberately not named here -- the adapter states the
# runtime's requirement, the agent decides where it lives.
_CACHE_AT = "/var/cache/tensorstead"

# Named once so the free-form host_config.environment validator that refuses
# it and inference_credential_env that sets it cannot drift apart -- the same
# reasoning as _TRANSPORT_ENVIRONMENT below, applied to the managed
# credential. host_config.environment is an
# operator-declared, unvalidated passthrough; without this it could silently
# overwrite the credential the agent route already placed in the same
# environment mapping, and _apply_requirements's own merge order made that
# override win.
_CREDENTIAL_ENV_VAR = "VLLM_API_KEY"

# The environment variables that select a distributed group's transport. Named
# once so the validator that refuses them and the code that sets them cannot
# drift apart.
_TRANSPORT_ENVIRONMENT = frozenset({"GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"})

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    # The model path is derived from what the node actually acquired. Letting
    # config name it would let a deployment serve a path it never staged, and
    # the recorded model id would then describe something that is not running.
    "model": "the model path comes from the acquired model, not from config",
    "served-model-name": "use the served_model_name field",
    # The inference credential is passed as container environment precisely so
    # it never appears in the host-visible process command line.
    # Accepting it here would undo that with no warning at all.
    "api-key": "the inference credential is delivered as environment, never argv",
    # The endpoint is the deployment's declared binding; the container
    # publishes it. A runtime told to listen elsewhere would be unreachable and
    # the product would report the port it declared rather than the one in use.
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
    # The group's topology is derived from the deployment's declared node list
    # and every one of these is rank-local. Passthrough could set
    # them too -- one shared runtime_config reaching every rank -- so a
    # deployment could claim two nodes and launch nine, name a master nothing
    # rendezvouses at, or hand a second `--headless` to a rank that already had
    # one. Found by review before it reached hardware.
    #
    # The same rule as `host` and `port`, one level down: a fact the product
    # derives may not also be supplied, because then the record and the argv can
    # disagree and nothing compares them.
    "nnodes": "the node count comes from the deployment's participating nodes",
    "node-rank": "each rank's index is derived per node, not declared once",
    "master-addr": "the rendezvous address is derived from the head node",
    "master-port": "the rendezvous port is the adapter's, not configuration's",
    "headless": (
        "which rank serves is derived from position; the head serves and every other rank does not"
    ),
}


# The probe vLLM answers with. It builds the *same* parser ``vllm serve``
# builds -- ``make_arg_parser`` is what the CLI itself calls -- and reads the
# argparse actions off it. Not ``--help`` text: that is generated output, and
# parsing generated text to recover the structure that generated it is how a
# schema drifts from the thing it describes.
#
# Emits a **tagged** envelope so ``parse_schema`` has a grammar rather than a
# format. The first draft accepted any trailing JSON object, so
# ``{"other": true}`` parsed to zero options and the route reported the image's
# surface as *known and empty* -- unknown-as-empty, through a payload that
# happened to be valid JSON, which is the precise defect this rule exists to forbid.
# A reader must be able to tell our payload from anything
# else a runtime chose to print.
_SCHEMA_GRAMMAR = "tensorstead.runtime-options/1"

_VLLM_SCHEMA_PROBE = """
import json

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = make_arg_parser(FlexibleArgumentParser())
options = []
for action in parser._actions:
    if not action.option_strings:
        continue
    kind = getattr(action.type, "__name__", None)
    if kind is None and action.type is not None:
        kind = type(action.type).__name__
    default = action.default
    if default is None:
        encoded = {"state": "absent"}
    else:
        try:
            json.dumps(default)
        except (TypeError, ValueError):
            encoded = {"state": "unrepresentable", "text": repr(default)[:200]}
        else:
            encoded = {"state": "value", "value": default}
    nargs = action.nargs
    options.append(
        {
            "name": action.dest,
            "flags": list(action.option_strings),
            "kind": kind,
            # The action *class name*, unnormalised. Normalising here would put
            # the mapping inside every built image, where fixing it means
            # rebuilding; the adapter can be corrected without touching one.
            "action": type(action).__name__,
            "nargs": None if nargs is None else str(nargs),
            "choices": [str(c) for c in (action.choices or [])],
            "default": encoded,
            "help": action.help,
        }
    )
print(json.dumps({"schema": "__GRAMMAR__", "runtime": "vllm", "options": options}))
""".replace("__GRAMMAR__", _SCHEMA_GRAMMAR)


# argparse's action classes, reduced to the shape distinctions a configuration
# has to be validated against. An unrecognised class is passed through by name
# rather than guessed at: a custom action this product has not seen is exactly
# where a guess would be wrong, and reporting the real name lets someone look.
_ACTION_KINDS: dict[str, tuple[str, bool, bool | None]] = {
    # class name -> (normalised action, repeatable, polarity)
    "_StoreAction": ("store", False, None),
    "_StoreTrueAction": ("store_true", False, True),
    "_StoreFalseAction": ("store_false", False, False),
    "_StoreConstAction": ("const", False, None),
    "_AppendAction": ("append", True, None),
    "_AppendConstAction": ("append", True, None),
    "_ExtendAction": ("append", True, None),
    "_CountAction": ("count", True, None),
    "BooleanOptionalAction": ("store_true", False, True),
}


class VLLMConfig(BaseModel):
    """vLLM's runtime-specific config schema.

    ``model`` is derived from the model path at launch, not taken from the
    operator, so a deployment cannot point at a path it has not acquired. The
    remaining fields are a documented vLLM surface, plus ``extra_args`` for the
    flags this product does not model.
    """

    model_config = ConfigDict(extra="forbid")

    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int | None = Field(default=None, ge=1)
    # Which host interface a multi-node group's collectives ride
    # Declared, with **no default**, because choosing
    # between a machine's interfaces is infrastructure policy and not something
    # this product may decide for an operator.
    #
    # Absent, the runtime chooses for itself -- which is what happened on the
    # first live TP=2 start: the node names resolve to public IPv6, the
    # collectives would have crossed the general LAN instead of the 200 Gb/s
    # RoCE link, and Gloo fell back to loopback and failed outright. That
    # failure was the lucky outcome; the same misconfiguration that formed a
    # group would have reported success while running over the wrong wire.
    #
    # Naming it here also puts *which link a deployment runs on* into the
    # record, rather than leaving it an accident of name resolution.
    distributed_interface: str | None = Field(default=None, min_length=1)
    # How vLLM executes across the group -- ``ray``, ``mp``,
    # ``external_launcher``, or whatever the deployed build offers.
    #
    # Declared with **no default**, and unvalidated against a list of names.
    # The design hardcoded ``mp`` on my judgement and never tested it; on hardware
    # the follower rank died in vLLM's KV-cache setup with "collective_rpc
    # should not be called on follower node". Which backends
    # a given vLLM build supports for a given topology is a fact about that
    # build, and this product does not have vLLM's real surface -- an allowlist
    # here would go stale on their release schedule, not ours.
    #
    # Absent means vLLM chooses for the topology it was handed, which is a
    # better answer than one this adapter guessed.
    distributed_executor_backend: str | None = Field(default=None, min_length=1)
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0.0, le=1.0)
    dtype: str | None = None
    quantization: str | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    # ``None`` means the operator did not set this switch. Keeping it distinct
    # from an explicit false prevents a runtime default from leaking into an
    # exported deployment configuration.
    enable_auto_tool_choice: bool | None = None
    max_num_seqs: int | None = Field(default=None, gt=0)
    served_model_name: str | None = Field(default=None, min_length=1)

    # -------------------------------------------------------------------------
    # Speculative decoding, including MTP (multi-token prediction). A nested
    # dict rather than flattened fields because that is the shape vLLM itself
    # takes, and because the valid inner keys depend on ``method`` and on the
    # vLLM build. Validated shallowly on purpose: see the validator below.
    speculative_config: dict[str, Any] | None = None
    # KV cache precision. The single highest-leverage memory knob on a unified
    # -memory appliance, where the KV cache and the model weights compete for
    # the same pool.
    kv_cache_dtype: str | None = None
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    async_scheduling: bool | None = None
    load_format: str | None = None
    swap_space: float | None = Field(default=None, ge=0)
    max_seq_len_to_capture: int | None = Field(default=None, gt=0)
    # Forces eager execution, skipping graph capture. Slower, and the first
    # thing worth trying when a runtime dies during startup for reasons the
    # logs do not explain -- which is what the 2026-08-10 restore attempt hit.
    enforce_eager: bool | None = None
    # Many recent architectures -- including the ones that carry MTP heads --
    # will not load without this.
    trust_remote_code: bool | None = None
    seed: int | None = None
    # vLLM logs each request by default, and a request contains a prompt.
    # Tensorstead does not read those logs into its records, but an operator
    # tailing them should be able to decide what the runtime writes
    # down at all.
    disable_log_requests: bool | None = None

    # Flags this product does not model, forwarded to vLLM verbatim.
    #
    # This exists because the alternative is worse. A closed allowlist means any
    # vLLM capability the product has not yet learned is *unreachable through
    # the product*, so an operator who needs it goes around the management
    # plane and starts a container by hand -- at which point the coordinator's
    # records describe something that is not what is running. That failure is
    # exactly what the design calls an unexpected instance, and a schema that
    # forces operators into it is a schema that manufactures its own drift.
    #
    # Recorded as unvalidated. Nothing here is checked against vLLM's real
    # surface, because this product does not have vLLM's real surface -- it has
    # a guess at it, which goes stale on the runtime's release schedule and not
    # on ours.
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    # Keep the image's own ``ENTRYPOINT`` instead of ``vllm serve``.
    #
    # For an image that has to do work between the container starting and the
    # server running -- installing a model-shipped encoder into the runtime, on
    # the DSpark recipe -- that work reads the mounted model directory, which
    # does not exist at build time. The image handles it in its own entrypoint
    # and execs the server afterwards; this says "let it".
    #
    # ``None`` rather than ``False`` so it is dropped by ``exclude_none`` and
    # no existing deployment's exported configuration grows a line for a
    # feature nobody used -- the same trap ``extra_args`` set when it defaulted
    # to an empty dict. Absent means the explicit ``vllm serve`` it has always
    # had. Declared here rather than inferred from
    # the image, because the product cannot tell a self-starting image from one
    # whose entrypoint it should override, and guessing wrong either way fails
    # at container start with nothing having said so.
    use_image_entrypoint: bool | None = None

    # Container-level needs this product does not model, on the same terms as
    # extra_args: forwarded, recorded as unvalidated, and bounded away from the
    # settings that would let a container escape its own constraints.
    #
    # Accepts ``shm_size`` (bytes), ``ipc_mode``, ``extra_ports``
    # ({"6379/tcp": 6379}), ``ulimits`` ({"memlock": [-1, -1]}), and
    # ``environment``. Deliberately *not* a general docker-run passthrough:
    # privileged, host networking, and arbitrary bind mounts are refused,
    # because an escape hatch that grants those is not configuration, it is a
    # way to run something the product cannot describe.
    host_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("host_config")
    @classmethod
    def _check_host_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"shm_size", "ipc_mode", "extra_ports", "ulimits", "environment"}
        for key in value:
            if key not in allowed:
                raise ValueError(
                    f"host_config does not accept {key!r}; accepted keys are "
                    f"{', '.join(sorted(allowed))}. Settings that let a container "
                    "escape its own constraints are deliberately not forwardable"
                )
        shm = value.get("shm_size")
        if shm is not None and (not isinstance(shm, int) or isinstance(shm, bool) or shm <= 0):
            raise ValueError("host_config.shm_size must be a positive number of bytes")
        environment = value.get("environment")
        if isinstance(environment, dict) and _CREDENTIAL_ENV_VAR in environment:
            # host_config.environment is an operator-declared, unvalidated
            # passthrough; without this a deployment could silently overwrite
            # the managed credential the agent route sets, since the merge
            # this value eventually reaches applies it after.
            # Bind the credential through inference-credential
            # binding instead -- that is the channel with the no-value-in-
            # output guarantee this one does not have.
            raise ValueError(
                f"host_config.environment may not set {_CREDENTIAL_ENV_VAR!r}; "
                "bind an inference credential instead"
            )
        return value

    @field_validator("trust_remote_code")
    @classmethod
    def _check_trust_remote_code(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        """Refuse what the authorization policy already classifies as refused.

        ``trust_remote_code`` is first-class, modelled config, so it never
        passed through ``authorize_extra_arg`` at all -- the policy's own
        ``LOADS_CODE`` classification was reported by the probe surface
        (``authorize()``, below in ``_parse_option_entry``) but never actually
        enforced against a real deployment, since nothing called it here
        Reusing ``authorize()`` rather than
        writing a second refusal message keeps the probe's answer and this
        gate from being able to disagree with each other.
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

    @model_validator(mode="after")
    def _one_source_for_the_transport(self) -> VLLMConfig:
        """A declared interface may not be contradicted by free-form environment.

        ``distributed_interface`` exists so a deployment's record states which
        link its collectives ride. ``host_config.environment`` could set the same
        variables, and the first draft let it win -- with a test named
        ``test_an_operator_override_wins`` blessing exactly that
        outcome.

        That was backwards. The general rule that an operator's explicit setting
        beats a product default is a good one, and it does not apply here,
        because these are not two settings: they are two spellings of one fact.
        Allowing both makes the record able to claim one transport while the
        container receives another -- the record-and-reality gap this product
        exists to close, reintroduced by the hand closing it.

        **Rejected even when the values agree.** Equality is not checkable for
        ``NCCL_IB_HCA``, which the agent resolves from the host after this
        validation runs, and two sources that happen to match today are two
        sources that can diverge tomorrow.

        An operator who wants a different interface changes
        ``distributed_interface``, where the choice is recorded, exported, and
        visible in `deployment show`.
        """
        if not self.distributed_interface:
            return self
        environment = self.host_config.get("environment") or {}
        conflicting = sorted(set(environment) & _TRANSPORT_ENVIRONMENT)
        if conflicting:
            raise ValueError(
                f"host_config.environment may not set {', '.join(conflicting)} while "
                f"distributed_interface is declared: the deployment record would claim "
                f"one transport and the container receive another. Set "
                f"distributed_interface alone, where the choice is recorded"
            )
        return self

    @field_validator("speculative_config")
    @classmethod
    def _check_speculative_config(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate the shape we can be sure of, and no further.

        Deliberately shallow. Each speculative method (``mtp``, ``eagle``,
        ``ngram``, a draft model) accepts a different inner key set, and which
        methods exist at all depends on the vLLM build that will run it. A
        thorough validator here would be this product asserting facts about a
        runtime it does not ship, and would refuse a working configuration the
        day vLLM adds a method -- the closed-allowlist failure, reproduced one
        level down.

        So: the keys we do understand are checked, and the rest is forwarded.
        vLLM is the authority on whether the whole is coherent, and the log tail
        is how its answer becomes visible.
        """
        if value is None:
            return None
        if not value:
            raise ValueError("speculative_config must not be empty; omit it instead")

        method = value.get("method")
        if method is not None and (not isinstance(method, str) or not method.strip()):
            raise ValueError("speculative_config.method must be a non-empty string")

        tokens = value.get("num_speculative_tokens")
        if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool)):
            raise ValueError("speculative_config.num_speculative_tokens must be an integer")
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens < 1:
            raise ValueError("speculative_config.num_speculative_tokens must be at least 1")

        draft = value.get("model")
        if draft is not None and (not isinstance(draft, str) or not draft.strip()):
            raise ValueError("speculative_config.model must be a non-empty string")

        if method is None and draft is None:
            raise ValueError(
                "speculative_config needs a 'method' (for example 'mtp') or a draft 'model'"
            )
        return value

    @field_validator("extra_args")
    @classmethod
    def _check_extra_args(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        """Refuse the passthrough keys that would override the product itself.

        An escape hatch that can reach ``--model``, ``--api-key``, or ``--port``
        is not an escape hatch, it is a way to make the coordinator's records
        describe a container that is doing something else.
        """
        for key in value:
            authorize_extra_arg(
                key,
                forbidden=_FORBIDDEN_EXTRA_ARGS,
                model_fields=cls.model_fields,
            )
        return value


class VLLMAdapter:
    """A ``RuntimeAdapter`` for vLLM (supports_distributed = True)."""

    runtime_type = "vllm"
    supports_distributed = True
    # The NGC vLLM image the estate actually runs. Named as a suggestion,
    # not a guarantee: the tag moves
    # on NVIDIA's release schedule, not on ours, and a newer tag is often
    # required for features added after this was written (the xgrammar
    # tool-calling bug is the case in point). An operator who has
    # pulled an image should prefer ``image list`` over this, since that names
    # what is actually present rather than what we guess.
    suggested_images = (
        SuggestedImage(
            "nvcr.io/nvidia/vllm:26.07-py3",
            "NVIDIA NGC vLLM image; the tag moves on NVIDIA's release schedule, "
            "not ours. Tested on DGX Spark as of 2026-08; a newer tag "
            "may be needed for features added since. Once pulled, prefer "
            "`image list` -- that names what is present, this names a starting "
            "point.",
        ),
    )

    def validate_config(
        self, config: dict[str, Any], *, approved_options: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        try:
            # A context rather than a bare constructor: the field validators
            # need to see what was approved, and Pydantic's validation context
            # is the only channel that reaches them without turning an approval
            # into a config field an operator could set.
            #
            # This is *the gate*. Every other construction in this module is a
            # re-parse of config that already passed it and uses
            # ``validated_context()`` instead.
            validated = VLLMConfig.model_validate(
                config, context={"approved_options": frozenset(approved_options)}
            )
        except ValidationError as exc:
            # Naming the accepted keys is the difference between an error a
            # user can act on and one that sends them back to guessing: the
            # bare Pydantic message reports the rejected key and a link to
            # Pydantic's docs, neither of which is about this product.
            accepted = ", ".join(sorted(VLLMConfig.model_fields))
            raise ValueError(
                f"invalid vllm config: {exc}\naccepted vllm keys: {accepted}\n"
                "a vllm flag this product does not model can be passed through "
                "extra_args, where it is forwarded unvalidated"
            ) from exc
        dumped = validated.model_dump(exclude_none=True)
        # An operator who forwarded nothing should not find `extra_args: {}` in
        # their exported deployment. `exclude_none` does not catch it -- an
        # empty dict is a value, not an absence -- and leaving it in would add
        # a line to every existing deployment's record announcing a feature
        # they did not use.
        if not dumped.get("extra_args"):
            dumped.pop("extra_args", None)
        if not dumped.get("host_config"):
            dumped.pop("host_config", None)
        return dumped

    def container_entrypoint(self, config: dict[str, Any] | None = None) -> list[str] | None:
        """``vllm serve``, unless the image is declared self-starting.

        ``None`` means "use whatever the image declares", which is what an
        image that must prepare itself before serving needs.
        """
        if (
            config
            and VLLMConfig.model_validate(config, context=validated_context()).use_image_entrypoint
        ):
            return None
        return ["vllm", "serve"]

    def inference_credential_env(self, secret: str) -> dict[str, Any]:
        """vLLM reads ``VLLM_API_KEY`` as an alternative to ``--api-key``.

        Passed as container environment rather than a launch argument so the
        credential never appears in the host-visible process command line.
        """
        return {_CREDENTIAL_ENV_VAR: secret}

    def readiness_probe(self) -> ReadinessProbe:
        """vLLM serves an OpenAI-compatible ``/v1/models`` once the model loads.

        Chosen over ``/health`` deliberately: vLLM's health endpoint answers
        from the HTTP server, which comes up before the engine finishes loading
        weights, so it reports success during exactly the window an operator
        most needs the truth. ``/v1/models`` requires a loaded, served model.

        A probe that actually generated tokens would prove more, and is
        deliberately not used: it would consume accelerator time on every
        observation, and its result would depend on sampling parameters this
        product does not own. Listing served models is a metadata read, which
        keeps the probe on the management side.
        """
        return ReadinessProbe(path="/v1/models", expect_status=200)

    def config_advisories(
        self, config: dict[str, Any], platform_facts: dict[str, Any] | None = None
    ) -> list[str]:
        """Valid vLLM configurations whose consequences are easy to miss."""
        validated = VLLMConfig.model_validate(config, context=validated_context())
        notes: list[str] = []
        notes += _quantization_advisory(validated, platform_facts or {})
        notes += _speculative_model_advisory(validated)
        notes += _speculative_backend_advisory(validated)

        if validated.speculative_config is not None and validated.max_num_batched_tokens is None:
            # Observed on the appliance after MTP went live: vLLM accepted the
            # speculative config, then reduced its own per-step token budget to
            # 2048 and said so in a startup line nobody reads until something is
            # already wrong. Speculative tokens draw on that budget, so the
            # effective batch size stops being the one that was chosen.
            #
            # The wording here avoids vLLM's internal term for that budget on
            # purpose: a guardrail forbids planner vocabulary in product code,
            # and it is right to -- this product places nothing and decides
            # nothing about where work runs. Quoting the runtime's field name
            # would be a false positive, but a rule relaxed for a string
            # literal is one a real violation can hide inside.
            notes.append(
                "speculative decoding is enabled without max_num_batched_tokens; "
                "vLLM reduces its per-step token budget to accommodate "
                "speculative tokens, often to 2048. Set max_num_batched_tokens "
                "explicitly, or lower num_speculative_tokens or max_num_seqs, "
                "if throughput under concurrency matters"
            )

        speculative = validated.speculative_config or {}
        if speculative.get("num_speculative_tokens", 0) and validated.max_num_seqs is None:
            notes.append(
                "speculative decoding pays at low concurrency and can cost "
                "throughput under load; max_num_seqs is unset, so concurrency is "
                "whatever vLLM defaults to. Worth measuring single-stream latency "
                "and aggregate throughput separately -- they can move in opposite "
                "directions"
            )

        return notes

    def container_requirements(
        self, config: dict[str, Any], position: NodePosition | None = None
    ) -> ContainerRequirements:
        """vLLM's container-level needs, which are not expressible as flags.

        Shared memory is the one that bites. vLLM's workers communicate through
        ``/dev/shm``, and Docker gives a container 64MB by default. A
        tensor-parallel deployment exhausts that immediately, and the failure
        names neither shared memory nor Docker -- it surfaces as a worker dying
        or a hang during initialisation, which is a long afternoon if nobody
        told you where to look.

        Scaled by parallelism rather than fixed, so a single-GPU deployment is
        not taxed for a capability it does not use.
        """
        validated = VLLMConfig.model_validate(config, context=validated_context())
        overridden = validated.host_config

        parallel = validated.tensor_parallel_size * (validated.pipeline_parallel_size or 1)
        # 1GiB is comfortable for a single worker; multi-worker groups need
        # room per worker. Deliberately generous: the cost of being wrong high
        # is some unused address space, and the cost of being wrong low is an
        # obscure hang.
        default_shm = _GIB if parallel <= 1 else _GIB * 2 * parallel

        # vLLM compiles on first use -- Triton kernels, and torch.compile
        # artefacts -- and caches both under a home-relative directory inside
        # the container. Tensorstead destroys and re-creates the container on
        # every restart and every revision change, so without somewhere durable
        # that work is redone every single time. Observed on the appliance
        # 2026-08-11: four Triton kernels compiling on live traffic, minutes
        # after a restart, on a deployment that had already been started
        # several times that day.
        environment = {str(k): str(v) for k, v in (overridden.get("environment") or {}).items()}
        environment.setdefault("VLLM_CACHE_ROOT", f"{_CACHE_AT}/vllm")
        environment.setdefault("TRITON_CACHE_DIR", f"{_CACHE_AT}/triton")
        # Set last and unconditionally: HOME decides where a library that
        # consults neither variable will still land, and several do.
        environment.setdefault("HOME", _CACHE_AT)

        # A distributed group rendezvouses over the host's fabric, which a
        # bridged container cannot reach, and the runtime's collective
        # transport needs the RDMA devices exposed. Declared **only when the
        # deployment actually spans nodes**:
        # a single-node deployment must not be handed the host network stack
        # because a distributed one needed it.
        network_mode: str | None = None
        devices: list[str] = []
        capabilities: list[str] = []
        # RDMA registers memory with the NIC, which pins it. Declared with the
        # device rather than separately, because handing a container
        # /dev/infiniband without these is handing it a device it cannot use --
        # which is what the first group to reach its own collectives found
        # on hardware.
        rdma_ulimits: dict[str, tuple[int, int]] = {}
        if position is not None and position.node_count > 1:
            network_mode = "host"
            devices = ["/dev/infiniband"]
            capabilities = ["IPC_LOCK"]
            rdma_ulimits = {"memlock": (-1, -1)}
            # Rank-local. ``self_address`` is the management hostname, which on
            # this estate resolves to public IPv6 and sent the collectives over
            # the general LAN. When an interface is declared,
            # the agent replaces this with that interface's own IPv4 before the
            # container is created -- it is the only party that can see it.
            environment.setdefault("VLLM_HOST_IP", position.self_address)
            if validated.distributed_interface:
                # Gloo's transport, which is what actually failed: it could not
                # resolve a usable peer address and fell back to loopback.
                environment.setdefault("GLOO_SOCKET_IFNAME", validated.distributed_interface)
                # The collective library's *bootstrap* only. Its data path
                # rides libibverbs and is selected by ``NCCL_IB_HCA``, which the
                # agent fills in from the RoCE device behind this interface.
                # Setting this one alone would have produced a group that
                # formed, ran, and reported success without using RDMA at all.
                environment.setdefault("NCCL_SOCKET_IFNAME", validated.distributed_interface)

        requirements = ContainerRequirements(
            network_mode=network_mode,
            devices=list(devices),
            # vLLM's documented default listener. Declared here rather than
            # assumed by the agent, which used to publish 8000 for every runtime
            # including llama.cpp's 8080.
            api_port=8000,
            # Only the head serves. Every other rank runs with ``--headless``
            # and starts no API server, so it has nothing to answer a readiness
            # probe with -- and this is the same derivation ``_distributed_args``
            # uses to decide which rank gets the flag, stated once here so the
            # argv and the observation cannot disagree about which rank
            # serves inference.
            serves_inference=position is None or position.node_index == 0,
            # Declared only for a group that spans nodes: vLLM's ranks cannot be
            # started concurrently, measured rather than assumed.
            # A single-node deployment declares nothing and
            # keeps starting exactly as it always has.
            rendezvous_port=(
                _DISTRIBUTED_RENDEZVOUS_PORT
                if position is not None and position.node_count > 1
                else None
            ),
            shm_size=int(overridden.get("shm_size", default_shm)),
            ipc_mode=overridden.get("ipc_mode"),
            extra_ports={str(k): int(v) for k, v in (overridden.get("extra_ports") or {}).items()},
            capabilities=list(capabilities),
            # The adapter's requirement first, the operator's last: an explicit
            # host_config value still wins, because an operator tuning a limit
            # for their own estate should not be silently overruled by a default
            # this adapter chose.
            ulimits={
                **rdma_ulimits,
                **{
                    str(name): (int(limits[0]), int(limits[1]))
                    for name, limits in (overridden.get("ulimits") or {}).items()
                },
            },
            environment=environment,
            cache_at=_CACHE_AT,
            # A drafter is a second set of weights this container reads, and
            # only vLLM knows that ``speculative_config.model`` is where one is
            # named -- so it is named here rather than pattern-matched by the
            # agent, which must stay runtime-agnostic.
            #
            # This is a request. The agent mounts it only if it resolves inside
            # the model store it owns, so an operator cannot turn this string
            # into a bind-mount of an arbitrary host path. When it is a
            # HuggingFace repo id -- the only form that worked before this --
            # nothing resolves, nothing is mounted, and the runtime fetches it
            # itself exactly as it did.
            model_references=_speculative_model_references(validated),
        )
        return requirements

    def schema_probe(self) -> SchemaProbe:
        """Ask this image's own parser, with a device attached."""
        return SchemaProbe(
            entrypoint=["python3"],
            command=[],
            script=_VLLM_SCHEMA_PROBE,
            # Building the parser constructs config defaults, which require
            # device detection. Without one the probe dies with "Failed to infer
            # device type" -- measured on the appliance before this was written.
            needs_accelerator=True,
        )

    def parse_schema(self, output: str) -> list[RuntimeOption]:
        """Read the probe's tagged envelope, refusing anything else.

        vLLM writes startup notices to stdout before our own output -- Triton
        availability, CUDA discovery, and on this estate a good deal more -- so
        the payload is found by scanning backwards. What it is **not** is "the
        last thing that happened to be JSON": the payload must carry our own
        grammar tag, and every option in it must have the shape that grammar
        promises.

        Raising rather than returning a short list is the point. The caller
        turns an exception into ``unknown``, and unknown is the only truthful
        answer when the thing we read was not our schema. Returning ``[]``
        reports "this runtime accepts nothing", which is never true and is the
        exact failure this rule names.
        """
        payload: dict[str, Any] | None = None
        for line in reversed(output.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                candidate = json.loads(line)
            except ValueError:
                continue
            if isinstance(candidate, dict) and candidate.get("schema") == _SCHEMA_GRAMMAR:
                payload = candidate
                break
        if payload is None:
            raise ValueError(
                f"the probe emitted no {_SCHEMA_GRAMMAR!r} payload; the image may predate "
                f"this probe, or its output may have been truncated"
            )

        entries = payload.get("options")
        if not isinstance(entries, list):
            raise ValueError(f"{_SCHEMA_GRAMMAR} payload has no options list")
        if not entries:
            # A parser with no options is not a thing vLLM produces. Far more
            # likely: the probe ran somewhere it could not build one.
            raise ValueError(f"{_SCHEMA_GRAMMAR} payload lists no options at all")

        options: list[RuntimeOption] = []
        for entry in entries:
            options.append(self._option_from(entry))
        return options

    @staticmethod
    def _option_from(entry: Any) -> RuntimeOption:
        """One envelope entry, validated rather than coerced.

        Every field is checked before use. The first draft called ``.get`` with
        a fallback on each, so a malformed entry became an option named ``""``
        that accepted nothing -- a defect rendered as a fact.
        """
        if not isinstance(entry, dict):
            raise ValueError(f"option entry is not an object: {entry!r}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"option entry has no name: {entry!r}")
        raw_flags = entry.get("flags")
        if not isinstance(raw_flags, list) or not raw_flags:
            raise ValueError(f"option {name!r} has no flags")
        flags = [str(f) for f in raw_flags]

        raw_action = entry.get("action")
        action_name = str(raw_action) if raw_action is not None else None
        normalised, repeatable, polarity = _ACTION_KINDS.get(
            action_name or "", (action_name, False, None)
        )

        raw_default = entry.get("default")
        if not isinstance(raw_default, dict) or "state" not in raw_default:
            raise ValueError(f"option {name!r} has no encoded default")
        state = str(raw_default["state"])
        if state not in ("value", "absent", "unrepresentable"):
            raise ValueError(f"option {name!r} has unknown default state {state!r}")
        default = OptionDefault(
            state=state,
            value=raw_default.get("value") if state == "value" else None,
            text=str(raw_default["text"]) if state == "unrepresentable" else None,
        )

        # Discovery is not authorization. The policy decides, and it
        # refuses anything nobody has classified -- including options this
        # adapter has never heard of, which is most of vLLM's surface and the
        # whole reason the probe exists.
        authorized, reason = authorize(name)
        # Product-owned argv keeps its own, more specific reason: an operator
        # who reaches `--headless` should be told the product derives it, not
        # merely that its effect class is refused.
        owned = next(
            (
                _FORBIDDEN_EXTRA_ARGS[k]
                for f in flags
                if (k := _normalize_flag(f)) in _FORBIDDEN_EXTRA_ARGS
            ),
            None,
        )
        if owned is not None:
            authorized, reason = False, owned

        return RuntimeOption(
            name=name,
            flags=flags,
            kind=str(entry["kind"]) if entry.get("kind") is not None else None,
            action=normalised,
            nargs=str(entry["nargs"]) if entry.get("nargs") is not None else None,
            repeatable=repeatable,
            polarity=polarity,
            choices=[str(c) for c in (entry.get("choices") or [])],
            default=default,
            help_text=entry.get("help"),
            authorized=authorized,
            unauthorized_reason=reason,
            policy_version=POLICY_VERSION,
        )

    def validate_distribution(self, config: dict[str, Any], *, node_count: int) -> None:
        """Refuse a declared group larger than the declared parallelism (017).

        vLLM spreads ``tensor_parallel_size * pipeline_parallel_size`` workers
        across the group. When that product is smaller than the node count, the
        record names nodes that can hold no worker -- a contradiction inside the
        product's own declaration, refusable without knowing anything further
        about vLLM.

        What is deliberately *not* refused here: a parallelism that does not
        divide evenly by the node count. That is plausibly invalid, but it is a
        rule about vLLM's rank assignment rather than about this record, and it
        is not one we have measured. vLLM will say so at launch, in its own
        words, and its refusal will be right where ours might not be.
        """
        if node_count <= 1:
            return
        validated = VLLMConfig.model_validate(config, context=validated_context())
        workers = validated.tensor_parallel_size * (validated.pipeline_parallel_size or 1)
        if workers < node_count:
            raise ValueError(
                f"tensor_parallel_size {validated.tensor_parallel_size} x "
                f"pipeline_parallel_size {validated.pipeline_parallel_size or 1} "
                f"= {workers} worker(s) cannot span the {node_count} nodes this "
                f"deployment names"
            )

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: NodePosition | None = None,
        endpoint_port: int | None = None,
    ) -> list[str]:
        validated = VLLMConfig.model_validate(config, context=validated_context())
        args = ["--model", model_path]
        args += _distributed_args(position)
        # The declared configuration is the single source for tensor
        # parallelism. Emitted explicitly for a multi-node
        # deployment even when it is 1, so the argv states the group's shape
        # instead of leaning on a default; single-node argv is unchanged, which
        # is what keeps every running deployment unaffected.
        spans_nodes = position is not None and position.node_count > 1
        # ``port`` is a reserved config key because the endpoint declares it and
        # the container publishes it -- true right up until this adapter asks for
        # host networking, which it does for exactly this case. With no port map
        # to translate, vLLM listened on its own default while the deployment
        # record named another port, and `deployment status` reported the head
        # unreachable on a port nothing was ever told to bind.
        #
        # Emitted only when the group spans nodes, so single-node argv stays
        # byte-identical and no running deployment is disturbed.
        if spans_nodes and endpoint_port is not None:
            args += ["--port", str(endpoint_port)]
        if validated.tensor_parallel_size != 1 or spans_nodes:
            args += ["--tensor-parallel-size", str(validated.tensor_parallel_size)]
        if validated.distributed_executor_backend:
            args += [
                "--distributed-executor-backend",
                validated.distributed_executor_backend,
            ]
        if validated.pipeline_parallel_size is not None:
            args += ["--pipeline-parallel-size", str(validated.pipeline_parallel_size)]
        if validated.max_model_len is not None:
            args += ["--max-model-len", str(validated.max_model_len)]
        if validated.gpu_memory_utilization is not None:
            # ``str``, not ``:.2f``. Two decimals silently truncated any finer
            # value: a deployment declaring 0.835 -- which is what the MiaAI
            # DeepSeek recipe asks for -- was rendering ``--gpu-memory-utilization
            # 0.83`` and running on a number nobody wrote down. Small in bytes,
            # and exactly the divergence this product exists to prevent: the
            # record said one thing and the runtime was told another, with
            # nothing comparing them.
            #
            # Every other numeric here already used ``str``; this was the only
            # formatted one, and nothing explained why.
            args += ["--gpu-memory-utilization", str(validated.gpu_memory_utilization)]
        if validated.dtype is not None:
            args += ["--dtype", validated.dtype]
        if validated.quantization is not None:
            args += ["--quantization", validated.quantization]
        if validated.reasoning_parser is not None:
            args += ["--reasoning-parser", validated.reasoning_parser]
        if validated.tool_call_parser is not None:
            args += ["--tool-call-parser", validated.tool_call_parser]
        if validated.enable_auto_tool_choice:
            args += ["--enable-auto-tool-choice"]
        if validated.max_num_seqs is not None:
            args += ["--max-num-seqs", str(validated.max_num_seqs)]
        if validated.served_model_name is not None:
            args += ["--served-model-name", validated.served_model_name]

        # -----------------------------------------------------------------
        if validated.speculative_config is not None:
            # Serialized with sorted keys and no incidental whitespace so the
            # same configuration always produces byte-identical argv. Launch
            # arguments are compared against what is running, and a
            # dict that serialized differently on each start would read as
            # drift that nobody caused.
            args += [
                "--speculative-config",
                json.dumps(validated.speculative_config, sort_keys=True, separators=(",", ":")),
            ]
        if validated.kv_cache_dtype is not None:
            args += ["--kv-cache-dtype", validated.kv_cache_dtype]
        if validated.max_num_batched_tokens is not None:
            args += ["--max-num-batched-tokens", str(validated.max_num_batched_tokens)]
        args += _switch("enable-prefix-caching", validated.enable_prefix_caching)
        args += _switch("enable-chunked-prefill", validated.enable_chunked_prefill)
        args += _switch("async-scheduling", validated.async_scheduling)
        if validated.load_format is not None:
            args += ["--load-format", validated.load_format]
        if validated.swap_space is not None:
            args += ["--swap-space", f"{validated.swap_space:g}"]
        if validated.max_seq_len_to_capture is not None:
            args += ["--max-seq-len-to-capture", str(validated.max_seq_len_to_capture)]
        args += _switch("enforce-eager", validated.enforce_eager)
        args += _switch("trust-remote-code", validated.trust_remote_code)
        if validated.seed is not None:
            args += ["--seed", str(validated.seed)]
        args += _switch("disable-log-requests", validated.disable_log_requests)

        # Forwarded last, verbatim, in the order the operator wrote them.
        # Last so that reading the generated argv shows plainly where this
        # product's opinions end and the operator's begin.
        for key, value in validated.extra_args.items():
            args += _passthrough(key, value)
        return args

    def declared_memory_fraction(self, config: dict[str, Any]) -> float | None:
        """vLLM spells it ``gpu_memory_utilization``."""
        return VLLMConfig.model_validate(config, context=validated_context()).gpu_memory_utilization

    def validate_model_path(self, path: Path) -> None:
        """Raise when the on-disk model tree is not usable by vLLM.

        vLLM loads from a directory with a parseable ``config.json`` and, when
        the weights are sharded safetensors, every shard the index names. A
        missing ``config.json`` is exactly the incident shape: the deployment
        "succeeded" at launch and surfaced only minutes later as an exited
        container with no classified cause. Checking here, before container
        creation, fails the start fast with a named reason instead.

        Mirrors the source-side structural verify but at the runtime
        boundary: the source verified what it downloaded, this verifies what
        the runtime will load. A tree without a safetensors index needs only
        ``config.json`` (non-sharded or non-safetensors formats).
        """
        if not path.is_dir():
            raise ValueError(f"model path is not a directory: {path}")
        config = path / "config.json"
        if not config.is_file():
            raise ValueError(f"model directory is missing config.json: {path}")
        try:
            json.loads(config.read_text())
        except ValueError as exc:
            raise ValueError(f"config.json at {config} is not valid JSON: {exc}") from exc

        index = path / "model.safetensors.index.json"
        if not index.is_file():
            return  # non-sharded model needs only config.json
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except ValueError as exc:
            raise ValueError(
                f"model.safetensors.index.json at {index} is not valid JSON: {exc}"
            ) from exc
        for shard in sorted(set(weight_map.values())):
            shard_path = path / shard
            if not shard_path.is_file():
                raise ValueError(
                    f"model directory is missing shard {shard!r} named by the index: {path}"
                )
            if shard_path.stat().st_size == 0:
                raise ValueError(f"model directory has an empty shard {shard!r}: {path}")


# Quantization formats whose *acceleration* depends on the accelerator rather
# than on vLLM. Naming them is not the same as claiming to know which hardware
# accelerates which -- see the advisory below for why that distinction is the
# whole point.
_HARDWARE_DEPENDENT_QUANTIZATION = {"modelopt", "nvfp4", "fp4", "modelopt_fp4"}


# The port every rank rendezvouses on. Fixed rather than configurable: it is
# an implementation detail of the group, not something a client ever reaches,
# and making it settable would be one more per-node fact to keep consistent.
_DISTRIBUTED_RENDEZVOUS_PORT = 29500


def _distributed_args(position: NodePosition | None) -> list[str]:
    """Rank and rendezvous facts, derived from position.

    Emitted only for a multi-node deployment. A single-node deployment produces
    exactly the argv it produced before this existed, which is what keeps every
    running deployment unaffected.

    **Rank 0 is the head, and it is the first declared node.** That is a
    convention this adapter chooses, not something the product knows: the
    product hands over a position in the declared order and nothing more.

    Note what this deliberately does *not* derive: tensor parallelism. It used
    to emit ``--tensor-parallel-size <node_count>``, which was wrong twice over
    It duplicated the flag, because ``build_launch_args``
    emits the declared value as well -- and a record declaring 3 across two
    nodes produced ``2 ... 3``, leaving vLLM's duplicate-option behaviour to
    decide which declared fact was real. It was also only ever *accidentally*
    right: tensor parallelism counts GPU shards, which equals the node count
    only when every node has exactly one GPU, and it ignored pipeline
    parallelism entirely -- a two-node ``PP=2, TP=1`` group is legitimate and
    was being overridden. Position supplies rank and rendezvous. Parallelism is
    declared.

    Note what is *not* here: no launch ordering. The recipe this was designed
    against starts the worker before the head, and `_drive_nodes` dispatches to
    every node concurrently on purpose (every nominated node is
    attempted even when one stalls for minutes pulling an image). Whether vLLM
    requires that order or merely documents it is unverified, and building an
    ordering mechanism against a guess is how a previous unwatched assertion
    cost two restarts. Attempt concurrently; let the log tail name the rank
    that failed if one does.
    """
    if position is None or position.node_count <= 1:
        return []
    head = position.peer_addresses[0] if position.peer_addresses else position.self_address
    args = [
        "--nnodes",
        str(position.node_count),
        "--node-rank",
        str(position.node_index),
        "--master-addr",
        head,
        "--master-port",
        str(_DISTRIBUTED_RENDEZVOUS_PORT),
    ]
    if position.node_index != 0:
        # Every rank but the head runs workers only: no API server, no engine
        # core. Read from the working recipe's compose command, which ends
        # ``${HEADLESS:+--headless}`` and sets HEADLESS on the worker alone.
        #
        # Its absence is what failed every distributed start. vLLM's
        # ``MultiprocExecutor`` gives a follower no ``rpc_broadcast_mq``; we
        # launched rank 1 as a leader anyway, so its engine core reached
        # ``get_kv_cache_specs`` -> ``collective_rpc`` and the build correctly
        # refused with "collective_rpc should not be called on follower node"
        # as a result.
        #
        # Worth recording how this was nearly missed twice: the flag's own help
        # text says "See multi-node data parallel documentation", which reads as
        # though it does not apply to tensor parallelism, and a summary of the
        # recipe claimed neither rank used it. The compose file says otherwise.
        # The runtime's source and the working recipe were right; both
        # descriptions of them were wrong.
        args.append("--headless")
    return args
    # No executor backend here any more. The design hardcoded ``mp``, which was
    # mine and was never verified: the comment beside it argued that *naming* a
    # backend is legitimate, and then quietly chose one. It reached hardware and
    # the follower rank died inside vLLM's KV-cache setup with
    # ``collective_rpc should not be called on follower node`` -- a leader-only
    # path running on rank 1.
    #
    # The backend is now declared configuration with no default, so an absent
    # value lets vLLM pick what suits the topology it was handed, and an
    # operator can try another without waiting on a code change. Replacing one
    # unverified constant with a different unverified constant would have been
    # the same mistake wearing a new value.


def _speculative_model_references(validated: VLLMConfig) -> list[str]:
    """Config values that name a drafter's weights, for the agent to consider.

    Separate from :func:`_speculative_model_advisory` on purpose. The advisory
    tells an operator that a drafter is outside what the product manages; this
    tells the *agent* where a drafter was named, so it can bring one back inside
    when the operator acquired it properly.

    Returns the value verbatim, repo id or path alike, because deciding which
    of those it is requires knowing the agent's model store -- and the agent is
    the only party that does.
    """
    speculative = validated.speculative_config or {}
    drafter = speculative.get("model")
    if not drafter or not isinstance(drafter, str):
        return []
    return [drafter]


def _speculative_model_advisory(validated: VLLMConfig) -> list[str]:
    """A drafter named here is outside everything the product manages.

    ``--model`` is derived from an acquired model precisely so a deployment
    cannot point at weights it never staged, and ``extra_args`` refuses
    ``model`` for the same reason. ``speculative_config.model`` is the same
    door one level down, and it was left open: whatever is named here is
    fetched by the runtime itself at container start.

    Found in practice on 2026-08-12. A drafter given as a HuggingFace repo id
    worked -- and vLLM downloaded it during startup, so the deployment now
    depends on weights that have no model record, no pinned revision, no
    replica on any node, and no presence in `model list`. The container is
    re-created on every restart, so that fetch is a network dependency at
    every start, in a product that otherwise resolves images locally to avoid
    exactly that.

    Not refused. A drafter is a legitimate thing to want, this is the only way
    vLLM accepts one, and refusing it would push the operator outside the
    management plane -- the closed-allowlist failure this design exists to avoid.
    Said out loud instead, because an unrecorded dependency an operator knows
    about is a choice, and one they do not know about is a surprise waiting
    for the next restart.
    """
    speculative = validated.speculative_config or {}
    drafter = speculative.get("model")
    if not drafter:
        return []
    return [
        f"speculative_config names a drafter model ({drafter!r}) that Tensorstead "
        "does not manage: it has no model record, no pinned revision, and no "
        "replica on any node, and the runtime fetches it at container start. "
        "Acquiring the drafter through model.acquire does not currently change "
        "this -- the field is passed to vLLM verbatim"
    ]


def _speculative_backend_advisory(validated: VLLMConfig) -> list[str]:
    """A drafter does not inherit the primary model's MoE backend.

    The primary's MoE backend is set through ``extra_args`` as ``moe-backend``,
    because it is not a field this product models. The drafter's is a *second*
    and separate choice, spelled ``moe_backend`` **inside**
    ``speculative_config``, and leaving it out does not mean "use the one above"
    -- it means vLLM picks for the drafter on its own.

    That matters when the two models are not quantised the same way, which is
    the ordinary case rather than an exotic one: a quantisation-specific MoE
    backend chosen for the primary can be the wrong one for an unquantised
    drafter, and the reverse. vLLM reports this as a model-construction failure
    at container start, naming a backend and a device, which reads like the
    backend is unsupported here rather than like it was applied to the wrong
    one of two models.

    Found by this estate the long way round, and the resolution is a live
    configuration rather than a theory: `qwen36-35b-a3b-nvfp4-perf` has served
    since 2026-08-16 with ``moe-backend: marlin`` for its NVFP4 primary and
    ``speculative_config.moe_backend: triton`` for its MTP drafter, both named,
    each suiting the model it applies to.

    **Fires only when the primary's backend was chosen explicitly.** An operator
    who set neither has taken vLLM's defaults for both and has no divergence to
    be surprised by; saying it anyway would make the note ambient, and an
    advisory nobody can act on is one everybody learns to scroll past.

    Deliberately does not say which backend to use for either model. That would
    be a table of which vLLM release accelerates which quantisation on which
    device -- the versioned capability table the design declines to maintain, and
    the one whose first plausible entry gets this appliance's own GB10 wrong.
    The note says the choice exists and is separate; which value is right stays
    where it already lives, in the exports and findings of a config that ran.
    """
    speculative = validated.speculative_config
    if not speculative:
        return []
    if any(_normalize_flag(key) == "moe-backend" for key in speculative):
        return []
    primary = [key for key in validated.extra_args if _normalize_flag(key) == "moe-backend"]
    if not primary:
        return []
    return [
        f"extra_args names a MoE backend for the model being served "
        f"({validated.extra_args[primary[0]]!r}), but speculative_config does not name one "
        "for the drafter, which does not inherit it -- vLLM chooses the drafter's "
        "separately. Where the two models are quantised differently, a backend that "
        "suits one can fail model construction for the other at container start. "
        "speculative_config.moe_backend sets the drafter's independently "
        ""
    ]


def _quantization_advisory(validated: VLLMConfig, platform_facts: dict[str, Any]) -> list[str]:
    """Note that a quantised model's speed depends on the accelerator.

    The appliance case: a dense 27B in ModelOpt NVFP4 logged that the GPU lacks
    native FP4 computation, so the weights were decompressed through Marlin
    rather than executed as FP4. The model still serves, correctly, and slower
    than the operator expected -- possibly slower than an unquantised or
    differently-quantised copy would have been. Nothing in the product said so
    before the model was chosen, staged, and deployed.

    **This deliberately does not claim whether *this* accelerator supports the
    format.** Encoding that would mean shipping a table of which compute
    capability accelerates which numeric format, and the first entry I would
    have written -- "capability >= 10.0 is Blackwell, therefore FP4" -- gets the
    appliance's own GB10 wrong, which is exactly the hardware this note exists
    for. A product that is confidently wrong about the one machine in the room
    is worse than one that says "this depends, and here is where to look".

    So it reports the capability it measured, names the dependency, and points
    at the place the runtime states its actual verdict.
    """
    quantization = (validated.quantization or "").strip().lower()
    if quantization not in _HARDWARE_DEPENDENT_QUANTIZATION:
        return []

    capability = platform_facts.get("accelerator_compute_capability")
    measured = f" (this node reports compute capability {capability})" if capability else ""
    return [
        f"{validated.quantization!r} quantization is requested{measured}. Whether "
        "these weights execute natively or are decompressed in software before "
        "each operation depends on the accelerator and the runtime build, and "
        "the difference is large -- a decompressed path can be slower than not "
        "quantising at all. vLLM states which it chose during startup: check "
        "`deployment runtime <name>` after the first start"
    ]


def _switch(flag: str, value: bool | None) -> list[str]:
    """Render a tri-state boolean as vLLM's paired ``--x`` / ``--no-x`` flags.

    ``None`` emits nothing, which leaves vLLM's own default in force. That is
    not the same as ``False``, which explicitly asks for the feature to be off
    — and the difference is load-bearing when vLLM changes a default between
    versions, since only the explicit form survives the change.
    """
    if value is None:
        return []
    return [f"--{flag}"] if value else [f"--no-{flag}"]


def _normalize_flag(key: str) -> str:
    """``max_num_seqs`` and ``--max-num-seqs`` name the same flag.

    Delegates to the shared helper so this adapter and llama.cpp normalise
    identically — they previously each held their own copy of this line, and
    each held its own copy of the bug above it.
    """
    return normalize_flag(key)


def _passthrough(key: str, value: str | int | float | bool) -> list[str]:
    """Render one unvalidated passthrough argument.

    Booleans use the same paired form as modelled switches; everything else is
    stringified as written. No attempt is made to guess whether vLLM wants this
    flag at all — that is the whole point of the field.
    """
    flag = _normalize_flag(key)
    if isinstance(value, bool):
        return [f"--{flag}"] if value else [f"--no-{flag}"]
    return [f"--{flag}", str(value)]
