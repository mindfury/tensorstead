"""TensorRT-LLM runtime adapter (``trtllm-serve``).

NVIDIA's own serving runtime. ``trtllm-serve`` loads a Hugging Face checkpoint
directory with its PyTorch backend -- no separate engine build -- and serves an
OpenAI-compatible API.

Everything below was read from TensorRT-LLM's source at commit ``d1db352d``
(2026-10-05), mostly ``tensorrt_llm/commands/serve.py``. Where a fact is
load-bearing, the docstring names the file so it can be re-checked.

It owns its own config schema in TensorRT-LLM's vocabulary (``tp_size``,
``kv_cache_free_gpu_memory_fraction``, ``max_num_tokens``). There is no
translation from vLLM's spelling of the same ideas.

``supports_distributed = False``. TensorRT-LLM spans nodes through MPI, where
one external launcher starts every rank. That is a different shape from the one
this product's seam supports, where each node's agent starts its own rank from
facts the product derives. Tensor parallelism
across the GPUs of one node works normally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg
from tensorstead.adapters.runtimes.option_policy import (
    already_validated,
    approved_from_context,
    authorize,
    validated_context,
)
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    ReadinessProbe,
    SchemaProbe,
    SuggestedImage,
)

# ``--port`` default in ``serve.py``.
_DEFAULT_API_PORT = 8000

# TensorRT-LLM's own default for the KV-cache share of free memory.
_DEFAULT_KV_FRACTION = 0.9

_LOADS_CODE = "it loads code into the runtime, which this product does not allow from config"

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
    "grpc": "it replaces the OpenAI HTTP API the deployment's endpoint declares",
    "tokenizer": "the tokenizer comes from the acquired model, not from config",
    "hf-revision": "the model revision is the one the node acquired",
    "revision": "the model revision is the one the node acquired",
    # A YAML file, or path overrides of the whole LLM config, that the
    # deployment record would not contain.
    "config": "an options file would be configuration the deployment record cannot show",
    "extra-llm-api-options": (
        "an options file would be configuration the deployment record cannot show"
    ),
    "set": "it can override any part of the model configuration, the model included",
    "custom-module-dirs": _LOADS_CODE,
    "post-processor-hook": _LOADS_CODE,
    "middleware": _LOADS_CODE,
    "custom-tokenizer": _LOADS_CODE,
    "report-addr": "it sends data to an address configuration names",
    # Second spellings of modelled fields: set the field, where it is validated.
    "tensor-parallel-size": "use the tp_size field",
    "pipeline-parallel-size": "use the pp_size field",
    "moe-expert-parallel-size": "use the ep_size field",
    "free-gpu-memory-fraction": "use the kv_cache_free_gpu_memory_fraction field",
    "telemetry": "use the telemetry field",
    "no-telemetry": "use the telemetry field",
}


class TRTLLMConfig(BaseModel):
    """TensorRT-LLM's runtime-specific config schema, in its own vocabulary."""

    model_config = ConfigDict(extra="forbid")

    # Parallelism within one node.
    tp_size: int | None = Field(default=None, ge=1)
    pp_size: int | None = Field(default=None, ge=1)
    ep_size: int | None = Field(default=None, ge=1)

    max_batch_size: int | None = Field(default=None, ge=1)
    max_num_tokens: int | None = Field(default=None, ge=1)
    max_seq_len: int | None = Field(default=None, ge=1)
    # Share of the memory still free *after* the weights load, given to the KV
    # cache. Not a share of the device's total; see declared_memory_fraction.
    kv_cache_free_gpu_memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    kv_cache_dtype: Literal["auto", "fp8", "nvfp4"] | None = None
    enable_chunked_prefill: bool | None = None

    reasoning_parser: str | None = Field(default=None, min_length=1)
    tool_parser: str | None = Field(default=None, min_length=1)
    served_model_name: str | None = Field(default=None, min_length=1)
    chat_template: str | None = Field(default=None, min_length=1)
    trust_remote_code: bool | None = None

    # TensorRT-LLM reports anonymous usage to NVIDIA unless told not to
    # (``tensorrt_llm/usage``). Off unless the operator turns it on: a
    # self-hosted appliance should not contact a third party by default.
    telemetry: bool = False

    # Flags this product does not model, forwarded verbatim.
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("trust_remote_code")
    @classmethod
    def _check_trust_remote_code(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        """The same code-loading gate as vLLM's and SGLang's identical field."""
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
        """The shared passthrough rule, plus one TensorRT-LLM's parser needs.

        Its switches (``--enable_attention_dp``) are click flags: present means
        on, and there is no ``--no-`` form to say off. ``false`` therefore has no
        spelling, and silently dropping it would leave an operator believing
        they had turned something off.
        """
        for key, setting in value.items():
            authorize_extra_arg(
                key,
                forbidden=_FORBIDDEN_EXTRA_ARGS,
                model_fields=cls.model_fields,
            )
            if setting is False:
                raise ValueError(
                    f"extra_args {key!r} is false: TensorRT-LLM switches are off unless "
                    f"given, so leave the key out instead"
                )
        return value


class TRTLLMAdapter:
    """A ``RuntimeAdapter`` for TensorRT-LLM (``supports_distributed = False``)."""

    runtime_type = "trtllm"
    supports_distributed = False
    suggested_images = (
        SuggestedImage(
            "nvcr.io/nvidia/tensorrt-llm/release:<version>",
            "NVIDIA's published release image on NGC; replace <version> with a "
            "release tag from the NGC catalog. Not verified on this estate. Confirm "
            "the tag you choose is published for your architecture (arm64 for a DGX "
            "Spark), then pin it by digest.",
        ),
    )

    def validate_config(
        self, config: dict[str, Any], *, approved_options: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        try:
            validated = TRTLLMConfig.model_validate(
                config, context={"approved_options": frozenset(approved_options)}
            )
        except ValidationError as exc:
            accepted = ", ".join(sorted(TRTLLMConfig.model_fields))
            raise ValueError(
                f"invalid trtllm config: {exc}\naccepted trtllm keys: {accepted}\n"
                "a trtllm-serve flag this product does not model can be passed through "
                "extra_args, where it is forwarded unvalidated"
            ) from exc
        dumped = validated.model_dump(exclude_none=True)
        if not dumped.get("extra_args"):
            dumped.pop("extra_args", None)
        if dumped.get("telemetry") is False:
            dumped.pop("telemetry")
        return dumped

    def container_entrypoint(
        self,
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str] | None:
        """``trtllm-serve``, stated rather than inherited from the image.

        The release image is a general environment, not a server image, so its
        own entrypoint does not start one.
        """
        return ["trtllm-serve"]

    def inference_credential_env(self, secret: str) -> dict[str, Any]:  # noqa: ARG002
        """No mechanism: ``trtllm-serve`` has no inference API key.

        Its only key (``rl_control_api_key`` in ``serve/openai_server.py``)
        guards reinforcement-learning control endpoints, not inference. A
        deployment serves unauthenticated, and a key bound to one refuses the
        start rather than being silently ignored.
        """
        return {}

    def readiness_probe(self) -> ReadinessProbe:
        """``/health``, which reflects the engine and not just the HTTP server.

        ``launch_server`` builds the model before it creates the HTTP server,
        so nothing answers until the weights are loaded, and ``/health``
        returns non-200 once the engine reports a fatal error. ``/v1/models``
        would keep answering 200 after that, because it reports a fixed name.
        """
        return ReadinessProbe(path="/health", expect_status=200)

    def config_advisories(
        self,
        config: dict[str, Any],
        platform_facts: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        """Advisory only: never a reason a request is refused."""
        validated = TRTLLMConfig.model_validate(config, context=validated_context())
        notes: list[str] = []
        fraction = validated.kv_cache_free_gpu_memory_fraction
        if fraction is None or fraction > 0.8:
            shown = fraction if fraction is not None else _DEFAULT_KV_FRACTION
            notes.append(
                f"kv_cache_free_gpu_memory_fraction is {shown}"
                f"{' (TensorRT-LLM default)' if fraction is None else ''}: the KV cache "
                f"takes that share of the memory left after the weights load. On a "
                f"unified-memory node such as a DGX Spark that memory is also the "
                f"host's, so a high share can starve the operating system and anything "
                f"else running there. Consider 0.5-0.7 on such nodes"
            )
        if validated.telemetry:
            notes.append(
                "telemetry is on: TensorRT-LLM will send anonymous usage reports to "
                "NVIDIA (events.gfe.nvidia.com)"
            )
        return notes

    def container_requirements(
        self,
        config: dict[str, Any],  # noqa: ARG002
        position: object | None = None,  # noqa: ARG002
    ) -> ContainerRequirements:
        """TensorRT-LLM's port, plus the container limits its own docs give.

        NVIDIA's ``docker run`` examples pass ``--ipc=host --ulimit memlock=-1
        --ulimit stack=67108864``. Shared memory is granted as a sized
        ``/dev/shm`` rather than the host's IPC namespace -- the same need,
        without exposing the host's IPC objects -- and the two limits as given.
        """
        return ContainerRequirements(
            api_port=_DEFAULT_API_PORT,
            shm_size=8 * 1024**3,
            ulimits={"memlock": (-1, -1), "stack": (67108864, 67108864)},
        )

    def schema_probe(self) -> SchemaProbe | None:
        """None: no probe written yet, so the surface is reported as unknown."""
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
        """Nothing to refuse: this adapter never forms a group across nodes."""

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: object | None = None,  # noqa: ARG002
        endpoint_port: int | None = None,  # noqa: ARG002
    ) -> list[str]:
        validated = TRTLLMConfig.model_validate(config, context=validated_context())
        # The model is ``serve``'s positional argument; ``trtllm-serve`` routes
        # an argument that is not a subcommand name to ``serve``.
        args = [model_path]
        # ``--host`` defaults to ``localhost``, which inside a container serves
        # no one outside it.
        args += ["--host", "0.0.0.0"]  # noqa: S104 - inside the container namespace
        modelled: dict[str, Any] = {
            "tp_size": validated.tp_size,
            "pp_size": validated.pp_size,
            "ep_size": validated.ep_size,
            "max_batch_size": validated.max_batch_size,
            "max_num_tokens": validated.max_num_tokens,
            "max_seq_len": validated.max_seq_len,
            "kv_cache_free_gpu_memory_fraction": validated.kv_cache_free_gpu_memory_fraction,
            "kv_cache_dtype": validated.kv_cache_dtype,
            "reasoning_parser": validated.reasoning_parser,
            "tool_parser": validated.tool_parser,
            "served_model_name": validated.served_model_name,
            "chat_template": validated.chat_template,
        }
        for flag, value in modelled.items():
            if value is not None:
                args += [f"--{flag}", str(value)]
        if validated.enable_chunked_prefill:
            args.append("--enable_chunked_prefill")
        if validated.trust_remote_code:
            args.append("--trust_remote_code")
        args.append("--telemetry" if validated.telemetry else "--no-telemetry")
        # Forwarded last, in the spelling the operator wrote. TensorRT-LLM's
        # flags mix ``--max_batch_size`` and ``--generation-config`` and click
        # does not treat ``_`` and ``-`` as equal, so normalizing either way
        # would break half of them. Authorization above compares normalized.
        for key, value in validated.extra_args.items():
            flag = f"--{key.strip().lstrip('-')}"
            args += [flag] if value is True else [flag, str(value)]
        return args

    def declared_memory_fraction(
        self,
        config: dict[str, Any],  # noqa: ARG002
    ) -> float | None:
        """None: TensorRT-LLM's fraction is of memory left free, not of the device.

        The preflight this feeds compares a share of the device's *total*
        against what is free. TensorRT-LLM sizes its KV cache from whatever is
        free after the weights load, so reporting its fraction there would
        refuse starts it would have fitted. The unified-memory risk is
        raised as an advisory instead.
        """
        return None

    def validate_model_path(self, path: Path) -> None:
        """Raise when the tree is not a Hugging Face checkpoint.

        The PyTorch backend loads a transformers-format directory: a parseable
        ``config.json`` and, when sharded, every shard its index names.
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
            return
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
        except ValueError as exc:
            raise ValueError(
                f"model.safetensors.index.json at {index} is not valid JSON: {exc}"
            ) from exc
        for shard in sorted(set(weight_map.values())):
            shard_path = path / shard
            if not shard_path.is_file() or shard_path.stat().st_size == 0:
                raise ValueError(
                    f"model directory is missing or has an empty shard {shard!r}: {path}"
                )
