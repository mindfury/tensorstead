"""ExLlama runtime adapter, served through TabbyAPI.

ExLlama is a *library*, not a server: ``exllamav3`` loads EXL3-quantized (and
unquantized) weights and runs them, and nothing in it listens on a port. The
server its own ecosystem ships is TabbyAPI, an OpenAI-compatible HTTP front end
whose only backend is ExLlamaV3. So this adapter's runtime is named for what an
operator is choosing -- the ExLlama format and engine -- and its argv,
config vocabulary, port and image are TabbyAPI's, because that is what runs.

Everything below about TabbyAPI was read from its source at commit ``be74bf0``
(2026-09-28), not inferred from documentation. TabbyAPI moves quickly; where a
fact here is load-bearing, the docstring says which file it came from so the
next person can re-check it rather than trust it.

Like every adapter it owns its own config schema, in TabbyAPI's own vocabulary
(``max_seq_len``, ``cache_mode``, ``gpu_split``) -- there is no translation from
vLLM's or llama.cpp's spelling of the same ideas.

``supports_distributed = False``. TabbyAPI splits a model across the GPUs of
one host (``gpu_split``, ``tensor_parallel``) but forms no group across hosts,
so a multi-node request is refused by the shared ``runtime_not_distributed``
rule before it reaches this file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg, normalize_flag
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    ReadinessProbe,
    SchemaProbe,
    SuggestedImage,
)

# TabbyAPI's own default listener (``NetworkConfig.port`` in
# ``common/config_models.py``, and ``EXPOSE 5000`` in its Dockerfile).
_DEFAULT_API_PORT = 5000

# Quantization methods TabbyAPI refuses at load time
# (``common/model.py: validate_backend``): both need the ExLlamaV2 backend, which
# TabbyAPI no longer ships. Refused here, before a container exists, so an EXL2
# repository fails with a named reason instead of an exited container.
_UNSUPPORTED_QUANT_METHODS = frozenset({"exl2", "gptq"})

# ``FP16``/``Q8``/``Q6``/``Q4``, or a ``k_bits,v_bits`` pair from 2 to 8 --
# TabbyAPI's ``CACHE_TYPE``.
_CACHE_MODE_RE = re.compile(r"^(FP16|Q8|Q6|Q4|[2-8]\s*,\s*[2-8])$")

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    "model-dir": "the model path comes from the acquired model, not from config",
    "model-name": "the model path comes from the acquired model, not from config",
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
    "disable-auth": "authentication is set by this product, see inference_credential_env",
    # ``--config`` does not add to the command line, it replaces it: TabbyAPI's
    # ``_from_args`` returns the named file's contents early and drops every
    # other argument, so one key would silently discard everything above.
    "config": "an override file would replace every argument this product renders",
}


class ExLlamaConfig(BaseModel):
    """ExLlama's runtime-specific config schema, in TabbyAPI's vocabulary.

    A small slice of TabbyAPI's ``model`` section. The model path is derived at
    launch, never taken from the operator, so a deployment cannot point at
    weights it has not acquired. Anything not modelled goes through
    ``extra_args``.
    """

    model_config = ConfigDict(extra="forbid")

    # Which directory inside the acquired model to serve, when the repository
    # keeps quantizations in subdirectories (``4.0bpw/``, ``6.0bpw/``). Unset,
    # the adapter serves the one directory that holds weights and refuses to
    # guess between several -- the same rule as llama.cpp's ``model_file``.
    model_subdir: str | None = Field(default=None, min_length=1)
    # -1 reads the context length from the model's own config.json.
    max_seq_len: int | None = Field(default=None, ge=-1)
    cache_size: int | None = Field(default=None, gt=0, multiple_of=256)
    cache_mode: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    max_batch_size: int | None = Field(default=None, ge=1)
    tensor_parallel: bool | None = None
    gpu_split_auto: bool | None = None
    # GB per GPU, in device order.
    gpu_split: list[float] | None = Field(default=None, min_length=1)
    prompt_template: str | None = Field(default=None, min_length=1)
    vision: bool | None = None

    # Flags this product does not model, forwarded verbatim.
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("model_subdir")
    @classmethod
    def _check_model_subdir(cls, value: str | None) -> str | None:
        """Keep the selection inside the acquired model.

        The resolved directory becomes ``--model-name`` beneath a
        ``--model-dir`` this product chose; ``..`` or an absolute path would
        turn that into a way to serve something else on the node.
        """
        if value is None:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"model_subdir {value!r} must be a relative path inside the acquired model"
            )
        return value

    @field_validator("max_seq_len")
    @classmethod
    def _check_max_seq_len(cls, value: int | None) -> int | None:
        if value == 0:
            raise ValueError("max_seq_len must be positive, or -1 to read it from the model")
        return value

    @field_validator("cache_mode")
    @classmethod
    def _check_cache_mode(cls, value: str | None) -> str | None:
        if value is not None and not _CACHE_MODE_RE.match(value):
            raise ValueError(
                f"cache_mode {value!r} is not one TabbyAPI accepts: use FP16, Q8, Q6, Q4, "
                f"or a k_bits,v_bits pair from 2 to 8 such as '8,8'"
            )
        return value

    @field_validator("gpu_split")
    @classmethod
    def _check_gpu_split(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and any(size <= 0 for size in value):
            raise ValueError("gpu_split sizes are GB per GPU and must be positive")
        return value

    @field_validator("extra_args")
    @classmethod
    def _check_extra_args(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        """The shared passthrough rule, plus one TabbyAPI makes necessary.

        TabbyAPI builds a stock ``argparse`` parser, and ``argparse`` accepts
        any unambiguous *prefix* of a long option. So ``--model-n`` reaches
        ``--model-name`` and ``--disable-a`` reaches ``--disable-auth``, and the
        shared rule -- which compares whole names -- passes both. A key that
        abbreviates a reserved or modelled option is refused for the same
        reason the option itself is.
        """
        reserved = set(_FORBIDDEN_EXTRA_ARGS) | {
            name.replace("_", "-") for name in cls.model_fields if name != "extra_args"
        }
        for key in value:
            authorize_extra_arg(
                key,
                forbidden=_FORBIDDEN_EXTRA_ARGS,
                model_fields=cls.model_fields,
            )
            flag = normalize_flag(key)
            reached = sorted(name for name in reserved if name != flag and name.startswith(flag))
            if reached:
                raise ValueError(
                    f"extra_args may not set {key!r}: TabbyAPI reads an abbreviated flag as "
                    f"the full option, and this one abbreviates {', '.join(reached)}"
                )
        return value


def _weight_dirs(root: Path) -> list[Path]:
    """Every directory at or below ``root`` that holds a loadable model, sorted.

    Loadable means a ``config.json`` beside at least one nonempty
    ``.safetensors`` -- what ExLlamaV3 reads. The single place the validator
    and the resolver ask the question, for the reason llama.cpp's ``_ggufs``
    gives: two searches that can disagree eventually do.
    """
    found: list[Path] = []
    for config in sorted(root.rglob("config.json")):
        directory = config.parent
        if any(
            shard.is_file() and shard.stat().st_size > 0
            for shard in directory.glob("*.safetensors")
        ):
            found.append(directory)
    return found


def _quant_method(directory: Path) -> str | None:
    """The ``quantization_config.quant_method`` a model declares, if any.

    Unreadable or absent reads as ``None`` -- an unquantized model, which
    ExLlamaV3 also loads. A malformed ``config.json`` is left for TabbyAPI to
    report in its own words rather than guessed at here.
    """
    try:
        parsed = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    quantization = parsed.get("quantization_config") if isinstance(parsed, dict) else None
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    return method.lower() if isinstance(method, str) else None


def _refuse_unsupported_quant(directory: Path) -> None:
    method = _quant_method(directory)
    if method in _UNSUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"model at {directory} is quantized with {method!r}, which TabbyAPI no longer "
            f"loads (it needs the retired ExLlamaV2 backend). Acquire an EXL3 or "
            f"unquantized revision of this model instead"
        )


def _resolve_model_dir(model_path: str, model_subdir: str | None) -> Path:
    """The directory TabbyAPI should load, inside ``model_path``.

    An explicit ``model_subdir`` wins. Otherwise the root itself if it is a
    model, or the single nested directory that is one. Several is refused by
    name, because a quantization repository's subdirectories are different
    models and choosing one is the operator's decision.

    A root that does not exist falls through to itself: the launch path runs
    ``validate_model_path`` first, which refuses it with a better message.
    """
    root = Path(model_path)
    if model_subdir is not None:
        return root / model_subdir
    try:
        candidates = _weight_dirs(root)
    except OSError:
        return root
    if not candidates or root in candidates:
        return root
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(str(candidate.relative_to(root)) for candidate in candidates)
    raise ValueError(
        f"model directory {model_path} holds {len(candidates)} loadable models and config "
        f"names no model_subdir: {names}. Set model_subdir to the one this deployment serves"
    )


def _render(value: str | int | float | bool) -> str:
    """One value as TabbyAPI's parser expects it.

    Every TabbyAPI option takes a value -- its parser declares no bare
    switches -- so a boolean is spelled ``true``/``false`` and left for
    Pydantic to read. Rendering ``--vision`` alone, as the llama.cpp adapter
    would, makes ``argparse`` swallow the next flag as its value.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class ExLlamaAdapter:
    """A ``RuntimeAdapter`` for ExLlama via TabbyAPI (``supports_distributed = False``)."""

    runtime_type = "exllama"
    supports_distributed = False
    suggested_images = (
        SuggestedImage(
            "ghcr.io/theroyallab/tabbyapi:latest",
            "TabbyAPI project's published image; not verified on this estate. "
            "x86_64 only: TabbyAPI pins exllamav3 wheels for x86_64 Linux and "
            "Windows, none for aarch64, so this image does not run on a DGX Spark "
            "or other ARM node -- build one there with `image build`. Pin a digest "
            "you have tested rather than a moving tag.",
        ),
    )

    def validate_config(
        self,
        config: dict[str, Any],
        *,
        approved_options: frozenset[str] = frozenset(),  # noqa: ARG002 - port parity
    ) -> dict[str, Any]:
        # Accepted and ignored, like llama.cpp: nothing modelled here is in an
        # approvable class.
        try:
            validated = ExLlamaConfig(**config)
        except ValidationError as exc:
            accepted = ", ".join(sorted(ExLlamaConfig.model_fields))
            raise ValueError(
                f"invalid exllama config: {exc}\naccepted exllama keys: {accepted}\n"
                "a TabbyAPI flag this product does not model can be passed through "
                "extra_args, where it is forwarded unvalidated"
            ) from exc
        dumped = validated.model_dump(exclude_none=True)
        if not dumped.get("extra_args"):
            dumped.pop("extra_args", None)
        return dumped

    def container_entrypoint(
        self,
        config: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str] | None:
        """``python3 main.py``, stated rather than inherited.

        TabbyAPI's image sets ``ENTRYPOINT ["python3"]`` and puts ``main.py`` in
        ``CMD``. The engine replaces ``CMD`` with this adapter's argv, so relying
        on the image would hand ``python3`` a list of flags and no script.
        """
        return ["python3", "main.py"]

    def inference_credential_env(self, secret: str) -> dict[str, Any]:  # noqa: ARG002
        """No mechanism: TabbyAPI cannot be handed a key without logging it.

        Read from ``common/auth.py``. TabbyAPI takes keys only from an
        ``api_tokens.yml`` in its working directory -- no flag, no environment
        variable -- and on every start logs them in clear::

            Your API key is: <key>
            Your admin key is: <key>

        Writing the credential into that file would therefore copy it into the
        container log, which ``deployment runtime`` returns verbatim and
        unredacted. That is a worse outcome than declaring none, so this
        declares none: a bound credential refuses the start
        (``credential_not_enforceable``), and an unbound deployment runs with
        authentication off and reports ``endpoint_authenticated: false``.
        """
        return {}

    def readiness_probe(self) -> ReadinessProbe:
        """``/v1/model``, not ``/v1/models``.

        The other adapters use ``/v1/models`` because it answers only once a
        model is served. TabbyAPI's does not: with authentication off every
        caller is treated as admin, and an admin's ``/v1/models`` lists the
        model *directory* whether or not anything is loaded
        (``endpoints/core/router.py``). ``/v1/model`` returns the loaded model,
        and 503 when there is none.
        """
        return ReadinessProbe(path="/v1/model", expect_status=200)

    def config_advisories(
        self,
        config: dict[str, Any],
        platform_facts: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        """Advisory only: never a reason a request is refused."""
        validated = ExLlamaConfig(**config)
        notes = [
            "TabbyAPI runs with authentication disabled under this product, and that also "
            "opens its admin endpoints: anyone who can reach this endpoint can unload the "
            "model, load another, or download one from Hugging Face into the container "
            "(/v1/model/load, /v1/model/unload, /v1/download). Expose it only to trusted "
            "clients"
        ]
        if validated.tensor_parallel and validated.gpu_split_auto is not None:
            notes.append(
                "gpu_split_auto is set alongside tensor_parallel; TabbyAPI ignores "
                "gpu_split_auto when tensor_parallel is on and splits automatically unless "
                "gpu_split is given"
            )
        return notes

    def container_requirements(
        self,
        config: dict[str, Any],  # noqa: ARG002
        position: object | None = None,  # noqa: ARG002
    ) -> ContainerRequirements:
        """TabbyAPI's port, plus the shared memory its own compose file asks for.

        TabbyAPI's ``docker-compose.yml`` sets ``shm_size: 8g`` and an unlimited
        ``memlock``, stating that ExLlamaV3 keeps tensor-parallel and CPU MoE
        offload buffers in ``/dev/shm`` and that Docker's 64 MiB default is far
        too small. Declared unconditionally because ``/dev/shm`` is a tmpfs
        whose size is an upper bound: memory is used only as it is written.
        """
        return ContainerRequirements(
            api_port=_DEFAULT_API_PORT,
            shm_size=8 * 1024**3,
            ulimits={"memlock": (-1, -1)},
        )

    def schema_probe(self) -> SchemaProbe | None:
        """None: no probe written yet, so the surface is reported as unknown.

        TabbyAPI's parser is generated from Pydantic models
        (``common/args.py``), so a probe is possible; nobody has written and
        verified one against a real image.
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
        """Nothing to refuse: this runtime never forms a group across nodes."""

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: object | None = None,  # noqa: ARG002
        endpoint_port: int | None = None,  # noqa: ARG002
    ) -> list[str]:
        validated = ExLlamaConfig(**config)
        served = _resolve_model_dir(model_path, validated.model_subdir)
        _refuse_unsupported_quant(served)
        # TabbyAPI loads ``model_dir / model_name`` and takes no path to a model
        # directly, so the served directory is split back into the two.
        args = ["--model-dir", str(served.parent), "--model-name", served.name]
        # TabbyAPI binds 127.0.0.1 by default; the image's own CMD passes
        # 0.0.0.0, and this argv replaces that CMD. Without it the container
        # serves only its own loopback -- the llama.cpp failure again.
        args += ["--host", "0.0.0.0"]  # noqa: S104 - inside the container namespace
        # See inference_credential_env: no key can be delivered safely, and with
        # auth on TabbyAPI invents keys nobody holds, so the endpoint would
        # refuse every client and every readiness probe.
        args += ["--disable-auth", "true"]
        modelled: dict[str, Any] = {
            "max-seq-len": validated.max_seq_len,
            "cache-size": validated.cache_size,
            "cache-mode": validated.cache_mode,
            "chunk-size": validated.chunk_size,
            "max-batch-size": validated.max_batch_size,
            "tensor-parallel": validated.tensor_parallel,
            "gpu-split-auto": validated.gpu_split_auto,
            "prompt-template": validated.prompt_template,
            "vision": validated.vision,
        }
        for flag, value in modelled.items():
            if value is not None:
                args += [f"--{flag}", _render(value)]
        if validated.gpu_split is not None:
            # ``nargs="+"``: one flag, then each GPU's share as its own token.
            args += ["--gpu-split", *(_render(size) for size in validated.gpu_split)]
        # Forwarded last and verbatim, as in every adapter.
        for key, value in validated.extra_args.items():
            args += [f"--{normalize_flag(key)}", _render(value)]
        return args

    def declared_memory_fraction(
        self,
        config: dict[str, Any],  # noqa: ARG002
    ) -> float | None:
        """None: TabbyAPI sizes by GB per GPU (``gpu_split``), not by a fraction."""
        return None

    def validate_model_path(self, path: Path) -> None:
        """Raise when the tree holds nothing TabbyAPI can load.

        Needs a ``config.json`` beside nonempty ``.safetensors``, at the root or
        in a subdirectory (``_weight_dirs``). EXL2 and GPTQ trees are refused
        by name when they are all the tree holds; with several candidates the
        chosen one is checked again at launch.
        """
        if not path.is_dir():
            raise ValueError(f"model path is not a directory: {path}")
        candidates = _weight_dirs(path)
        if not candidates:
            raise ValueError(
                f"model directory has no config.json beside .safetensors weights: {path}"
            )
        if all(_quant_method(c) in _UNSUPPORTED_QUANT_METHODS for c in candidates):
            _refuse_unsupported_quant(candidates[0])
