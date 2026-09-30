"""llama.cpp runtime adapter.

The **counter-example** that keeps the runtime seam honest. vLLM declares
``supports_distributed = True``; llama.cpp declares ``False``. Because a real
adapter says no, the multi-node rejection is exercised by a shipped runtime
rather than by a test double — the difference between a seam that is tested and
a seam that is merely asserted.

That matters for distribution in particular. The product coordinates and the runtime
distributes; we implement no distribution of our own. A runtime that *cannot*
distribute is what proves there is no hidden fallback path quietly doing it for
them: with llama.cpp selected, a two-node request has nowhere to go but a
refusal.

Like every adapter it owns its own config schema — there is no
cross-runtime config union, so llama.cpp's ``n_gpu_layers`` and vLLM's
``tensor_parallel_size`` never meet in a shared model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg, normalize_flag
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    ReadinessProbe,
    SchemaProbe,
    SuggestedImage,
)

# Flags an operator may never set through ``extra_args``, on the same grounds
# as the vLLM adapter: each is owned by a part of the product that would
# otherwise be silently overridden.
# ``name-00001-of-00003.gguf`` -- llama.cpp's split naming.
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)")

# Named once so the forbidden ``api-key`` passthrough above and the mechanism
# below cannot drift apart, the same way vLLM names its own.
_CREDENTIAL_ENV_VAR = "LLAMA_API_KEY"

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    "model": "the model path comes from the acquired model, not from config",
    "api-key": "the inference credential is delivered as environment, never argv",
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
}


class LlamaCppConfig(BaseModel):
    """llama.cpp's runtime-specific config schema.

    The model path is derived at launch rather than taken from the operator, so
    a deployment cannot point at weights it has not acquired. The rest is a
    small, documented slice of ``llama-server``'s surface.
    """

    model_config = ConfigDict(extra="forbid")

    # Which file inside the acquired model directory to load. llama.cpp's
    # ``--model`` names a *file*; the deployment records a *directory*. When the
    # directory holds exactly one .gguf (or one sharded set) the adapter
    # resolves it, and this stays unset. It is required when the directory holds
    # several unrelated .gguf files -- a GGUF repository routinely ships twenty
    # quantizations of the same weights, and guessing which one an operator
    # meant is precisely the kind of silent mismatch this product exists to
    # prevent.
    model_file: str | None = Field(default=None, min_length=1)
    # Multimodal projector, loaded alongside the weights for a vision model.
    # Named the same way and resolved against the same directory.
    mmproj_file: str | None = Field(default=None, min_length=1)
    # -1 offloads every layer it can; 0 keeps the model on CPU.
    n_gpu_layers: int = Field(default=-1, ge=-1)
    ctx_size: int | None = Field(default=None, gt=0)
    threads: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    parallel: int | None = Field(default=None, ge=1)
    flash_attn: bool | None = None

    # Flags this product does not model, forwarded verbatim.
    #
    # llama.cpp had the same closed allowlist vLLM did -- six fields and
    # ``extra="forbid"`` -- and it was found by asking whether the fix applied
    # here too rather than by anyone hitting it. The reasoning is identical: a
    # schema that cannot express what an operator needs does not stop them, it
    # sends them around the management plane, after which the coordinator's
    # records describe a container nobody is managing.
    #
    # Recorded as unvalidated, because it is. This product does not have
    # llama-server's real surface, and llama.cpp moves faster than this file.
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("extra_args")
    @classmethod
    def _check_extra_args(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        """Refuse the passthrough keys that would override the product itself.

        The same shared rule vLLM uses. This adapter previously held its own
        copy, which is why it held its own copy of the alternate-spelling defect
        too — ``model=/tmp/other`` and ``no-flash-attn`` both reproduced here
        before the rule was shared.
        """
        for key in value:
            authorize_extra_arg(
                key,
                forbidden=_FORBIDDEN_EXTRA_ARGS,
                model_fields=cls.model_fields,
            )
        return value


def _ggufs(directory: Path) -> list[Path]:
    """Every usable .gguf below ``directory``, sorted.

    The single place both the resolver and the validator ask what weights a
    tree holds. They are one decision, and this file has already paid once for
    stating it twice: when only the validator recursed, a nested tree passed
    validation and llama-server was handed the *directory* as ``--model``.

    **Nonempty**, because a zero-byte .gguf is an interrupted download rather
    than a model. ``huggingface.verify`` refuses one at acquisition, but a
    counted-but-empty candidate is worse than an uncounted one at this end: it
    is resolvable, so it would be selected and handed to llama-server, which
    fails with no classified cause -- and beside real weights it turns one
    model into a refusal naming a file that holds nothing.
    """
    return sorted(
        candidate
        for candidate in directory.rglob("*.gguf")
        if candidate.is_file() and candidate.stat().st_size > 0
    )


def _resolve_gguf(model_path: str, model_file: str | None) -> str:
    """Return the .gguf file ``--model`` should name, inside ``model_path``.

    ``build_launch_args`` was handed the deployment's model *directory* and
    passed it straight to ``--model``, which wants a file. The adapter's own
    ``validate_model_path`` said so in its docstring -- "the container then
    names the file" -- but nothing named it, and llama-server was handed a
    directory it cannot load. It never surfaced because this estate runs vLLM
    and no test rendered llama.cpp argv against a real tree.

    An explicit ``model_file`` wins. Otherwise a directory holding exactly one
    .gguf resolves to it, and a sharded set resolves to its first shard, which
    is the one llama.cpp is given (it finds the rest itself). Several unrelated
    .gguf files is refused by name rather than guessed at.

    A directory that does not exist falls through to the path as given: the
    launch path calls ``validate_model_path`` first, which refuses a missing or
    gguf-less directory with a better message than this function could. That
    fallback is why this function and ``validate_model_path`` must search the
    same way: if validation recursed and this did not, a nested tree would pass
    validation and llama-server would be handed the *directory* as ``--model``,
    which is a worse failure than the one validation exists to give.

    **Searched recursively**, because a file selection keeps the repository's
    own directories: ``UD-IQ1_S/*.gguf`` stages as ``UD-IQ1_S/`` below the model
    root, not as loose files in it (the acquisition side of this is
    ``huggingface.verify``). A quantization directory is the ordinary shape of
    the repositories this estate acquires, not an exotic one.
    """
    directory = Path(model_path)
    if model_file is not None:
        return str(directory / model_file)
    try:
        ggufs = _ggufs(directory)
    except OSError:
        return model_path
    if not ggufs:
        return model_path
    if len(ggufs) == 1:
        return str(ggufs[0])
    # A sharded model is one model: llama.cpp is given shard 1 and opens the
    # rest by name. Recognised before the ambiguity check so a legitimately
    # split model is not reported as a choice the operator has to make.
    #
    # Grouped by **directory and stem together**, which matters only once the
    # search recurses. Two quantizations of the same weights routinely carry
    # byte-identical filenames -- ``UD-IQ1_S/model-00001-of-00003.gguf`` beside
    # ``UD-Q8_K_XL/model-00001-of-00002.gguf`` -- so a stem-only grouping sees
    # one stem, concludes "one sharded model", and silently serves whichever
    # quantization sorts first. Picking a quantization for the operator is the
    # exact silent mismatch ``model_file`` exists to prevent.
    groups: dict[tuple[Path, str], list[Path]] = {}
    for gguf in ggufs:
        groups.setdefault((gguf.parent, _SHARD_RE.sub("", gguf.name)), []).append(gguf)
    if len(groups) == 1:
        members = next(iter(groups.values()))
        if all(_SHARD_RE.search(member.name) for member in members):
            return str(members[0])
    # Relative to the model root, because that is the value ``model_file`` takes:
    # the message names the choices in the form the operator has to write back.
    names = ", ".join(str(gguf.relative_to(directory)) for gguf in ggufs)
    raise ValueError(
        f"model directory {model_path} holds {len(ggufs)} .gguf files and "
        f"config names no model_file: {names}. Set model_file to the one this "
        f"deployment serves -- a GGUF repository commonly ships many "
        f"quantizations of the same weights, and the choice is the operator's"
    )


class LlamaCppAdapter:
    """A ``RuntimeAdapter`` for llama.cpp (``supports_distributed = False``).

    The capability declaration is the point of this class. Everything else is
    ordinary argument translation.
    """

    runtime_type = "llamacpp"
    supports_distributed = False
    # llama.cpp publishes its own server image; this names that project's
    # published reference rather than anything verified on this estate (the
    # estate runs vLLM). A suggestion, not a guarantee: llama.cpp releases far
    # more often than this product tracks, and which build variant an operator
    # wants -- ``light``, a CUDA build, a specific commit -- is their choice to
    # make and test (the no-scheduling rule).
    suggested_images = (
        SuggestedImage(
            "ghcr.io/ggerganov/llama.cpp:light",
            "llama.cpp project's published server image; not verified on this "
            "estate. llama.cpp releases more often than this product tracks -- "
            "pin a tag you have tested rather than a moving label, and prefer "
            "`image list` once pulled.",
        ),
    )

    def validate_config(
        self,
        config: dict[str, Any],
        *,
        approved_options: frozenset[str] = frozenset(),  # noqa: ARG002 - port parity, see below
    ) -> dict[str, Any]:
        # ``approved_options`` is accepted and ignored: llama.cpp models no
        # option in an approvable class, so there is nothing here an approval
        # could open. Accepting it keeps every adapter substitutable for the
        # port, which is what stopped that defect from recurring.
        try:
            validated = LlamaCppConfig(**config)
        except ValidationError as exc:
            # Naming the accepted keys is the difference between an error a
            # user can act on and one that sends them back to guessing: the
            # bare Pydantic message reports the rejected key and a link to
            # Pydantic's docs, neither of which is about this product.
            accepted = ", ".join(sorted(LlamaCppConfig.model_fields))
            raise ValueError(
                f"invalid llamacpp config: {exc}\naccepted llamacpp keys: {accepted}\n"
                "a llama.cpp flag this product does not model can be passed through "
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
        """The image's own entrypoint is used."""
        return None

    def inference_credential_env(self, secret: str) -> dict[str, Any]:
        """``llama-server`` reads ``LLAMA_API_KEY`` as an alternative to ``--api-key``.

        This adapter declared **no** mechanism until now, and the reasoning it
        gave was right at the time: ``--api-key`` puts the secret on the
        process command line where any local user reads it from ``ps``, and an
        environment-based path was needed before a credential could be honoured
        here. A later finding then made that refusal enforced rather than merely
        documented -- a bound credential now refuses the start instead of
        publishing an open endpoint -- and explicitly left implementing a real
        mechanism as follow-up needing hardware to verify.

        This is that follow-up, with the verification the finding asked for.
        ``llama-server --help``, from the image this estate runs
        (``local/llamacpp-qwen38-flash-next:pr27742-eaf9376``), states::

            --api-key KEY   API key to use for authentication, ...
                            (env: LLAMA_API_KEY)

        so the variable is read from the environment by llama.cpp itself. That
        is checked evidence from the running artifact, not an inference from
        upstream documentation -- which is the distinction the finding drew
        between a mechanism worth declaring and a guess worth refusing.

        Delivered as container environment, never argv: ``api-key`` stays in
        ``_FORBIDDEN_EXTRA_ARGS``, so the argv spelling an operator might reach
        for is still refused, and that entry's stated reason -- "delivered as
        environment, never argv" -- is now literally true.
        """
        return {_CREDENTIAL_ENV_VAR: secret}

    def readiness_probe(self) -> ReadinessProbe:
        """``llama-server`` serves an OpenAI-compatible ``/v1/models``.

        Its ``/health`` returns 503 while the model loads and 200 afterwards, so
        either would work here. ``/v1/models`` is chosen for the same reason as
        vLLM's: it answers only when a model is actually being served, and using
        one probe shape across both v1 runtimes keeps the observed-state field
        meaning the same thing regardless of runtime.
        """
        return ReadinessProbe(path="/v1/models", expect_status=200)

    def config_advisories(
        self,
        config: dict[str, Any],  # noqa: ARG002
        platform_facts: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        """llama.cpp has no such interaction to warn about."""
        return []

    def container_requirements(
        self,
        config: dict[str, Any],  # noqa: ARG002
        position: object | None = None,  # noqa: ARG002
    ) -> ContainerRequirements:
        """llama.cpp needs nothing beyond the default container, but its own port.

        Stated rather than inherited. It is single-process and does not use
        shared memory between workers, so the vLLM requirement genuinely does
        not apply -- and saying so is what makes this a real counter-example to
        the seam rather than an adapter that merely forgot to implement it.

        ``api_port`` is the exception to "needs nothing": ``llama-server``
        listens on 8080, and the agent published every deployment's endpoint to
        8000 because that is vLLM's port. A llama.cpp deployment therefore
        started, reported running, and mapped its recorded endpoint to a
        container port nothing was listening on -- the
        runtime-agnostic layer holding one runtime's fact, which is the mistake
        this seam exists to prevent.
        """
        return ContainerRequirements(api_port=8080)

    def schema_probe(self) -> SchemaProbe | None:
        """No probe declared for llama.cpp.

        ``llama-server`` reports its surface as ``--help`` text, and recovering
        structure from generated text is the drift this mechanism exists to
        avoid. Returning ``None`` records the surface as **unknown**, which is
        true, rather than empty, which would say llama.cpp accepts nothing.

        The second shipped runtime earns its keep here as it does elsewhere: an
        adapter that cannot probe is a case the design has to handle, and this
        one proves the product handles it rather than assuming every runtime is
        as introspectable as vLLM.
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
        """Nothing to refuse: this runtime never forms a group.

        ``supports_distributed = False``, so a deployment naming more than one
        node is already refused with ``runtime_not_distributed`` before anything
        reaches here. Implemented rather than omitted so the adapter satisfies
        the whole port -- the second shipped runtime is what keeps the seam
        honest, and a seam one adapter silently opts out of is not a seam.
        """

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: object | None = None,  # noqa: ARG002
        endpoint_port: int | None = None,  # noqa: ARG002
    ) -> list[str]:
        validated = LlamaCppConfig(**config)
        args = ["--model", _resolve_gguf(model_path, validated.model_file)]
        # ``llama-server`` binds **127.0.0.1** by default. vLLM binds 0.0.0.0,
        # which is why the container port map has always worked and why this
        # was never noticed: under bridge networking the engine maps host
        # <endpoint port> to container 8080, and a server listening only on
        # loopback inside its own namespace is reachable from nothing outside
        # it. The container came up, reported healthy, logged "model loaded",
        # and served no one.
        #
        # This is the mirror of an earlier case -- there the runtime bound a
        # port nobody published; here it publishes a port nobody bound.
        #
        # Not a violation of the reserved ``host`` key. That key withholds the
        # *deployment's* declared address, which is a host-side fact the engine
        # translates. This is the container-side interface, which is the
        # adapter's own business and is not configurable: an
        # operator who could set it could only break the mapping.
        args += ["--host", "0.0.0.0"]  # noqa: S104 - inside the container namespace
        if validated.mmproj_file is not None:
            args += ["--mmproj", str(Path(model_path) / validated.mmproj_file)]
        # -1 is llama-server's own default meaning; passing it adds nothing.
        if validated.n_gpu_layers != -1:
            args += ["--n-gpu-layers", str(validated.n_gpu_layers)]
        if validated.ctx_size is not None:
            args += ["--ctx-size", str(validated.ctx_size)]
        if validated.threads is not None:
            args += ["--threads", str(validated.threads)]
        if validated.batch_size is not None:
            args += ["--batch-size", str(validated.batch_size)]
        if validated.parallel is not None:
            args += ["--parallel", str(validated.parallel)]
        if validated.flash_attn is not None:
            # ``--flash-attn`` takes a value in current llama.cpp
            # (``on|off|auto``, default ``auto``). Emitted as a bare flag it
            # swallows the next token: with ``--flash-attn --jinja`` the server
            # exits with
            #
            #   error: unknown value for --flash-attn: '--jinja'
            #
            # which names neither the option that is wrong nor the one that
            # disappeared. Emitting the value explicitly also makes ``false``
            # expressible, which a bare flag could only ever say by omission --
            # and omission means "auto" here, not "off".
            args += ["--flash-attn", "on" if validated.flash_attn else "off"]
        # Forwarded last and verbatim, so reading the generated argv shows
        # plainly where this product's opinions end and the operator's begin.
        for key, value in validated.extra_args.items():
            flag = normalize_flag(key)
            if isinstance(value, bool):
                args += [f"--{flag}"] if value else [f"--no-{flag}"]
            else:
                args += [f"--{flag}", str(value)]
        return args

    def declared_memory_fraction(
        self,
        config: dict[str, Any],  # noqa: ARG002
    ) -> float | None:
        """llama.cpp declares no memory budget.

        It sizes itself from the model and ``ctx_size`` and allocates as it
        goes, so there is no fraction to report. ``None`` says exactly that
        rather than inventing a number the runtime never accepted.
        """
        return None

    def validate_model_path(self, path: Path) -> None:
        """Raise when the on-disk model tree is not usable by llama.cpp.

        llama.cpp loads a single ``.gguf`` file (its ``--model`` argument points
        at a file, not a directory, but the deployment records a directory). The
        directory must be a directory and contain at least one ``.gguf``; the
        container then names the file. An empty or gguf-less directory would
        otherwise "succeed" at launch and exit with no classified cause.

        **Searched recursively, for the same reason and in the same way as**
        ``_resolve_gguf``. A selection such as ``UD-IQ1_S/*.gguf`` stages the
        repository's own directory below the model root, so a root-only search
        refused a tree that had downloaded correctly and held exactly the
        weights the deployment asked for. The two searches are one decision
        stated twice; changing either alone reintroduces the mismatch this
        pairing exists to prevent.
        """
        if not path.is_dir():
            raise ValueError(f"model path is not a directory: {path}")
        ggufs = _ggufs(path)
        if not ggufs:
            raise ValueError(f"model directory has no .gguf file: {path}")
