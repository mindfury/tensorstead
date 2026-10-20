"""TensorFold runtime adapter.

TensorFold (github.com/ashhart/TensorFold) serves models from one native
binary, ``tensorfold-native``, with its CUDA kernels compiled in: no Python,
no model download at startup, an OpenAI-compatible API. Its draft-and-verify
decoding is exact -- greedy output equals plain decoding -- which is the
property that earned it a place beside vLLM on this estate.

Measured on the estate's GB10s (2026-10-20, TensorFold 1.1.1, same stored
checkpoints, same requests, vLLM through its existing Tensorstead deployments):

- Nemotron 3.5 Lightning NVFP4: 140 tok/s single-request decode against vLLM
  MTP-1's 103; 8.2k against 7.0k tok/s prefill; loaded in 14 s against 176 s.
- Nemotron 3.5 Super VL NVFP4: 50 tok/s on one GB10, 61 tok/s across two,
  against vLLM DFlash-7's 48 on one.
- Nemotron 3.5 Bolt: 39 tok/s against vLLM DFlash's 81 -- TensorFold has no
  drafter for it, and an adapter does not get to pretend otherwise.

Like every adapter it owns its own config schema. The model path, the
listening address, the served name and the inference credential are the
product's; the rest of ``tensorfold serve``'s surface is either modelled here
or forwarded through ``extra_args``.

**Two nodes.** TensorFold forms a two-rank group for the families that support
one (GLM-5.3-Flash and Nemotron Super VL on CUDA): rank 1 holds a share of the
experts and serves rank 0's jobs; rank 0 serves the API. The ranks meet over
the collective library at a rendezvous address, and each hashes its checkpoint
copy, refusing to serve if the two differ by one bit. The product's ordering -- workers first, head
last -- is the order TensorFold's recipe asks for.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tensorstead.adapters.runtimes.extra_args import authorize_extra_arg, normalize_flag
from tensorstead.adapters.runtimes.option_policy import (
    POLICY_VERSION,
    authorize,
    normalize_dest,
)
from tensorstead.ports.runtime_adapter import (
    ContainerRequirements,
    NodePosition,
    ReadinessProbe,
    RuntimeOption,
    SchemaProbe,
    SuggestedImage,
)

# ``tensorfold-native`` reads comma-separated keys from this variable, as an
# alternative to ``--api-key``. Read from the binary itself
# (``capabilities --json`` lists it under ``env``), not inferred from docs.
_CREDENTIAL_ENV_VAR = "TENSORFOLD_API_KEY"

# The image recipe in docs/builds.md links the binary onto PATH under this name.
_BINARY = "tensorfold-native"

# ``tensorfold serve`` listens on 8080 unless told otherwise.
_DEFAULT_PORT = 8080

# The collective library's bootstrap port on rank 0. TensorFold opens a second
# TCP line to rank 0 on the next port, before the group forms, so both are taken
# under host networking.
_RENDEZVOUS_PORT = 29741

# Position variables for the two families with CUDA groups in TensorFold 1.1.1
# (the only such variables compiled into the binary): (index, count) per family.
# Each family reads only its own, so both are set: the adapter does not see
# which family a checkpoint is, and guessing from a directory name is the kind
# of silent mismatch the product exists to prevent.
_POSITION_VARIABLES = (
    ("TF_NEMOTRON_RANK", "TF_NEMOTRON_WORLD"),
    ("TF_GLM_RANK", "TF_GLM_WORLD"),
)
_RENDEZVOUS_VARIABLE = "NCCL_COMM_ID"
_INTERFACE_VARIABLE = "NCCL_SOCKET_IFNAME"

# TensorFold's groups are qualified for two GB10s; more is not covered.
_MAX_NODES = 2

_FORBIDDEN_EXTRA_ARGS: dict[str, str] = {
    "model": "the model path comes from the acquired model, not from config",
    "host": "the endpoint is declared on the deployment",
    "port": "the endpoint is declared on the deployment",
    "name": "set served_model_name, which the product checks for conflicts",
    "api-key": "the inference credential is delivered as environment, never argv",
    "api-key-file": "the inference credential is delivered as environment, never argv",
    "drafter": "name the drafter with drafter, which the agent mounts",
    # Loads a separately built native library named in a file -- code the
    # deployment record does not describe.
    "speed-up": "loads a native library from a file the deployment does not describe",
    # Server-run web research: outbound requests from the node on a client's
    # behalf, which no deployment field records.
    "cortheon": "makes the server issue outbound web requests on a client's behalf",
}

_ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


class TensorFoldConfig(BaseModel):
    """TensorFold's runtime-specific config schema.

    A small, documented slice of ``tensorfold serve`` (``serve --help``, 1.1.1);
    anything else goes through ``extra_args``.
    """

    model_config = ConfigDict(extra="forbid")

    # ``--name``: the model id clients send. Modelled rather than passed
    # through so the product's served-name conflict check sees it.
    served_model_name: str | None = Field(default=None, min_length=1)
    context: int | None = Field(default=None, gt=0)
    parallel: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    # The chat template's reasoning block. ``None`` keeps the template default.
    thinking: bool | None = None
    reasoning_effort: _ReasoningEffort | None = None
    # ``False`` renders ``--no-drafts``: plain decoding through the same engine.
    drafts: bool | None = None
    # MTP depth for the families with MTP heads (at most 31 for Super VL).
    mtp_drafts: int | None = Field(default=None, ge=0, le=31)
    # An external drafter's acquired model directory -- a second set of weights
    # the agent must mount, so modelled, as llama.cpp's ``draft_model`` is.
    drafter: str | None = Field(default=None, min_length=1)
    kv_cache: Literal["bf16", "fp8"] | None = None
    prompt_cache_gib: float | None = Field(default=None, ge=0)
    vision: bool | None = None
    # Both ranks of a group must run the same precision; they check it.
    precision: Literal["checkpoint", "full"] | None = None
    # The interface a two-node group's collective traffic rides. The agent resolves
    # the RDMA device and GID behind it on each host.
    distributed_interface: str | None = Field(default=None, min_length=1)
    extra_args: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("drafter")
    @classmethod
    def _check_drafter(cls, value: str | None) -> str | None:
        if value is None:
            return value
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"drafter {value!r} must be the absolute path of an acquired model "
                f"directory in the node's model store"
            )
        return value

    @field_validator("extra_args")
    @classmethod
    def _check_extra_args(
        cls, value: dict[str, str | int | float | bool]
    ) -> dict[str, str | int | float | bool]:
        for key in value:
            authorize_extra_arg(key, forbidden=_FORBIDDEN_EXTRA_ARGS, model_fields=cls.model_fields)
        return value


def _config(config: dict[str, Any] | None) -> TensorFoldConfig:
    return TensorFoldConfig(**(config or {}))


def _ipv4(address: str) -> str:
    """``address`` as an IPv4 literal, resolved on this node.

    TensorFold parses the rendezvous variable itself and accepts only an
    address -- ``spark-6d42.internal:29741`` failed both ranks with
    ``NoCommId`` on the estate, where ``10.100.89.2:29741`` formed the group.
    The product's peer addresses are the agents' endpoint hosts, which on this
    estate's nodes resolve to their fabric addresses; resolving here, on the
    agent, asks the node's own resolver, the one the ranks will use.
    """
    try:
        return str(ipaddress.IPv4Address(address))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(address, None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(
            f"rank 0's address {address!r} does not resolve to IPv4 on this node; "
            f"TensorFold's rendezvous needs an address"
        ) from exc
    return str(infos[0][4][0])


def _spans_nodes(position: NodePosition | None) -> TypeGuard[NodePosition]:
    return position is not None and position.node_count > 1


class TensorFoldAdapter:
    """A ``RuntimeAdapter`` for TensorFold's native server."""

    runtime_type = "tensorfold"
    supports_distributed = True
    suggested_images = (
        SuggestedImage(
            "local/tensorfold:1.1.1-cuda13",
            "Built on the estate from the tensorfold-1.1.1 build spec in docs/builds.md: the "
            "project's linux-aarch64 release, SHA-256 checked, on a CUDA 13 base "
            "with the collective library. "
            "TensorFold publishes no container image; the binary needs only glibc, the "
            "driver's libcuda and, for two nodes, the collective library.",
        ),
    )

    def validate_config(
        self,
        config: dict[str, Any],
        *,
        approved_options: frozenset[str] = frozenset(),  # noqa: ARG002 - port parity
    ) -> dict[str, Any]:
        # No modelled option is in an approvable class, so an approval has
        # nothing to open here; accepted for substitutability across adapters.
        try:
            validated = TensorFoldConfig(**config)
        except ValidationError as exc:
            accepted = ", ".join(sorted(TensorFoldConfig.model_fields))
            raise ValueError(
                f"invalid tensorfold config: {exc}\naccepted tensorfold keys: {accepted}\n"
                "a tensorfold serve flag this product does not model can be passed through "
                "extra_args, where it is forwarded unvalidated"
            ) from exc
        dumped = validated.model_dump(exclude_none=True)
        if not dumped.get("extra_args"):
            dumped.pop("extra_args", None)
        return dumped

    def container_entrypoint(self, config: dict[str, Any] | None = None) -> list[str]:  # noqa: ARG002
        """``tensorfold-native serve``; the argv names the model first."""
        return [_BINARY, "serve"]

    def inference_credential_env(self, secret: str) -> dict[str, Any]:
        """Delivered as ``TENSORFOLD_API_KEY``, never argv (``ps`` reads argv)."""
        return {_CREDENTIAL_ENV_VAR: secret}

    def readiness_probe(self) -> ReadinessProbe:
        """``/v1/models``: answered only once the model is loaded and serving.

        ``/health`` is public and would do as well; ``/v1/models`` keeps the
        probe's meaning the same as vLLM's and llama.cpp's, and the agent sends
        the inference key with it.
        """
        return ReadinessProbe(path="/v1/models", expect_status=200)

    def config_advisories(
        self,
        config: dict[str, Any],  # noqa: ARG002
        platform_facts: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        return []

    def container_requirements(
        self,
        config: dict[str, Any],
        position: NodePosition | None = None,
    ) -> ContainerRequirements:
        """One process; two nodes add collectives over RDMA, as vLLM's group does.

        Single node: bridge networking with the server's own port published --
        TensorFold accepts any ``Host`` on a ``0.0.0.0`` bind, measured on the
        estate, so the deployment's hostname needs no ``--allowed-host``.

        Two nodes: host networking (the collective bootstrap and TensorFold's own
        line), ``/dev/infiniband`` with ``IPC_LOCK`` and unlimited ``memlock``
        so RDMA can pin memory, and the rank variables. Rank 1 serves no API:
        it computes rank 0's expert shares, so it is not probed for inference.
        """
        validated = _config(config)
        environment = {"TENSORFOLD_NO_UPDATE_CHECK": "1"}
        references = [validated.drafter] if validated.drafter else []
        if not _spans_nodes(position):
            return ContainerRequirements(
                api_port=_DEFAULT_PORT, environment=environment, model_references=references
            )
        head = position.peer_addresses[0] if position.peer_addresses else position.self_address
        for index_variable, count_variable in _POSITION_VARIABLES:
            environment[index_variable] = str(position.node_index)
            environment[count_variable] = str(position.node_count)
        environment[_RENDEZVOUS_VARIABLE] = f"{_ipv4(head)}:{_RENDEZVOUS_PORT}"
        if validated.distributed_interface:
            environment[_INTERFACE_VARIABLE] = validated.distributed_interface
        return ContainerRequirements(
            api_port=_DEFAULT_PORT,
            environment=environment,
            model_references=references,
            network_mode="host",
            devices=["/dev/infiniband"],
            capabilities=["IPC_LOCK"],
            ulimits={"memlock": (-1, -1)},
            serves_inference=position.node_index == 0,
        )

    def schema_probe(self) -> SchemaProbe:
        """``tensorfold-native capabilities --json``: the binary's own account.

        Structured, unlike llama.cpp's ``--help`` text, so the surface is read
        rather than scraped. No script and no device: the flag list does not
        depend on one.
        """
        return SchemaProbe(entrypoint=[_BINARY], command=["capabilities", "--json"], script=None)

    def parse_schema(self, output: str) -> list[RuntimeOption]:
        """Flags from ``serve.flags``; their shapes are unknown and reported as such.

        TensorFold names each flag and, for some, its accepted values -- not a
        type or an arity. Those stay ``None`` rather than being guessed, and
        each option carries the policy's verdict, whose default is refusal.
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
            if isinstance(candidate, dict) and isinstance(
                (candidate.get("serve") or {}).get("flags"), dict
            ):
                payload = candidate
                break
        if payload is None:
            raise ValueError("capabilities --json emitted no serve.flags payload")
        options = []
        for flag, facts in sorted(payload["serve"]["flags"].items()):
            if not isinstance(flag, str) or not flag.startswith("--"):
                raise ValueError(f"unexpected flag entry {flag!r} in serve.flags")
            dest = normalize_dest(flag.lstrip("-"))
            choices = facts.get("values") if isinstance(facts, dict) else None
            authorized, reason = authorize(dest)
            options.append(
                RuntimeOption(
                    name=dest,
                    flags=[flag],
                    choices=[str(c) for c in choices] if isinstance(choices, list) else [],
                    authorized=authorized,
                    unauthorized_reason=reason,
                    policy_version=POLICY_VERSION,
                )
            )
        return options

    def validate_distribution(self, config: dict[str, Any], *, node_count: int) -> None:
        """Two ranks at most, over a declared interface.

        Refused without ``distributed_interface`` for the reason vLLM refuses
        it: a group that forms over whatever interface it picks forms over the LAN on
        this estate, and reports success.
        """
        if node_count > _MAX_NODES:
            raise ValueError(
                f"tensorfold forms groups of at most {_MAX_NODES} nodes (GLM-5.3-Flash and "
                f"Nemotron Super VL on two GB10s); {node_count} nodes were declared"
            )
        if node_count > 1 and not _config(config).distributed_interface:
            raise ValueError(
                "a two-node tensorfold deployment needs distributed_interface: the fabric "
                "interface its collectives bind (on this estate, the ConnectX-7 port)"
            )

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: NodePosition | None = None,
        endpoint_port: int | None = None,
    ) -> list[str]:
        validated = _config(config)
        args = [model_path]
        spans = _spans_nodes(position)
        serves = not spans or (position is not None and position.node_index == 0)
        if serves:
            # The container-side interface: bridge networking maps the
            # deployment's port to 8080; host networking listens on the
            # deployment's port directly.
            args += ["--host", "0.0.0.0"]  # noqa: S104 - inside the container namespace
            if spans and endpoint_port is not None:
                args += ["--port", str(endpoint_port)]
            if validated.served_model_name is not None:
                args += ["--name", validated.served_model_name]
        if validated.context is not None:
            args += ["--context", str(validated.context)]
        if validated.parallel is not None:
            args += ["--parallel", str(validated.parallel)]
        if validated.max_tokens is not None:
            args += ["--max-tokens", str(validated.max_tokens)]
        if validated.thinking is not None:
            args += ["--thinking" if validated.thinking else "--no-thinking"]
        if validated.reasoning_effort is not None:
            args += ["--reasoning-effort", validated.reasoning_effort]
        if validated.drafts is False:
            args += ["--no-drafts"]
        if validated.mtp_drafts is not None:
            args += ["--mtp-drafts", str(validated.mtp_drafts)]
        if validated.drafter is not None:
            if not Path(validated.drafter).is_dir():
                raise ValueError(
                    f"drafter {validated.drafter!r} is not a model directory on this node; "
                    f"acquire the drafter to this node and use its store path"
                )
            args += ["--drafter", validated.drafter]
        if validated.kv_cache is not None:
            args += ["--kv-cache", validated.kv_cache]
        if validated.prompt_cache_gib is not None:
            args += ["--prompt-cache-gib", f"{validated.prompt_cache_gib:g}"]
        if validated.vision:
            args += ["--vision"]
        if validated.precision is not None:
            args += ["--precision", validated.precision]
        for key, value in validated.extra_args.items():
            flag = normalize_flag(key)
            if isinstance(value, bool):
                # TensorFold's switches are mostly bare (``--vision``,
                # ``--metrics-open``) with no ``--no-`` form, so ``false`` means
                # "leave the switch off" rather than a negation it would refuse.
                if value:
                    args += [f"--{flag}"]
            else:
                args += [f"--{flag}", str(value)]
        return args

    def declared_memory_fraction(self, config: dict[str, Any]) -> float | None:  # noqa: ARG002
        """None: TensorFold sizes itself from what is free after a fixed reserve."""
        return None

    def validate_model_path(self, path: Path) -> None:
        """A checkpoint directory: ``config.json`` and safetensors weights.

        TensorFold reads Hugging Face trees (NVFP4, MLX 4-bit, EXL3); a
        directory without both would fail at load with a less useful message.
        """
        if not path.is_dir():
            raise ValueError(f"model path is not a directory: {path}")
        if not (path / "config.json").is_file():
            raise ValueError(f"model directory has no config.json: {path}")
        if not any(p.stat().st_size > 0 for p in path.rglob("*.safetensors") if p.is_file()):
            raise ValueError(f"model directory has no .safetensors weights: {path}")
