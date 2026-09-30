"""Runtime-adapter port (seam).

A runtime adapter knows how to turn a deployment definition into the concrete
arguments a specific inference runtime needs. It
validates runtime-specific configuration against its own schema — there is never
a cross-runtime config union.

``supports_distributed`` is the capability declaration this design tests
against: a multi-node request is refused unless the selected adapter declares
distributed support, and the product never implements distribution itself.

Two v1 adapters exist: vLLM (``supports_distributed = True``) and
llama.cpp (``False``), chosen so the seam has a real counter-example rather than
an asserted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SuggestedImage:
    """One runtime image reference the product can name as a starting point.

    The product *reports* a usable reference per runtime
    without requiring an existing deployment, and the "out of scope" note
    is explicit that reporting is all this is: selection stays with the
    operator.

    So this is a suggestion, not a guarantee. The ``reference`` is one known to
    have worked or to be published by the runtime's own project; the ``note``
    says which, and is the whole point of carrying the two together. Naming a
    tag without the note would be the recurring defect in this codebase --
    absence rendered as a specific answer -- because an image tag goes stale on
    the runtime's release schedule, not on ours, and a stale tag presented as
    authoritative is worse than no tag at all. See ``validate_config`` for the same
    modelled-versus-claimed distinction.
    """

    reference: str
    note: str


@dataclass(frozen=True)
class ReadinessProbe:
    """How one runtime answers "are you serving?".

    Part of the adapter contract, so it lives with the port rather than with
    either side that uses it: the agent issues the probe, and the adapter — which
    the coordinator also loads — declares it.
    """

    path: str
    expect_status: int = 200


@dataclass(frozen=True)
class SchemaProbe:
    """How to ask one runtime image what it accepts.

    The adapter supplies this; it does not supply the answer. A probe is
    runtime-specific knowledge -- vLLM's parser is built by one function,
    llama.cpp's surface is discovered another way -- and an adapter that cannot
    probe returns ``None``, which is a legitimate answer rather than an
    omission.
    """

    # Container entrypoint and leading arguments. ``script`` is appended as the
    # final argument by the engine after being written into the container.
    entrypoint: list[str]
    command: list[str]
    script: str
    # vLLM cannot build its argument parser without a device: constructing the
    # config defaults requires device detection. Measured, not assumed.
    needs_accelerator: bool = False


@dataclass(frozen=True)
class OptionDefault:
    """An option's default, with the difference between kinds of absence kept.

    The first draft stringified every default with ``str()``, which rendered
    three unlike facts identically: a default of ``None``, a default of the
    string ``"None"``, and a default that is a Python object with no
    serialisable form at all. A schema that cannot tell those apart cannot
    validate a configuration against them.

    ``state`` is one of:

    - ``value`` — there is a default and it survives JSON, in ``value``;
    - ``absent`` — the option has no default;
    - ``unrepresentable`` — there is a default, it is a Python object, and
      ``text`` holds its ``repr`` for a human. Deliberately *not* placed in
      ``value``: a reader must not be able to mistake a rendering for the thing.
    """

    state: str
    value: Any = None
    text: str | None = None


@dataclass(frozen=True)
class RuntimeOption:
    """One accepted option, in a form that survives being written down.

    ``argparse`` stores types as Python callables, defaults as arbitrary
    objects, and the *shape* of an option in its action class; none of that is
    JSON. This is the normalised form — enough to validate a
    configuration against twice and get the same answer, which a schema that
    records only a name and a type name cannot do.

    Recording the shape matters as much as the type. ``--x`` taking one value,
    ``--x`` repeatable, and ``--x`` as a bare switch are three different
    contracts that a type name alone renders identically.
    """

    name: str
    flags: list[str]
    # The value's type as the runtime names it (``int``, ``str``, a converter's
    # class name), or ``None`` for a switch that takes no value.
    kind: str | None = None
    # Normalised argparse action: ``store``, ``store_true``, ``store_false``,
    # ``append``, ``count``, ``const``, or the raw class name when unrecognised.
    # Unrecognised is reported rather than guessed -- a custom action this
    # product has not seen is exactly the case where guessing goes wrong.
    action: str | None = None
    # Collection shape as argparse spells it: ``+``, ``*``, ``?``, or a count.
    nargs: str | None = None
    # Whether repeating the flag accumulates rather than overwrites.
    repeatable: bool = False
    # For a bare switch, which way it sets its destination. ``None`` for options
    # that take a value -- the distinction ``--x`` / ``--no-x`` collapses into
    # without it.
    polarity: bool | None = None
    choices: list[str] = field(default_factory=list)
    default: OptionDefault = OptionDefault(state="absent")
    help_text: str | None = None
    # Whether this product will accept the option from an operator.
    #
    # **Defaults to False**, and that default is the fix.
    # It previously defaulted to True, so every option the probe learned and
    # nobody had reviewed arrived pre-authorized -- including
    # ``--trust-remote-code``. Discovery is not authorization: the
    # runtime accepting an option says nothing about whether this product
    # should let a deployment record set it.
    #
    # **Marked, never hidden.** Concealing an option the runtime genuinely has
    # would put an operator back to guessing why a documented flag appears to do
    # nothing, and `--headless` is the case in point: it existed all along,
    # nothing surfaced it, and its absence cost days. The surface says what the
    # runtime accepts; this field says what we do.
    authorized: bool = False
    unauthorized_reason: str | None = None
    # Which version of the authorization policy produced the two fields above.
    # A decision without the rules that made it cannot be re-checked later.
    policy_version: str | None = None


@dataclass(frozen=True)
class ContainerRequirements:
    """What a runtime needs from its container, as opposed to its argv.

    Some runtime needs are not arguments. vLLM will not run in a container with
    Docker's default 64MB of shared memory; a distributed group needs ports
    beyond the API port; RDMA needs raised locked-memory limits. None of that is
    expressible as a launch flag, and all of it is **runtime-specific
    knowledge**, so it belongs to the adapter rather than to the
    agent's runtime-agnostic deployment route — which is exactly where a second
    runtime's variant would otherwise be bolted on as ``if runtime_type ==``.
    That mistake has already been found twice.

    Every field defaults to "no requirement", so a runtime that needs nothing
    declares nothing and the container is created exactly as before.
    """

    # Shared memory, in bytes. Docker's 64MB default is too small for vLLM and
    # the resulting failure names neither shared memory nor Docker.
    shm_size: int | None = None
    # ``host`` lets the runtime use the host's IPC namespace, which some
    # multi-process runtimes need alongside raised shared memory.
    ipc_mode: str | None = None
    # Container ports beyond the deployment's endpoint, published as-is. A
    # distributed runtime forms its own group and needs its own channels
    # (it distributes, we coordinate).
    extra_ports: dict[str, int] = field(default_factory=dict)
    # ``{"memlock": (soft, hard)}``. ``-1`` means unlimited.
    ulimits: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Environment the *runtime* needs, never credentials. Credentials have a
    # dedicated mechanism (``inference_credential_env``) precisely so they do
    # not travel through a general-purpose channel.
    environment: dict[str, str] = field(default_factory=dict)
    # Docker network mode. ``host`` puts the container on the host's network
    # stack, which a distributed runtime rendezvousing over RDMA needs and an
    # ordinary one must not have.
    #
    # Declarable **only here**, by adapter code that is reviewed and tested.
    # ``host_config`` continues to refuse it, and that asymmetry is the whole
    # point: the product may grant what its own code declares; an operator may
    # not name arbitrary host access in configuration, because a deployment
    # record that cannot describe what its container can reach has stopped
    # being a true account of what is running.
    network_mode: str | None = None
    # Host devices to expose, e.g. ``/dev/infiniband``. Same rule as
    # ``network_mode``: adapters declare, configuration may not.
    devices: list[str] = field(default_factory=list)
    # Linux capabilities to add, e.g. ``IPC_LOCK``. Same rule again: adapters
    # declare, configuration may not.
    #
    # Exists because declaring ``/dev/infiniband`` alone was incoherent. RDMA
    # registers memory with the NIC, which pins it, which needs ``IPC_LOCK`` and
    # a raised ``memlock`` -- so the container was handed the device and
    # withheld the ability to use it. The first live TP=2 group that reached its
    # own collectives died there, with the collective library reporting only its
    # generic "unhandled system error".
    #
    # Deliberately a list of named capabilities and not a ``privileged`` flag.
    # Privileged would also solve it, and would hand the container everything
    # else besides; a deployment record that cannot describe what its container
    # may do has stopped being a true account of what is running.
    capabilities: list[str] = field(default_factory=list)
    # Values in this runtime's configuration that name *additional models* the
    # container will read -- a speculative-decoding drafter being the case that
    # forced this. Recognising which config key means "a model" is
    # runtime-specific knowledge, so the adapter names them and
    # the agent decides what to do about it.
    #
    # **Naming one is a request, not a grant.** The agent mounts a named path
    # only when it resolves inside the model store the agent itself owns, and
    # ignores it otherwise. That asymmetry is deliberate and is the same rule
    # as ``devices`` and ``network_mode`` read from the other side: an adapter
    # may say "this is a model", but an *operator* must not be able to turn a
    # configuration string into a host bind-mount. Without the check,
    # ``speculative_config.model`` would be a general-purpose channel for
    # mounting any host path into a container, which is precisely the kind of
    # thing the closed-allowlist failure exists to
    # keep out of configuration.
    #
    # Found on hardware: the agent bind-mounts exactly one
    # model directory, so a drafter acquired through ``model.acquire`` --
    # recorded, pinned, replicated -- was still invisible inside the container,
    # and vLLM rejected its on-node path with "Invalid repository ID or local
    # directory specified". The only thing that worked was a HuggingFace repo
    # id the runtime fetched itself at every container start.
    model_references: list[str] = field(default_factory=list)
    # The port this runtime listens on *inside* the container. The deployment's
    # declared endpoint selects the host-side port; this is the container-side
    # port it maps to, and the two are different facts.
    #
    # The agent published the deployment's endpoint to container port 8000 for
    # every runtime, which is vLLM's listener. llama.cpp listens on 8080, so a
    # llama.cpp deployment started normally and mapped its recorded endpoint to
    # a port nothing was listening on -- reachable-looking and dead.
    # Both shipped adapters declare this explicitly, and a
    # guardrail test asserts they do, so "forgot to declare" is not reachable
    # for a shipped runtime.
    api_port: int | None = None
    # Whether *this* container serves inference, as opposed to participating in
    # a group that does.
    #
    # Almost always True, and deliberately not inferred from ``api_port``: a
    # rank can have a port and still not serve on it. vLLM's non-head ranks run
    # with ``--headless`` and start no API server at all, so probing them finds
    # nothing -- correctly, because there is nothing there. Without this the
    # product would probe every rank, one would always fail, and a *working*
    # two-node deployment would report ``inference_ready: false`` for as long as
    # it ran. Reporting a healthy cluster as broken is the same failure as
    # reporting a broken one as healthy: the record stops matching reality and
    # nothing compares them.
    #
    # Declared by the adapter because which ranks serve is runtime-specific
    # knowledge. A runtime whose every rank serves says nothing
    # and gets the default.
    serves_inference: bool = True
    # The port the group's **head** binds for its own rendezvous, declared only
    # when this runtime's ranks cannot be started concurrently.
    #
    # This runtime deliberately shipped without launch ordering: most distributed
    # backends rendezvous with retry, in which case ordering is convention, and
    # designing around an unwatched guess had already cost this project two
    # restarts. The first live TP=2 start settled it -- the worker exhausted
    # Gloo's short retry budget and exited while the head was still
    # initialising. So this is declared from measurement.
    #
    # It carries the whole staging decision. The agent waits on it before
    # reporting the head started; the lifecycle service reads its presence to
    # know the ranks must be sequenced. One declared fact, so the two cannot
    # disagree about whether a runtime needs staging.
    rendezvous_port: int | None = None
    # An **in-container** path that must survive the container.
    #
    # Deliberately not a volume, a bind, or a host path. The runtime states
    # what must be true for it -- "I need durable storage visible here" -- and
    # the *agent* decides where that lives, provisions it, and can account for
    # it. That difference is what lets this be granted at all: an operator who
    # could name a host path would be describing a mount the product cannot
    # record, while a product-chosen location is as accountable as the model
    # store.
    #
    # An earlier design modelled container needs as a bag of Docker knobs
    # to forbid ``volumes`` outright, because a volume can point anywhere. That
    # forbade the mechanism and the legitimate need along with it. Requirements
    # say what must be true, not which flag to set.
    cache_at: str | None = None


@dataclass(frozen=True)
class NodePosition:
    """Where one container already sits among the declared nodes.

    Named *position*, not placement, and the no-scheduling guardrail is what
    caught the difference. A placement is a decision about where work should
    run, which this product does not make and must never look like it makes:
    the operator declares ``participating_nodes`` and that is the whole of it.
    This says only "you are the second of two, here is everyone's address".

    Passed to ``build_launch_args`` so an adapter can *derive* whatever its
    runtime needs to form a distributed group — a rank, a world size, an
    address to rendezvous at. The product states a position it already knows
    from the revision's declared node order and learns nothing about what a
    rank is, which is what keeps the boundary intact: the runtime forms its own
    group, we hand it the facts and stay out of it.

    ``None`` on a single-node deployment, so every existing adapter and every
    existing deployment behaves exactly as before.

    **Derived, never declared.** The obvious alternative is a ``per_node``
    override map in ``runtime_config``, and it is refused: a deployment is one
    definition, and divergence, export and comparability all compare one
    declared shape against reality. A per-node map makes "the deployment" a set
    of related-but-different ones, and every one of those comparisons starts
    comparing things that were never the same — a large loss of truthfulness
    bought for the convenience of not deriving an integer.
    """

    # Position in ``revision.participating_nodes``, which is ordered and
    # declared. An adapter that needs a rank derives it from here.
    node_index: int
    node_count: int
    # This node's address on the fabric the runtime will rendezvous over, and
    # every participating node's, in the same declared order. Addresses rather
    # than ids: what a distributed runtime needs is somewhere to connect.
    self_address: str
    peer_addresses: list[str]


class RuntimeAdapter(Protocol):
    """Translate one deployment revision into runtime launch parameters."""

    runtime_type: str
    supports_distributed: bool
    # One or more image references an operator can try, each with a note saying
    # what kind of suggestion it is. Runtime-specific knowledge, so it
    # belongs to the adapter: which image is a workable starting
    # point is a fact about the runtime, not about deployments. A runtime that
    # cannot name one declares the empty tuple, and ``runtime list`` says so
    # rather than inventing a reference -- reporting ``unknown`` over a stale
    # guess is the rule this exists to enforce.
    suggested_images: tuple[SuggestedImage, ...]

    def validate_config(
        self, config: dict[str, Any], *, approved_options: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        """Validate runtime-specific config against this runtime's own schema.

        Returns the validated (possibly normalized) config. Raises a
        validation error naming the offending key(s) with the runtime's reason
        before the deployment is treated as valid.

        ``approved_options`` names options a reviewed out-of-band approval has
        authorized for this exact deployment (``tensorstead.domain.approvals``).
        It defaults to empty, which is the unchanged refusal: the guard fails
        closed at every call site that does not deliberately supply one, rather
        than depending on each of them to remember to ask.
        """

    def container_entrypoint(self, config: dict[str, Any] | None = None) -> list[str] | None:
        """Container entrypoint this runtime needs, or None for the image's own.

        Runtime-specific, and therefore the adapter's. This was
        a second ``if runtime_type == "vllm"`` branch in the agent's generic
        deployment route, found by the guardrail written for the first one.
        """

    def inference_credential_env(self, secret: str) -> dict[str, Any]:
        """Container environment carrying an inference credential, if supported.

        The **value** is not this product's to own. It authenticates inference
        clients to the runtime, which is the data plane the product keeps the
        management plane out of, and it is provisioned onto the host by
        external tooling. The agent is handed it and passes it
        on; nothing here stores, records, or exports it.

        What *is* runtime-specific is the mechanism — vLLM reads
        ``VLLM_API_KEY``, another runtime will not. That knowledge belongs to
        the adapter. It previously sat in the agent's
        runtime-agnostic deployment route as ``if runtime_type == "vllm"``,
        which is where a second runtime's variant would have been added next.

        Returning an empty mapping means this adapter offers no mechanism, and
        the runtime will serve **unauthenticated**. That is a statement about
        the adapter, not a claim about what the runtime could support.
        """

    def readiness_probe(self) -> ReadinessProbe | None:
        """The request that proves this runtime is serving, or None.

        Runtime-specific, and therefore the adapter's.
        Both v1 runtimes happen to expose
        an OpenAI-compatible ``/v1/models``, but that is a fact about them, not
        a cross-runtime invariant we may assume on behalf of a runtime nobody
        has written an adapter for yet.

        Returning ``None`` means this adapter cannot ask, and readiness is
        reported as unknown rather than as ready — the failure this exists to
        prevent is a healthy-looking answer nobody checked.
        """

    def container_requirements(
        self, config: dict[str, Any], position: NodePosition | None = None
    ) -> ContainerRequirements:
        """What this runtime needs from its container, given its config.

        Takes the validated config because the requirement often depends on it:
        a single-GPU deployment needs far less shared memory than a
        tensor-parallel one, and reserving the larger amount unconditionally
        would tax every deployment for a capability most do not use.

        An adapter with no such needs returns the default, and the container is
        created exactly as it was before this existed.
        """

    def config_advisories(
        self, config: dict[str, Any], platform_facts: dict[str, Any] | None = None
    ) -> list[str]:
        """Notes about a *valid* configuration that an operator should know.

        Distinct from ``validate_config``, which refuses. These configurations
        are accepted and will run; the runtime will simply do something the
        operator probably did not intend, and will say so somewhere they are
        unlikely to look.

        **Advisory only, never gating**. Returning a note
        must not affect whether an operation proceeds, because a product that
        refuses on the strength of its own guesses about a runtime it does not
        ship is the closed-allowlist failure wearing a different hat.

        Runtime-specific, so it belongs to the adapter: the
        interaction between speculative decoding and batch sizing is a fact
        about vLLM, not about deployments.
        """

    def schema_probe(self) -> SchemaProbe | None:
        """How to ask an image of this runtime what it accepts, or ``None``.

        ``None`` means this adapter has no way to interrogate its runtime. The
        product then knows the surface is *unknown* rather than empty, which is
        a different fact and must stay one.
        """

    def parse_schema(self, output: str) -> list[RuntimeOption]:
        """Turn a probe's raw output into normalised options.

        Paired with ``schema_probe``: the adapter chooses the wire format its
        own probe emits, so nothing outside it needs to know that vLLM's probe
        speaks JSON while another runtime's might not.
        """

    def validate_distribution(self, config: dict[str, Any], *, node_count: int) -> None:
        """Raise when a configuration cannot describe a group of ``node_count``.

        Separate from ``validate_config`` because it needs a fact the config
        does not carry -- how many nodes the deployment names -- and separate
        from ``config_advisories`` because this refuses rather than notes.

        Called before a revision is stored, so a contradiction between the
        declared parallelism and the declared node list is refused while it is
        still only a record, rather than at container start on some node.

        Deliberately narrow. An adapter must refuse only what contradicts the
        *product's own* record -- a declared group whose nodes cannot all
        participate -- and must not litigate the runtime's internal rules on its
        behalf. Refusing on the strength of guesses about a runtime we do not
        ship is the closed-allowlist failure in a different hat.
        """

    def build_launch_args(
        self,
        config: dict[str, Any],
        *,
        model_path: str,
        position: NodePosition | None = None,
        endpoint_port: int | None = None,
    ) -> list[str]:
        """Produce the container/command arguments for this runtime.

        ``position`` is present only for a multi-node deployment.
        An adapter that does not distribute ignores it; one that does derives
        its rank and rendezvous facts from it rather than reading them out of
        configuration.

        ``endpoint_port`` is the port the deployment declared. It is supplied
        because a bind is not always a publish: with a published port map the
        engine translates the container's own port to the declared one, and the
        runtime never needs to know. Under host networking there is no map, and
        a runtime left on its default listens somewhere the deployment record
        does not name. An adapter emits its own port flag
        only when that translation is absent -- the product does not otherwise
        tell a runtime where to listen.
        """

    def declared_memory_fraction(self, config: dict[str, Any]) -> float | None:
        """What share of the accelerator this config claims, or None.

        Runtimes spell this differently and mean subtly different things by it
        -- vLLM's ``gpu_memory_utilization``, SGLang's ``mem_fraction_static``
        -- so the number is the adapter's to report and nobody else's to guess.
        ``None`` means this runtime declares no such budget,
        which is a fact about the runtime rather than a claim that it needs
        nothing.

        It exists because the agent knew the node's free memory, knew the
        deployment's declared fraction, and compared them nowhere. Starting
        SGLang at ``mem_fraction_static: 0.90`` on a node with 13% free asked
        for roughly 109 GB of a 121 GB device and left the host thrashing until
        it needed a power cycle.

        The number is a *fraction of the device's total*, not of what is free.
        That distinction is the whole trap: on a node already serving something
        else, this is not the share a runtime may use, it is the share it will
        take.
        """

    def validate_model_path(self, path: Path) -> None:
        """Raise when the on-disk model tree is not usable by this runtime.

        Called before container creation, so a bad model directory fails fast
        with a named reason instead of "succeeded" at launch and surfacing
        minutes later as an exited container with no classified cause. The
        check is structural and runtime-specific: vLLM wants a
        ``config.json`` and any shards its index names; llama.cpp wants at least
        one ``.gguf``. Raises ``ValueError`` with an actionable message; returns
        silently only on a usable tree.
        """


@dataclass
class RuntimeCapability:
    """Static capability record surfaced by ``runtime.list``."""

    type: str
    versions: list[str] = field(default_factory=list)
    supports_distributed: bool = False
