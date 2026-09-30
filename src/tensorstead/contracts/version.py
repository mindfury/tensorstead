"""Contract version constants and the per-operation minimum-version table.

The design ratifies a ``MAJOR.MINOR`` contract version for the coordinator to
agent hop, with *per-operation minimum versions*, so an operator upgrading the
coordinator does not strand a node whose agent merely predates one new
capability. This module is **data, not machinery**: the
compatibility policy that consumes it lives in the service layer and is unit
tested independently.

Because installation and upgrade are external to this product,
coordinator and agent move on their own schedules and skew is the normal case,
not an error. The published contract version is the *only* compatibility
signal — shared Python types align authoring and validation but say
nothing about whether two deployed sides agree.
"""

from __future__ import annotations

from tensorstead.version import VERSION

CONTRACT_VERSION_MAJOR = 1
# 1.1 adds ObservedStateResponse.endpoint_authenticated. Additive and
# optional, so an older agent omits it and the coordinator reports it as
# unknown rather than as unauthenticated.
# 1.2 adds ObservedStateResponse.inference_ready on the same terms.
# An agent below 1.2 cannot report whether its runtime serves, and the
# coordinator must render that as unknown -- never as ready, which would
# restore the false confidence this change removed, and never as a fault, which
# mark every un-upgraded node broken.
# 1.3 adds ObservedStateResponse.managed_containers, the node's view of the
# ``tensorstead-`` namespace. Additive and optional: an older agent omits it, and
# the coordinator then makes no unexpected-instance claim at all rather than
# reading silence as a clean node.
# 1.4 adds GET /agent/v1/deployments/{id}/runtime -- the runtime's argv,
# restart count, and log tail. A new operation rather than a new
# field, so an agent below 1.4 simply does not serve it and the coordinator
# reports that rather than an empty log, which would read as a silent runtime.
# 1.5 adds RuntimeReport.cache -- what the durable compile cache actually
# holds. Additive and optional. It exists because "a cache was
# provisioned" and "the runtime used it" turned out to be different facts, and
# the product could state only the first while an operator needed the second.
# 1.6 widens NodeResourceObservationResponse.memory_is_unified to bool | None.
# It defaulted to False, so a node whose topology could not be established was
# reported as having denied unified memory -- which both DGX Sparks did, about
# hardware that is unified.
# 1.7 adds DeploymentCreateRequest.node_position on the agent hop.
# Unlike the additive *response* fields above, this one travels coordinator ->
# agent into a model that is `extra="forbid"`, so an agent below 1.7 does not
# ignore it -- it refuses the create outright.
#
# That refusal is correct and deliberate: an agent that silently
# dropped the field would start a container with no rank and report success,
# which is worse than failing. What was missing is the signal, and it is here
# now. Only a *multi-node* create carries the field, so single-node deployments
# remain compatible with any 1.x agent.
# 1.8 adds RuntimeReport.running_processes and argv_honoured.
# Additive and optional. An agent below 1.8 omits them, and the coordinator
# reports "could not be established" rather than "honoured" -- an unchecked
# claim must not read as a verified one, which is the entire subject.
# 1.9 adds ImageBuildRequest.entrypoint on the agent hop.
# Like node_position at 1.7 and unlike the additive response fields, this
# travels coordinator -> agent into a model that is `extra="forbid"`, so an
# agent below 1.9 refuses the build rather than ignoring the field. Correct,
# and it needs the signal: the coordinator stored and sent it while the agent
# rejected it, which is the same one-layer-lagged defect as earlier changes.
# 1.10 adds AgentInfoResponse.build -- which build the agent process actually
# is. Additive and optional: an agent below it omits the field and
# the answer reads "too old to report it", which is distinct from an agent that
# reported and could not identify its own build.
# 1.11 *removes* ImageDistributeRequest.replication_token.
# The destination agent now authenticates the peer fetch with its own configured
# credential rather than one the coordinator hands it. A removal from an
# `extra="forbid"` model would normally be a breaking change; it is not one here
# because nothing ever sent the field -- which is exactly the defect: the source
# content endpoint requires the token, so every multi-node image build would have
# failed distribution with a 401 naming nothing. No coordinator can regress by
# losing it, and an older agent that still accepts it is unaffected because no
# caller populates it.
# 1.12 adds ObservedStateResponse.serves_inference. Additive
# and optional, and unlike the fields above its omission has a *safe* meaning
# rather than an unknown one: an agent below 1.12 always probed, so True is
# exactly what its silence means.
#
# It exists because a vLLM rank other than the head runs with `--headless` and
# starts no API server. The coordinator aggregates endpoint facts worst-case, so
# such a rank -- doing precisely what it was told -- reported `inference_ready:
# false` and vetoed the whole deployment's verdict. A correctly configured
# two-node group would have reported itself broken for as long as it ran, which
# is the same defect as reporting a broken one healthy: the record stops
# matching reality and nothing compares them.
#
# An older agent in a distributed group still vetoes, and that is right. It
# genuinely probed a headless rank and genuinely found nothing; the repair is to
# upgrade the node, not to have the coordinator assume a fact the node never
# reported.
# 1.13 adds DeploymentCreateRequest.restore_on_boot on the agent hop.
# Like node_position at 1.7 and the build entrypoint at 1.9,
# it travels coordinator -> agent into an `extra="forbid"` model, so an agent
# below 1.13 refuses the create rather than ignoring the field.
#
# That refusal is the right one. An agent that dropped the field would arrange
# boot restoration the old way -- unconditionally -- and the coordinator would
# record a deployment as non-persistent while the node had made it persistent.
# A start that fails loudly is better than an estate whose records disagree with
# its systemd units, which is the entire defect class this product exists for.
#
# It exists because boot restoration used to follow from desired lifecycle state
# alone, so a deployment earned the right to run on every future boot merely by
# being started once. One that had never completed a single successful start
# deadlocked its node's GPU driver and was restored into the same deadlock on
# every reboot, until the node was reimaged.
# 1.14 adds AcquireRequest.file_selector on the agent hop.
# Same shape as restore_on_boot at 1.13: it travels coordinator -> agent into an
# `extra="forbid"` model, so an agent below 1.14 refuses the acquire rather than
# ignoring the field.
#
# That refusal is the right one, and the alternative is worse than usual here.
# An agent that dropped the field would acquire the *whole repository* -- for
# the GGUF repo this was built for, twenty-three quantizations and a two-shard
# BF16 against the 39 GB the operator asked for. It would then promote that tree
# and record it under a model whose selection says otherwise. Failing the
# acquire is cheaper than filling the appliance's disk with weights nobody
# asked for and recording them as something they are not.
#
# The coordinator omits the field entirely when the selection is empty, so a
# whole-repository acquire still works against an agent of any version -- which
# is every acquire this estate has ever performed. Only a request that actually
# selects requires 1.14.
# 1.15 adds two agent image operations, `POST /agent/v1/images:remove` and
# `GET /agent/v1/images`.
#
# It exists because `image delete` was a registry operation. The coordinator
# deleted its own row and called no node, so 48 deletes freed zero bytes and
# 430 GB of images survived with no record naming them any more. The agent had
# pull, build, import, probe, serve and distribute, and no way to remove
# anything; the engine had `remove_image` and a docstring saying an
# operator-facing delete never reaches it.
#
# New operations rather than a changed field, so an agent below 1.15 refuses
# them at the version check and the coordinator surfaces that refusal. The
# failure mode being avoided is the one this product exists for: the
# coordinator must not delete a record on the strength of a call the agent
# never performed. Every other operation is unchanged and works against an
# agent of any version.
CONTRACT_VERSION_MINOR = 15

# The contract version this build speaks and reports in-band on every response.
CONTRACT_VERSION = f"{CONTRACT_VERSION_MAJOR}.{CONTRACT_VERSION_MINOR}"

# Agent version. Independent of the wire contract version.
AGENT_VERSION = VERSION


def parse(version: str) -> tuple[int, int]:
    """Parse a ``MAJOR.MINOR`` contract version string into a numeric pair.

    Raises ``ValueError`` on a malformed string so a bogus agent response is a
    hard failure rather than a silently misread compatibility check.
    """
    try:
        major, minor = version.split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError) as exc:  # missing '.' or non-numeric
        raise ValueError(f"invalid contract version: {version!r}") from exc


# Operation identifiers, matching the catalogue in ``service/registry.py``.
# These are the keys of ``OPERATION_MINIMUM_VERSIONS``.
OP_NODE_REGISTER = "node.register"
OP_NODE_LIST = "node.list"
OP_NODE_GET = "node.get"
OP_NODE_DEREGISTER = "node.deregister"
OP_NODE_REACHABILITY = "node.reachability"
OP_NODE_RESOURCES = "node.resources"
OP_RUNTIME_LIST = "runtime.list"
OP_MODEL_ACQUIRE = "model.acquire"
OP_MODEL_LIST = "model.list"
OP_MODEL_GET = "model.get"
OP_MODEL_DELETE = "model.delete"
OP_IMAGE_LIST = "image.list"
OP_IMAGE_DELETE = "image.delete"
OP_IMAGE_RECONCILE = "image.reconcile"
OP_CREDENTIAL_SET = "credential.set"
OP_CREDENTIAL_LIST = "credential.list"
OP_CREDENTIAL_DELETE = "credential.delete"
OP_DEPLOYMENT_CREATE = "deployment.create"
OP_DEPLOYMENT_MODIFY = "deployment.modify"
OP_DEPLOYMENT_LIST = "deployment.list"
OP_DEPLOYMENT_GET = "deployment.get"
OP_DEPLOYMENT_REVISIONS = "deployment.revisions"
OP_DEPLOYMENT_STATUS = "deployment.status"
OP_DEPLOYMENT_RUNTIME = "deployment.runtime"
OP_DEPLOYMENT_EXPORT = "deployment.export"
OP_DEPLOYMENT_START = "deployment.start"
OP_DEPLOYMENT_STOP = "deployment.stop"
OP_DEPLOYMENT_RESTART = "deployment.restart"
OP_DEPLOYMENT_RECONCILE = "deployment.reconcile"
OP_DEPLOYMENT_REMOVE = "deployment.remove"
OP_OPERATION_GET = "operation.get"
OP_OPERATION_LIST = "operation.list"


def _min(major: int, minor: int) -> tuple[int, int]:
    """Literal ``(major, minor)`` helper so the table reads as data."""
    return (major, minor)


# The lookup table from operation to the minimum contract version an agent
# must speak to support it. Every operation in the v1 catalogue is present; an
# operation missing from this table is a bug in the catalogue, not an
# unrestricted operation.
OPERATION_MINIMUM_VERSIONS: dict[str, tuple[int, int]] = {
    OP_NODE_REGISTER: _min(1, 0),
    OP_NODE_LIST: _min(1, 0),
    OP_NODE_GET: _min(1, 0),
    OP_NODE_DEREGISTER: _min(1, 0),
    OP_NODE_REACHABILITY: _min(1, 0),
    OP_NODE_RESOURCES: _min(1, 0),
    OP_RUNTIME_LIST: _min(1, 0),
    OP_MODEL_ACQUIRE: _min(1, 0),
    OP_MODEL_LIST: _min(1, 0),
    OP_MODEL_GET: _min(1, 0),
    OP_MODEL_DELETE: _min(1, 0),
    OP_IMAGE_LIST: _min(1, 0),
    # Both were 1.0 while `image.delete` deleted a coordinator row and called no
    # node at all. Now that it removes the image, it needs an
    # agent that serves `images:remove`.
    #
    # The refusal comes from the *agent* — an older one has no such route and
    # answers 404, which the coordinator surfaces and which leaves the record
    # standing. It is deliberately not a coordinator-side pre-check: the only
    # version the coordinator holds per node is the registration snapshot, and
    # gating capability on that refuses what an upgraded node has had for days
    # (the defect recorded in contracts/api.py). This entry states the
    # requirement and governs registration; it is not a call-time gate.
    OP_IMAGE_DELETE: _min(1, 15),
    OP_IMAGE_RECONCILE: _min(1, 15),
    OP_CREDENTIAL_SET: _min(1, 0),
    OP_CREDENTIAL_LIST: _min(1, 0),
    OP_CREDENTIAL_DELETE: _min(1, 0),
    OP_DEPLOYMENT_CREATE: _min(1, 0),
    OP_DEPLOYMENT_MODIFY: _min(1, 0),
    OP_DEPLOYMENT_LIST: _min(1, 0),
    OP_DEPLOYMENT_GET: _min(1, 0),
    OP_DEPLOYMENT_REVISIONS: _min(1, 0),
    OP_DEPLOYMENT_STATUS: _min(1, 0),
    # The first operation in this table with a minimum above 1.0, which is the
    # first time the per-operation compatibility policy does anything. Every
    # earlier entry was 1.0, so a coordinator could not previously strand one
    # capability without stranding the node -- the mechanism existed and had
    # never been exercised. An agent below 1.4 does not serve this route, and
    # refusing the single operation is the whole point of the design.
    OP_DEPLOYMENT_RUNTIME: _min(1, 4),
    OP_DEPLOYMENT_EXPORT: _min(1, 0),
    OP_DEPLOYMENT_START: _min(1, 0),
    OP_DEPLOYMENT_STOP: _min(1, 0),
    OP_DEPLOYMENT_RESTART: _min(1, 0),
    OP_DEPLOYMENT_RECONCILE: _min(1, 0),
    OP_DEPLOYMENT_REMOVE: _min(1, 0),
    OP_OPERATION_GET: _min(1, 0),
    OP_OPERATION_LIST: _min(1, 0),
}


# Failure codes emitted by the compatibility policy. These are part of the
# published contract and must not drift from the coordinator->agent contract.
CODE_AGENT_VERSION_INCOMPATIBLE = "agent_version_incompatible"
CODE_OPERATION_UNSUPPORTED_BY_AGENT = "operation_unsupported_by_agent"


# A rejection produced by the compatibility policy. Named fields (not just a
# message string) so the caller can act on the code and render required/actual.
class CompatibilityRefusal(Exception):
    """Raised when a coordinator->agent operation is refused on version grounds.

    Carries the structured failure shape the contract requires:
    ``code``, ``required_version``, ``actual_version``, and the operation id.
    """

    def __init__(
        self,
        *,
        code: str,
        operation: str,
        required_version: tuple[int, int],
        actual_version: tuple[int, int],
    ) -> None:
        super().__init__(
            f"{code}: operation {operation!r} requires contract "
            f"{required_version[0]}.{required_version[1]} but agent speaks "
            f"{actual_version[0]}.{actual_version[1]}"
        )
        self.code = code
        self.operation = operation
        self.required_version = required_version
        self.actual_version = actual_version


def check_operation_supported(
    operation: str,
    agent_version: str,
    *,
    contract_major: int = CONTRACT_VERSION_MAJOR,
) -> None:
    """Apply the compatibility policy to one operation.

    Raises ``CompatibilityRefusal`` when the operation is refused:

    - A major mismatch refuses the operation with ``agent_version_incompatible``,
      naming both versions.
    - A matching major with the agent below the operation's minimum refuses
      *that operation only* with ``operation_unsupported_by_agent``, naming the
      required and actual versions.

    A matching major with the agent at or above the operation's minimum passes
    (no exception). ``contract_major`` is the major the coordinator speaks; it
    defaults to the in-build constant so callers rarely pass it, but is a
    parameter so the policy is testable against a coordinator of any major.
    """
    actual = parse(agent_version)
    if actual[0] != contract_major:
        raise CompatibilityRefusal(
            code=CODE_AGENT_VERSION_INCOMPATIBLE,
            operation=operation,
            required_version=(contract_major, 0),
            actual_version=actual,
        )
    minimum = OPERATION_MINIMUM_VERSIONS[operation]
    if actual < minimum:
        raise CompatibilityRefusal(
            code=CODE_OPERATION_UNSUPPORTED_BY_AGENT,
            operation=operation,
            required_version=minimum,
            actual_version=actual,
        )
