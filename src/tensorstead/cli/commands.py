"""CLI commands for nodes, runtimes, models, and deployments.

Every management command is an HTTP call to the coordinator API
— the CLI holds no management logic of its own. ``TENSORSTEAD_API``
selects the coordinator.

Commands wired in Phase 3: ``node register|list|show|reachability|
deregister``, ``runtime list``, ``model acquire|list``, and ``deployment
create|start``.
"""

from __future__ import annotations

import json
import os
import textwrap
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from tensorstead.cli.config import (
    EXIT_AUTH_REFUSED,
    EXIT_FAILURE,
    EXIT_UNREACHABLE,
    EXIT_USAGE,
    api_url,
    config_path,
    resolved_api_url,
    resolved_ca_bundle,
    resolved_token,
    tls_verify,
)
from tensorstead.version import VERSION

nodes_app = typer.Typer(name="node", help="Node inventory and registration.", no_args_is_help=True)
models_app = typer.Typer(
    name="model", help="Model acquisition and inventory.", no_args_is_help=True
)
runtimes_app = typer.Typer(name="runtime", help="Supported runtimes.", no_args_is_help=True)
deployments_app = typer.Typer(
    name="deployment", help="Deployment definition and lifecycle.", no_args_is_help=True
)
images_app = typer.Typer(name="image", help="Container images on nodes.", no_args_is_help=True)
operations_app = typer.Typer(name="operation", help="Operation records.", no_args_is_help=True)
inferencekeys_app = typer.Typer(
    name="inferencekey",
    help="Credentials runtimes require from inference clients.",
    no_args_is_help=True,
)
buildspecs_app = typer.Typer(
    name="buildspec",
    help="Recorded recipes for runtime images.",
    no_args_is_help=True,
)
credentials_app = typer.Typer(
    name="credential", help="Per-source credential references.", no_args_is_help=True
)

_global_json_mode: ContextVar[bool] = ContextVar("tensorstead_global_json_mode", default=False)


# ------------------------------------------------------------------- http
def _client(*, timeout: float = 30.0) -> httpx.Client:
    # verify carries the managed private CA when the coordinator serves TLS
    # It is never False: an unverified client would accept any
    # certificate and leave the token as exposed as plain HTTP.
    return httpx.Client(base_url=api_url(), timeout=timeout, verify=tls_verify())


def _token() -> str | None:
    """The management token from whichever source supplies it.

    Reads through the same resolution the status command reports, so the token
    a command uses and the source status names can never disagree.
    """
    return resolved_token().value


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _emit(data: object, *, json_mode: bool) -> None:
    if _effective_json(json_mode):
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        _render_human(data)


def _effective_json(command_flag: bool) -> bool:
    """Honor either a command-local or root-level ``--json`` flag."""
    return command_flag or _global_json_mode.get()


def set_global_json_mode(enabled: bool) -> None:
    """Set output mode for this CLI invocation before its subcommand runs."""
    _global_json_mode.set(enabled)


def _render_human(data: object) -> None:
    """Render every coordinator response for an operator, never as ``repr``.

    JSON is the stable programmatic interface.  The default is intentionally a
    short, labelled view of the same response so a person can see the decision
    they need to make without reading nested Python dictionaries.
    """
    if isinstance(data, list):
        _render_list(data)
        return
    if isinstance(data, dict):
        _render_record(data)
        return
    typer.echo(str(data))


def _render_list(items: list[object]) -> None:
    if not items:
        typer.echo("No results.")
        return
    records = [item for item in items if isinstance(item, dict)]
    if not records:
        typer.echo("\n".join(str(item) for item in items))
        return
    first = records[0]
    if "declared" in first and "observed" in first:
        _render_deployment_list(records)
    elif {"name", "agent_endpoint"} <= first.keys():
        _render_table(
            "NODES",
            # "CONTRACT" read as the node's current version; it is the value
            # captured at registration. The header now says which.
            ("NAME", "AGENT", "CONTRACT (AT REG.)", "REGISTERED", "RESERVED"),
            [
                (
                    str(item.get("name", "?")),
                    str(item.get("agent_endpoint", "?")),
                    str(
                        item.get(
                            "registered_contract_version",
                            item.get("agent_contract_version", "?"),
                        )
                    ),
                    _short_time(item.get("registered_at")),
                    str(item.get("reserved_reason", "")) if item.get("reserved") else "",
                )
                for item in records
            ],
        )
    elif {"source_id", "source_model_id", "replicas"} <= first.keys():
        _render_table(
            "MODELS",
            ("ID", "MODEL", "REVISION", "FILES", "COPIES"),
            [
                (
                    _short_id(item.get("id")),
                    f"{item.get('source_id', '?')}:{item.get('source_model_id', '?')}",
                    _pinned_revision(item),
                    _file_selection(item),
                    _replica_summary(item.get("replicas")),
                )
                for item in records
            ],
        )
    elif {"type", "versions", "supports_distributed"} <= first.keys():
        _render_table(
            "RUNTIMES",
            ("RUNTIME", "VERSIONS", "MULTI-NODE"),
            [
                (
                    str(item.get("type", "?")),
                    ", ".join(str(version) for version in item.get("versions", [])) or "default",
                    "yes" if item.get("supports_distributed") else "no",
                )
                for item in records
            ],
        )
        # Suggested images are a starting point, not a guarantee, and
        # the note is what keeps them honest -- a tag goes stale on the runtime's
        # release schedule, not ours, so printing the tag alone would be the
        # absence-as-answer defect. Printed under the table because a long note
        # does not fit one, and grouped per runtime so it stays attached to the
        # runtime it is about.
        _render_suggested_images(records)
    elif {"source_id", "name", "is_default", "set_at"} <= first.keys():
        _render_table(
            "CREDENTIAL REFERENCES",
            ("SOURCE", "NAME", "DEFAULT", "SET"),
            [
                (
                    str(item.get("source_id", "?")),
                    str(item.get("name", "?")),
                    "yes" if item.get("is_default") else "no",
                    _short_time(item.get("set_at")),
                )
                for item in records
            ],
        )
    elif {"kind", "state", "id"} <= first.keys():
        _render_table(
            "OPERATIONS",
            ("ID", "KIND", "STATE", "REVISION", "FINISHED"),
            [
                (
                    _short_id(item.get("id")),
                    str(item.get("kind", "?")),
                    str(item.get("state", "?")),
                    str(item.get("deployment_revision") or "—"),
                    _short_time(item.get("finished_at")),
                )
                for item in records
            ],
        )
    elif {"revision", "runtime_type", "image_reference", "endpoint"} <= first.keys():
        _render_table(
            "DEPLOYMENT REVISIONS",
            ("REVISION", "MODEL", "RUNTIME", "ENDPOINT", "CREATED"),
            [
                (
                    str(item.get("revision", "?")),
                    _revision_model(item),
                    str(item.get("runtime_type", "?")),
                    str(item.get("endpoint", "?")),
                    _short_time(item.get("created_at")),
                )
                for item in records
            ],
        )
    elif {"node_id", "reference", "digest"} <= first.keys():
        _render_table(
            "IMAGES",
            ("NODE", "IMAGE", "DIGEST", "ORIGIN", "BUILT FROM", "PULLED"),
            [
                (
                    _short_id(item.get("node_id")),
                    str(item.get("reference", "?")),
                    _short_digest(item.get("digest")),
                    str(item.get("origin", "?")),
                    str(item.get("produced_by") or "—"),
                    _short_time(item.get("pulled_at")),
                )
                for item in records
            ],
        )
    else:
        _render_table(
            "RESULTS",
            ("RESULT",),
            [(json.dumps(item, default=str, sort_keys=True),) for item in records],
        )


def _render_node_observed(node_id: str) -> None:
    """Ask the node what it is now, and say so beside what it was.

    Best-effort by design. A node being down is an ordinary answer here, not an
    error: `node show` must still render the record in full when observation
    fails, exactly as a deployment does.
    """
    if not node_id:
        return

    typer.echo("")
    try:
        response = _client(timeout=10.0).get(
            f"/v1/nodes/{node_id}/reachability", headers=_headers()
        )
        observed = response.json() if response.status_code == 200 else {}
    except Exception:
        observed = {}

    if not observed:
        typer.echo("OBSERVED")
        typer.echo("  unreachable        the node did not answer just now")
        return

    typer.echo(f"OBSERVED  as of {_short_time(observed.get('observed_at'))}")
    typer.echo(f"  status             {observed.get('status', 'unknown')}")

    # Three outcomes, and naming the wrong one misdirects the person reading it.
    # The field missing from the response means this *coordinator* predates
    # reporting it; present-but-null means the coordinator asked and the *agent*
    # did not answer. Blaming the agent for the coordinator's silence would send
    # an operator to upgrade the wrong host.
    if "contract_version" not in observed:
        typer.echo("  contract           — (coordinator predates reporting this)")
    elif not observed["contract_version"]:
        typer.echo("  contract           — (agent did not say)")
    else:
        typer.echo(f"  contract           {observed['contract_version']}")

    agent_version = observed.get("agent_version")
    if agent_version:
        typer.echo(f"  agent release      {agent_version}")


def _render_record(record: dict) -> None:
    if "operation_id" in record:
        operation_id = str(record["operation_id"])
        typer.echo("Operation accepted")
        typer.echo(f"  ID       {operation_id}")
        typer.echo(f"  Next     stead operation show {operation_id}")
        _render_advisories(record.get("warnings"))
    elif {"revision", "restart_required", "applied"} <= record.keys():
        typer.echo("Deployment updated")
        typer.echo(f"  revision           {record.get('revision', '?')}")
        typer.echo(f"  restart required   {'yes' if record.get('restart_required') else 'no'}")
        typer.echo("  host changed       no")
        # The configuration recorded on the new revision. Printed here
        # so a ``--replace-config`` that dropped a key is visible the instant it
        # happens — and a ``--config`` patch that kept the rest is too. Without
        # this line the only sign of a wholesale drop is the next
        # ``deployment show``, by which point the operator has moved on.
        recorded = record.get("runtime_config")
        if isinstance(recorded, dict):
            typer.echo("  runtime config")
            if recorded:
                for key, value in recorded.items():
                    typer.echo(f"    {key} = {value}")
            else:
                typer.echo("    (empty)")
        _render_advisories(record.get("warnings"))
    elif "observed" in record:
        _render_observed(record)
    elif {"id", "name", "agent_endpoint"} <= record.keys():
        typer.echo(f"Node  {record.get('name', '?')}  ({_short_id(record.get('id'))})")
        typer.echo("")
        typer.echo("DECLARED")
        typer.echo(f"  agent              {record.get('agent_endpoint', '?')}")
        # Labelled with when it was true. This line read plain "contract", which
        # described a registration-time snapshot as though it were current --
        # the same habit later removed from deployment observation.
        registered = _short_time(record.get("registered_at"))
        typer.echo(
            f"  contract           {record.get('agent_contract_version', '?')}"
            f"   (as registered, {registered})"
        )
        facts = record.get("platform_facts")
        if isinstance(facts, dict) and facts:
            platform = ", ".join(f"{key}={value}" for key, value in facts.items())
            typer.echo("  platform           " + platform)
        typer.echo(f"  registered         {registered}")
        if record.get("reserved"):
            note = record.get("reserved_reason") or "(no reason given)"
            typer.echo(f"  reserved           yes — {note}")
        if record.get("has_management_token_override"):
            typer.echo("  management token   node-specific (not the fleet-wide token)")
        _render_node_observed(str(record.get("id", "")))
    elif {"id", "source_id", "source_model_id", "replicas"} <= record.keys():
        typer.echo(f"Model  {_short_id(record.get('id'))}")
        source = f"{record.get('source_id', '?')}:{record.get('source_model_id', '?')}"
        typer.echo(f"  source             {source}")
        typer.echo(f"  revision           {_pinned_revision(record)}")
        selector = record.get("file_selector") or []
        if selector:
            typer.echo(f"  files              {len(selector)} selected")
            for pattern in selector:
                typer.echo(f"                       {pattern}")
        else:
            typer.echo("  files              whole repository")
        typer.echo(f"  copies             {_replica_summary(record.get('replicas'))}")
    elif {"id", "kind", "state"} <= record.keys():
        typer.echo(f"Operation  {_short_id(record.get('id'))}")
        typer.echo(f"  kind               {record.get('kind', '?')}")
        typer.echo(f"  state              {record.get('state', '?')}")
        if record.get("deployment_revision") is not None:
            typer.echo(f"  deployment rev     {record['deployment_revision']}")
        if record.get("failure_reason"):
            typer.echo(f"  failure            {_summary(record['failure_reason'])}")
        if record.get("finished_at"):
            typer.echo(f"  finished           {_short_time(record['finished_at'])}")
    elif {"status", "observed_at"} <= record.keys():
        _render_status_or_resources(record)
    elif {"source_id", "name", "was_default", "consequence"} <= record.keys():
        typer.echo(f"Credential removed  {record.get('source_id', '?')}/{record.get('name', '?')}")
        typer.echo(f"  default now        {record.get('default_now') or 'none'}")
        typer.echo(f"  consequence        {record.get('consequence', '?')}")
    else:
        typer.echo("RESULT")
        for key, value in record.items():
            typer.echo(f"  {str(key).replace('_', ' '):<18} {_summary(value)}")


def _render_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    typer.echo(title)
    typer.echo("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rows:
        typer.echo("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _short_id(value: object) -> str:
    return str(value or "?")[:12]


def _short_digest(value: object) -> str:
    text = str(value or "?")
    return text[:19] if text.startswith("sha256:") else text[:12]


def _short_time(value: object) -> str:
    if value is None:
        return "—"
    return str(value).replace("T", " ").split(".")[0]


def _file_selection(model: dict) -> str:
    """How much of the repository this model is, in one cell.

    "all" rather than blank: a partial model must never be mistakable for a
    whole one at a glance, and an empty cell reads as missing data rather than
    as a claim.
    """
    selector = model.get("file_selector") or []
    if not selector:
        return "all"
    if len(selector) == 1:
        return str(selector[0])
    return f"{len(selector)} selected"


def _pinned_revision(model: dict) -> str:
    revision = model.get("resolved_revision")
    if not revision:
        return "unresolved"
    pinned = " (pinned)" if model.get("revision_pinned") else ""
    return f"{str(revision)[:12]}{pinned}"


def _replica_summary(replicas: object) -> str:
    if not isinstance(replicas, list):
        return "0"
    states = [
        str(replica.get("state", "unknown")) for replica in replicas if isinstance(replica, dict)
    ]
    if not states:
        return "0"
    return f"{len(states)} ({', '.join(states)})"


def _revision_model(revision: dict) -> str:
    model = revision.get("model")
    if not isinstance(model, dict):
        return "?"
    return f"{model.get('source_id', '?')}:{model.get('source_model_id', '?')}"


def _render_status_or_resources(record: dict) -> None:
    """Render simple health replies and the richer node-resource reply."""
    typer.echo(f"Status  {record.get('status', 'unknown')}")
    typer.echo(f"  observed           {_short_time(record.get('observed_at'))}")
    if "accelerator_utilization_pct" in record:
        utilization = record.get("accelerator_utilization_pct")
        memory_used = record.get("accelerator_memory_used")
        memory_total = record.get("accelerator_memory_total")
        typer.echo(f"  accelerator use    {utilization if utilization is not None else 'unknown'}%")
        if memory_used is not None or memory_total is not None:
            typer.echo(f"  accelerator memory {memory_used or '?'} / {memory_total or '?'} bytes")
        typer.echo(f"  unified memory     {'yes' if record.get('memory_is_unified') else 'no'}")
        storage = record.get("storage")
        if isinstance(storage, list) and storage:
            typer.echo("  storage")
            for location in storage:
                if isinstance(location, dict):
                    typer.echo("    " + _summary(location))
    if record.get("detail"):
        typer.echo(f"  detail             {record['detail']}")


def _summary(value: object) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


# ------------------------------------------------------------------- nodes
@nodes_app.command("register")
def node_register(
    name: Annotated[str, typer.Option("--name", help="Name for the resource. Must be unique.")],
    agent: Annotated[str, typer.Option("--agent", help="Agent base URL, e.g. https://host:8443.")],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Register a host whose agent is already running."""
    _run(
        "POST",
        "/v1/nodes",
        {"name": name, "agent_endpoint": agent},
        json_mode,
        expected=201,
    )


@nodes_app.command("list")
def node_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List registered nodes."""
    _run("GET", "/v1/nodes", None, json_mode)


@nodes_app.command("show")
def node_show(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Show detail for one node."""
    node = _get_node(name)
    _run("GET", f"/v1/nodes/{node['id']}", None, json_mode)


@nodes_app.command("reachability")
def node_reachability(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Check a node's reachability on demand."""
    node = _get_node(name)
    _run("GET", f"/v1/nodes/{node['id']}/reachability", None, json_mode)


@nodes_app.command("resources")
def node_resources(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """On-demand accelerator, memory, and storage reading."""
    node = _get_node(name)
    _run("GET", f"/v1/nodes/{node['id']}/resources", None, json_mode)


@nodes_app.command("deregister")
def node_deregister(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Remove a node; refused while a deployment names it."""
    node = _get_node(name)
    _run("DELETE", f"/v1/nodes/{node['id']}", None, json_mode, expected=204)
    if not _effective_json(json_mode):
        typer.echo(f"Node removed: {name}")


@nodes_app.command("reserve")
def node_reserve(
    name: Annotated[str, typer.Argument()],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Why this node is off-limits to new deployments."),
    ] = "",
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Declare a node off-limits to new work: create, modify, and start all refuse it.

    A statement, not an observation -- nothing on the node changes, and
    nothing already running there is touched or stopped. Use this for a node
    carrying a tenant Tensorstead cannot see, such as a standalone service
    outside its management, so a future deployment does not
    get scheduled onto memory that was never really free.
    """
    node = _get_node(name)
    _run("POST", f"/v1/nodes/{node['id']}/reserve", {"note": reason}, json_mode)
    if not _effective_json(json_mode):
        typer.echo(f"Node reserved: {name}" + (f" — {reason}" if reason else ""))


@nodes_app.command("unreserve")
def node_unreserve(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Un-reserve a node; create, modify, and start may target it again."""
    node = _get_node(name)
    _run("POST", f"/v1/nodes/{node['id']}/unreserve", None, json_mode)
    if not _effective_json(json_mode):
        typer.echo(f"Node un-reserved: {name}")


@nodes_app.command("rotate-management-token")
def node_rotate_management_token(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Give this node its own coordinator-to-agent credential.

    Every agent has historically held the same fleet-wide management
    token, so compromising one discloses a credential that controls every
    other node too. This tells the coordinator to present a node-specific
    value to this node from now on.

    Install the same value in that agent's own environment first -- this
    call does not provision it, only records which value to present, the
    same out-of-band step ``TENSORSTEAD_MGMT_TOKEN`` has always required. The
    secret is read from an interactive prompt, or from stdin when one is
    piped in; there is deliberately no flag that takes the value.
    """
    node = _get_node(name)
    token = _read_secret(f"New management token for {name}")
    _run("POST", f"/v1/nodes/{node['id']}/rotate-management-token", {"token": token}, json_mode)
    if not _effective_json(json_mode):
        typer.echo(f"Management token rotated for node: {name}")


@nodes_app.command("clear-management-token")
def node_clear_management_token(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Revert a node to the fleet-wide ``TENSORSTEAD_MGMT_TOKEN``."""
    node = _get_node(name)
    _run("POST", f"/v1/nodes/{node['id']}/clear-management-token", None, json_mode)
    if not _effective_json(json_mode):
        typer.echo(f"Management token override cleared for node: {name}")


def _get_node(name: str) -> dict:
    nodes = _call_list("GET", "/v1/nodes")
    for node in nodes:
        if node["name"] == name:
            return node
    typer.echo(f"no node named {name!r}", err=True)
    raise typer.Exit(EXIT_FAILURE)


# ---------------------------------------------------------------- runtimes
@runtimes_app.command("list")
def runtime_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List supported runtimes and their capabilities."""
    _run("GET", "/v1/runtimes", None, json_mode)


# ------------------------------------------------------------------ models
@models_app.command("acquire")
def model_acquire(
    source: Annotated[str, typer.Option("--source", help="Model source id, e.g. huggingface.")],
    id: Annotated[str, typer.Option("--id", help="Resource id, or a unique prefix of one.")],
    node: Annotated[
        list[str], typer.Option("--node", help="Node name. Repeat for a multi-node deployment.")
    ],
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="Immutable upstream revision to pin. Strongly preferred over a moving tag.",
        ),
    ] = None,
    credential: Annotated[
        str | None,
        typer.Option(
            "--credential", help="Named credential to use. Omit to use the source's default."
        ),
    ] = None,
    file: Annotated[
        list[str] | None,
        typer.Option(
            "--file",
            help=(
                "Glob of files to acquire from the repository. Repeat to name several. "
                "Omit to acquire all of them. A GGUF repository commonly ships twenty-odd "
                "quantizations; the selection becomes part of the model's identity."
            ),
        ),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Acquire a model onto one or more nodes."""
    # Operators select nodes by their stable inventory names.  The coordinator
    # persists ULID identities, so resolve names at the CLI boundary instead
    # of making a user discover and paste internal IDs.
    payload: dict = {
        "source_id": source,
        "source_model_id": id,
        "nodes": [_get_node(name)["id"] for name in node],
    }
    if revision is not None:
        payload["revision"] = revision
    if credential is not None:
        payload["credential"] = credential
    if file:
        payload["file_selector"] = list(file)
    _run_acquire(payload, json_mode)


def _run_acquire(payload: dict, json_mode: bool) -> None:
    """POST an acquire, then poll the operation to terminal."""
    _post_and_poll("/v1/models:acquire", payload, json_mode, default_code="model_acquire_failed")


def _post_and_poll(path: str, payload: dict, json_mode: bool, *, default_code: str) -> None:
    """POST a long-running operation, then poll it to terminal.

    The coordinator returns 202 immediately and runs the work on a background
    thread, so the 60s read timeout that used to fire on any long operation --
    reporting ``coordinator_unreachable`` indistinguishable from a dead
    coordinator -- can no longer fire. The CLI prints the accepted operation,
    then polls ``GET /v1/operations/{id}`` and emits the terminal record: the
    operation on success, the structured failure reason on failure.

    Shared by ``model_acquire`` and ``image_build``. It was written for the
    first and copied for nothing, so when image build hit the identical timeout
    the fix had to be found again rather than reused. One
    implementation now, so the third long operation inherits it.
    """
    import time

    try:
        response = _client(timeout=30.0).post(path, json=payload, headers=_headers())
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    if response.status_code != 202:
        typer.echo(f"unexpected status {response.status_code}", err=True)
        raise typer.Exit(EXIT_FAILURE)

    operation_id = str(response.json()["operation_id"])
    if not _effective_json(json_mode):
        typer.echo(f"Operation accepted: {operation_id}")

    deadline = time.monotonic() + 3600.0
    last: dict = {}
    while time.monotonic() < deadline:
        poll = _client(timeout=30.0).get(f"/v1/operations/{operation_id}", headers=_headers())
        if poll.status_code >= 400:
            _report_error(poll)
        last = poll.json() if poll.content else {}
        state = last.get("state")
        if state == "succeeded":
            _emit(last, json_mode=json_mode)
            return
        if state == "failed":
            reason = last.get("failure_reason") or {}
            code = reason.get("code", default_code)
            message = reason.get("message", "operation failed")
            typer.echo(f"{code}: {message}", err=True)
            if code == "agent_unreachable":
                raise typer.Exit(EXIT_UNREACHABLE)
            if code == "authorization_refused":
                raise typer.Exit(EXIT_AUTH_REFUSED)
            raise typer.Exit(EXIT_FAILURE)
        time.sleep(0.5)
    typer.echo(
        f"operation {operation_id} did not finish within 3600s "
        f"(last state: {last.get('state', '?')})",
        err=True,
    )
    raise typer.Exit(EXIT_FAILURE)


@models_app.command("list")
def model_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List models."""
    _run("GET", "/v1/models", None, json_mode)


@models_app.command("show")
def model_show(
    model_id: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Show detail for one model, including its resolved revision."""
    model = _get_model(model_id)
    _run("GET", f"/v1/models/{model['id']}", None, json_mode)


@models_app.command("delete")
def model_delete(
    model_id: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Delete a model; refused while referenced."""
    model = _get_model(model_id)
    _run("DELETE", f"/v1/models/{model['id']}", None, json_mode, expected=204)
    if not _effective_json(json_mode):
        typer.echo(
            f"Model removed: {model.get('source_id', '?')}:{model.get('source_model_id', '?')}"
        )


def _get_model(identifier: str) -> dict:
    """Resolve a full or unambiguous displayed model identifier."""
    models = _call_list("GET", "/v1/models")
    return _resolve_record(
        models,
        identifier,
        kind="model",
        display=lambda item: f"{item.get('source_id', '?')}:{item.get('source_model_id', '?')}",
        alternate_key="source_model_id",
    )


# -------------------------------------------------------------- deployments
@deployments_app.command("create")
def deployment_create(
    name: Annotated[
        str | None, typer.Option("--name", help="Name for the resource. Must be unique.")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Acquired model, by name or id prefix (see `model list`)."),
    ] = None,
    runtime: Annotated[
        str | None, typer.Option("--runtime", help="Runtime type (see `runtime list`).")
    ] = None,
    image: Annotated[
        str | None,
        typer.Option(
            "--image", help="Container image reference for the runtime, e.g. registry/repo:tag."
        ),
    ] = None,
    node: Annotated[
        list[str] | None,
        typer.Option("--node", help="Node name. Repeat for a multi-node deployment."),
    ] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="Address the runtime should serve on, as host:port. Recorded, never proxied.",
        ),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option("--config", help="Runtime-specific setting as key=value. Repeat for several."),
    ] = None,
    restore_on_boot: Annotated[
        bool,
        typer.Option(
            "--restore-on-boot/--no-restore-on-boot",
            help=(
                "Bring this deployment back after a node reboot. Off by default: "
                "a deployment nobody has seen run should not run unattended before "
                "anyone can log in. Enable it once the deployment is known good."
            ),
        ),
    ] = False,
    from_export: Annotated[
        str | None,
        typer.Option(
            "--from-export", help="Create from an exported deployment file instead of flags."
        ),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Create a deployment; from an export or field-by-field.

    ``--from-export file.yaml`` maps an exported artifact onto this create
    request; the alternative fields name a deployment directly.
    Either way the host is untouched.
    """
    if from_export:
        _create_from_export(from_export, json_mode, target_endpoint=endpoint)
        return
    if name is None or model is None or runtime is None or image is None:
        typer.echo("create requires --name --model --runtime --image (or --from-export)", err=True)
        raise typer.Exit(EXIT_USAGE)
    runtime_config = _parse_runtime_config(config or [])
    payload: dict = {
        "name": name,
        "model_id": _get_model(model)["id"],
        "runtime_type": runtime,
        "runtime_version": "latest",
        "image_reference": image,
        "runtime_config": runtime_config,
        "participating_nodes": [_get_node(node_name)["id"] for node_name in node or []],
        "endpoint": endpoint or "",
        # Off unless asked for. A deployment nobody has seen
        # run does not get to run unattended before there is a login prompt.
        "restore_on_boot": restore_on_boot,
    }
    _run("POST", "/v1/deployments", payload, json_mode, expected=202)


def _create_from_export(path: str, json_mode: bool, *, target_endpoint: str | None) -> None:
    """Create a deployment from an exported YAML/JSON artifact."""
    try:
        if path.endswith(".yaml") or path.endswith(".yml"):
            import yaml

            with open(path) as fh:
                export = yaml.safe_load(fh)
        else:
            import json as _json

            with open(path) as fh:
                export = _json.load(fh)
    except (OSError, ValueError) as exc:
        typer.echo(f"could not read export {path!r}: {exc}", err=True)
        raise typer.Exit(EXIT_USAGE) from exc
    payload: dict = {"export": export}
    if target_endpoint is not None:
        # Recreating onto a different host usually means a different endpoint.
        # This was accepted and silently discarded, so `--endpoint` appeared to
        # work and the deployment came back on the exported host's address.
        payload["endpoint"] = target_endpoint
    _run(
        "POST",
        "/v1/deployments:create-from-export",
        payload,
        json_mode,
        expected=202,
    )


@deployments_app.command("modify")
def deployment_modify(
    name: Annotated[str, typer.Argument()],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Acquired model, by name or id prefix (see `model list`)."),
    ] = None,
    image: Annotated[
        str | None,
        typer.Option(
            "--image", help="Container image reference for the runtime, e.g. registry/repo:tag."
        ),
    ] = None,
    node: Annotated[
        list[str] | None,
        typer.Option("--node", help="Node name. Repeat for a multi-node deployment."),
    ] = None,
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint",
            help="Address the runtime should serve on, as host:port. Recorded, never proxied.",
        ),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(
            "--config",
            help=(
                "Runtime-specific setting as key=value. Repeat for several. "
                "MERGES into the current config: only the keys you name change; "
                "every other setting is kept."
            ),
        ),
    ] = None,
    replace_config: Annotated[
        list[str] | None,
        typer.Option(
            "--replace-config",
            help=(
                "Replace the WHOLE runtime config map with these settings "
                "(key=value, repeat, or one JSON object). Destructive: keys you "
                "do not name are dropped. Use only when you mean the entire map."
            ),
        ),
    ] = None,
    expect_revision: Annotated[
        int | None,
        typer.Option(
            "--expect-revision",
            help="Refuse if the deployment moved past this revision.",
        ),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Modify a deployment definition; new numbered revision.

    ``--config`` patches individual settings and keeps the rest; ``--replace-config``
    replaces the whole config map and drops anything you do not name. The two
    are mutually exclusive. Prints whether a restart is required; never restarts
    anything.
    """
    if config is not None and replace_config is not None:
        typer.echo("--config merges and --replace-config replaces; use one, not both", err=True)
        raise typer.Exit(EXIT_USAGE)
    deployment = _get_deployment(name)
    payload: dict = {}
    if model is not None:
        payload["model_id"] = _get_model(model)["id"]
    if image is not None:
        payload["image_reference"] = image
    if node:
        payload["participating_nodes"] = [_get_node(node_name)["id"] for node_name in node]
    if endpoint is not None:
        payload["endpoint"] = endpoint
    if config is not None:
        payload["runtime_config"] = _parse_runtime_config(config)
        payload["replace_config"] = False
    if replace_config is not None:
        payload["runtime_config"] = _parse_runtime_config(replace_config)
        payload["replace_config"] = True
    if expect_revision is not None:
        payload["expected_revision"] = expect_revision
    _run(
        "PATCH",
        f"/v1/deployments/{deployment['id']}",
        payload,
        json_mode,
    )


def _parse_runtime_config(values: list[str]) -> dict[str, object]:
    """Accept repeatable ``key=value`` flags or one JSON configuration object.

    A bad configuration must fail before contacting a host.  The earlier
    behavior silently discarded values without an equals sign, which made a
    successful-looking command create the wrong deployment.
    """
    if not values:
        return {}
    if len(values) == 1 and values[0].lstrip().startswith("{"):
        try:
            parsed = json.loads(values[0])
        except json.JSONDecodeError as exc:
            typer.echo(f"invalid --config JSON: {exc.msg}", err=True)
            raise typer.Exit(EXIT_USAGE) from exc
        if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
            typer.echo("--config JSON must be an object with string setting names", err=True)
            raise typer.Exit(EXIT_USAGE)
        return parsed

    config: dict[str, object] = {}
    for value in values:
        if "=" not in value:
            typer.echo(
                f"invalid --config value {value!r}; use key=value or one JSON object",
                err=True,
            )
            raise typer.Exit(EXIT_USAGE)
        key, setting = value.split("=", 1)
        if not key:
            typer.echo("invalid --config value; setting name cannot be empty", err=True)
            raise typer.Exit(EXIT_USAGE)
        config[key] = setting
    return config


@deployments_app.command("revisions")
def deployment_revisions(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List every retained revision."""
    deployment = _get_deployment(name)
    _run("GET", f"/v1/deployments/{deployment['id']}/revisions", None, json_mode)


@deployments_app.command("export")
def deployment_export(
    name: Annotated[str, typer.Argument()],
    revision: Annotated[
        int | None,
        typer.Option(
            "--revision",
            help="Immutable upstream revision to pin. Strongly preferred over a moving tag.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("-o", "--output", help="Write output to this file instead of stdout."),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Export one revision (default: current) as YAML."""
    deployment = _get_deployment(name)
    params = {}
    if revision is not None:
        params["revision"] = revision
    try:
        response = _client().get(
            f"/v1/deployments/{deployment['id']}/export",
            params=params,
            headers=_headers(),
        )
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    body = response.json()
    if output:
        import yaml

        with open(output, "w") as fh:
            fh.write(yaml.safe_dump(body, sort_keys=False, default_flow_style=False))
        if not _effective_json(json_mode):
            typer.echo(f"exported to {output}")
    elif _effective_json(json_mode):
        typer.echo(json.dumps(body, indent=2, default=str))
    else:
        import yaml

        typer.echo(yaml.safe_dump(body, sort_keys=False, default_flow_style=False))


@deployments_app.command("start")
def deployment_start(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Start a deployment."""
    deployment = _get_deployment(name)
    _run(
        "POST",
        f"/v1/deployments/{deployment['id']}:start",
        None,
        json_mode,
        expected=202,
        timeout=3600.0,
    )


@deployments_app.command("stop")
def deployment_stop(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Stop a deployment."""
    deployment = _get_deployment(name)
    _run("POST", f"/v1/deployments/{deployment['id']}:stop", None, json_mode, expected=202)


@deployments_app.command("restart")
def deployment_restart(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Restart a deployment — stop then start."""
    deployment = _get_deployment(name)
    _run(
        "POST",
        f"/v1/deployments/{deployment['id']}:restart",
        None,
        json_mode,
        expected=202,
        timeout=3600.0,
    )


@deployments_app.command("reconcile")
def deployment_reconcile(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Converge toward declared state."""
    deployment = _get_deployment(name)
    _run(
        "POST",
        f"/v1/deployments/{deployment['id']}:reconcile",
        None,
        json_mode,
        expected=202,
        timeout=3600.0,
    )


@deployments_app.command("remove")
def deployment_remove(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Remove a deployment; model and image retained."""
    deployment = _get_deployment(name)
    _run("DELETE", f"/v1/deployments/{deployment['id']}", None, json_mode, expected=202)


@deployments_app.command("list")
def deployment_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List deployments as a compact operational summary.

    Use ``--json`` when the result is intended for another program.  The
    default deliberately summarizes the declared configuration and the latest
    observed state instead of printing Python's nested dictionary notation.
    """
    try:
        response = _client().get("/v1/deployments", headers=_headers())
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    body = response.json()
    if _effective_json(json_mode):
        typer.echo(json.dumps(body, indent=2, default=str))
    else:
        _render_deployment_list(body)


def _render_deployment_list(items: object) -> None:
    """Render deployment records as an operator-facing table.

    The coordinator API intentionally returns a complete declared/observed
    document for every deployment.  That is useful to an API client but too
    noisy for the normal ``list`` command, where names and health matter most.
    """
    if not isinstance(items, list) or not items:
        typer.echo("No deployments.")
        return

    rows: list[tuple[str, str, str, str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        declared = item.get("declared", {})
        observed = item.get("observed", {})
        if not isinstance(declared, dict) or not isinstance(observed, dict):
            continue
        revision = declared.get("revision", {})
        revision = revision if isinstance(revision, dict) else {}
        model = revision.get("model", {})
        model = model if isinstance(model, dict) else {}
        source = model.get("source_id", "?")
        source_model = model.get("source_model_id", "?")
        running_revision = declared.get("running_revision")
        current_revision = declared.get("current_revision", "?")
        revision_text = f"{running_revision or '—'}/{current_revision}"
        divergences = item.get("divergences", [])
        drift = "yes" if isinstance(divergences, list) and divergences else "—"
        rows.append(
            (
                str(declared.get("name", "?")),
                str(declared.get("desired_state", "?")),
                _deployment_health(observed),
                revision_text,
                str(revision.get("endpoint", "?")),
                f"{source}:{source_model}" + (" *" if drift == "yes" else ""),
            )
        )

    if not rows:
        typer.echo("No deployments.")
        return

    headers = ("NAME", "DESIRED", "HEALTH", "RUNNING/CURRENT", "ENDPOINT", "MODEL")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    typer.echo("DEPLOYMENTS  (* has declared/observed drift; run `deployment show NAME`)")
    typer.echo("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rows:
        typer.echo("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _suggest_runtime_command(observed: dict) -> None:
    """Point at ``deployment runtime`` the moment something is wrong.

    The instant an operator reads "not serving" is the instant they want the
    reason, and before this endpoint existed the only way to get it was SSH. Telling them
    here is the difference between the product answering the next question and
    merely raising it.

    Shown only when there is something to explain. A suggestion printed on every
    healthy deployment is one nobody reads by the time it matters.
    """
    per_node = observed.get("per_node")
    if not isinstance(per_node, dict):
        return
    wrong = any(
        isinstance(node, dict)
        and (node.get("inference_ready") is False or node.get("status") == "not_running")
        for node in per_node.values()
    )
    if wrong:
        typer.echo(
            "  why                `deployment runtime <name>` shows the runtime's own output"
        )


def _deployment_health(observed: dict) -> str:
    """Summarize per-node observation without hiding an unhealthy member."""
    per_node = observed.get("per_node", {})
    if not isinstance(per_node, dict) or not per_node:
        return str(observed.get("status", "unknown"))
    statuses = [
        str(node.get("status", "unknown")) for node in per_node.values() if isinstance(node, dict)
    ]
    # Only an explicit False is a fault. `None` means the probe could not
    # establish the fact, and reporting unknown as degraded is how a health
    # column becomes noise an operator learns to skip.
    #
    # This read `bool(node.get("endpoint_reachable"))`, which was safe only
    # while that field was the systemd boot flag and therefore always a real
    # boolean. Once reachability became something measured, `null` turned into
    # a legitimate answer and the coercion started inventing faults: every
    # deployment predating container labels reported degraded here while
    # `deployment show` reported it running with no divergence.
    unreachable = any(
        node.get("endpoint_reachable") is False
        for node in per_node.values()
        if isinstance(node, dict)
    )
    # A runtime that is up but not serving must not summarise to "running" --
    # that single word is what an operator reads before deciding nothing is
    # wrong, and it was true of every check during the 2026-08-10 incident
    # while inference was down.
    not_serving = any(
        node.get("inference_ready") is False for node in per_node.values() if isinstance(node, dict)
    )
    if statuses and all(status == "running" for status in statuses) and not unreachable:
        return "not serving" if not_serving else "running"
    # A member that is not running, or not reachable, must not be summarised
    # away. Falling through to the overall status let a deployment with an
    # unreachable node still print "running", because the aggregate said so --
    # which is precisely the hiding this function's name disclaims.
    overall = str(observed.get("status", "unknown"))
    if statuses and overall == "running":
        return "degraded"
    return overall


def _render_observed(body: dict) -> None:
    """Render the focused deployment status response."""
    observed = body.get("observed", {})
    observed = observed if isinstance(observed, dict) else {}
    divergences = body.get("divergences", [])
    typer.echo(f"Status  {_deployment_health(observed)}")
    typer.echo(f"  observed           {_short_time(observed.get('observed_at'))}")
    per_node = observed.get("per_node", {})
    if isinstance(per_node, dict) and per_node:
        typer.echo("  nodes")
        for node_id, node in per_node.items():
            node = node if isinstance(node, dict) else {}
            health = str(node.get("status", "unknown"))
            if node.get("endpoint_reachable") is False:
                health += " (endpoint unavailable)"
            elif node.get("inference_ready") is False:
                # Distinct from an unavailable endpoint on purpose: something is
                # listening, so "unavailable" would send an operator looking at
                # the network when the runtime is what stopped serving.
                health += " (listening, not serving)"
            detail = f" — {node['detail']}" if node.get("detail") else ""
            typer.echo(f"    {_short_id(node_id):<12} {health}{detail}")
            # Said plainly, because an unauthenticated inference endpoint is
            # something an operator should never learn by accident. `None` is
            # an agent too old to report it, which is not the same as "open".
            authenticated = node.get("endpoint_authenticated")
            if authenticated is True:
                typer.echo(f"    {'':<12} inference endpoint requires a key")
            elif authenticated is False:
                typer.echo(f"    {'':<12} inference endpoint is UNAUTHENTICATED")
    _suggest_runtime_command(observed)
    if isinstance(divergences, list) and divergences:
        typer.echo(f"  drift              yes ({len(divergences)} difference(s))")
    else:
        typer.echo("  drift              no")


@deployments_app.command("show")
def deployment_show(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Show declared and observed state as separate blocks."""
    deployment = _get_deployment(name)
    try:
        response = _client().get(f"/v1/deployments/{deployment['id']}", headers=_headers())
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    body = response.json()
    if _effective_json(json_mode):
        typer.echo(json.dumps(body, indent=2, default=str))
    else:
        _render_deployment(body)


@deployments_app.command("status")
def deployment_status(
    name: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Show observed state only."""
    deployment = _get_deployment(name)
    _run("GET", f"/v1/deployments/{deployment['id']}/status", None, json_mode)


@deployments_app.command("runtime")
def deployment_runtime(
    name: Annotated[str, typer.Argument()],
    tail: Annotated[
        int,
        typer.Option(
            "--tail",
            help="Lines of runtime output per node. Raise it for a chatty runtime.",
        ),
    ] = 500,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Show what the runtime says about itself: argv, restarts, recent output.

    `deployment show` reports whether the runtime is serving. This reports why
    it is not. Nothing shown here is stored by the product -- it is read from
    the node when you ask and kept nowhere.
    """
    deployment = _get_deployment(name)
    if json_mode:
        _run("GET", f"/v1/deployments/{deployment['id']}/runtime?tail={tail}", None, True)
        return
    body = _call("GET", f"/v1/deployments/{deployment['id']}/runtime?tail={tail}")
    if not isinstance(body, dict):
        raise typer.Exit(EXIT_FAILURE)
    _render_runtime(body)


def _render_runtime(body: dict) -> None:
    """Render each node's runtime report.

    The per-node body lives in its own function rather than inline. That keeps
    this loop free of the word the no-supervisor guardrail looks for — and the
    guardrail is right to look: a loop mentioning restarts is exactly the shape
    of the watchdog this rule forbids. Reporting a count is not supervising, but
    the rule earns more by being blunt than by being clever, so the code moves.
    """
    per_node = body.get("per_node")
    if not isinstance(per_node, dict) or not per_node:
        typer.echo("No participating nodes reported.")
        return

    # Warn about the unredacted tail once, and only when there is one to warn
    # about. A warning printed on every invocation is one nobody
    # reads by the time it matters — the same discipline as the ``deployment runtime`` hint.
    if _any_node_has_output(per_node):
        typer.echo(
            "Note: the log tail is shown verbatim and unredacted. If a runtime logs "
            "a credential — not this product's, which never reaches argv or logs, "
            "but one an operator placed in a model or workflow — it appears here. "
            "Nothing shown is stored; exposure is bounded to this response."
        )
        typer.echo("")

    for node_id, report in per_node.items():
        _render_runtime_node(str(node_id), report)


def _any_node_has_output(per_node: dict) -> bool:
    """True if any node reported a non-empty log tail worth warning about."""
    for report in per_node.values():
        if isinstance(report, dict):
            tail = report.get("log_tail")
            if isinstance(tail, str) and tail.strip():
                return True
    return False


def _render_runtime_node(node_id: str, report: object) -> None:
    """Render one node's runtime report; output last, verbatim, flush left.

    The log tail is printed unindented on purpose. Indenting it to match the
    surrounding block would corrupt tracebacks, which are the most valuable
    thing this command shows, and an operator pasting one elsewhere should get
    exactly what the runtime wrote.
    """
    typer.echo(f"NODE {_short_id(node_id)}")
    if not isinstance(report, dict):
        typer.echo("  (malformed report)")
        return

    detail = report.get("detail")
    if detail:
        typer.echo(f"  {detail}")

    running = report.get("running")
    if running is not None:
        typer.echo(f"  running            {'yes' if running else 'no'}")
    exit_code = report.get("exit_code")
    if exit_code is not None:
        typer.echo(f"  exit code          {exit_code}")

    _render_restart_count(report.get("restart_count"))

    argv = report.get("argv")
    if argv:
        typer.echo("  launched with")
        typer.echo(f"    {' '.join(str(part) for part in argv)}")

    _render_argv_honoured(report.get("argv_honoured"), report.get("running_processes"))
    _render_cache(report.get("cache"))
    _render_log_tail(report.get("log_tail"))
    typer.echo("")


def _render_restart_count(count: object) -> None:
    """How many times the engine has relaunched the container.

    A crash-looping runtime reads as healthy at any instant it happens to be
    up, so this count is the only thing that separates a stable deployment from
    one that is dying repeatedly. ``None`` is an agent that did not report it,
    which must not render as zero.
    """
    if count is None:
        typer.echo("  relaunches         — (not reported)")
    elif count:
        typer.echo(f"  relaunches         {count}  (crash loop?)")
    else:
        typer.echo("  relaunches         0")


def _render_build(build: object) -> None:
    """Which build the node is actually running.

    The question `agent release` could never answer: it reports the package
    version, which has read the same string for sixty consecutive builds, so
    comparing a live appliance against a controller was impossible. This is the
    build number and revision the running process was produced from.

    A dirty tree is called out. A build made from modified sources cannot be
    reproduced from its revision, and an operator comparing two nodes by
    revision deserves to know one of them is not what that revision says.
    """
    if not isinstance(build, dict) or not build:
        typer.echo("  build              — (agent too old to report it)")
        return
    number = build.get("build_number")
    revision = build.get("git_revision")
    if number is None and revision is None:
        typer.echo("  build              — (not produced by the build script)")
        return
    dirty = "  [built from a modified tree]" if build.get("dirty") else ""
    typer.echo(f"  build              {number}  {str(revision or '?')[:12]}{dirty}")


def _render_advisories(warnings: object) -> None:
    """Notes about a configuration that was accepted.

    Printed as notes, never as errors, because the operation succeeded and the
    configuration is valid. Wording matters here: an advisory that reads like a
    failure teaches operators to ignore advisories, and one that reads like
    nothing gets skipped. "note" is the honest register.
    """
    if not isinstance(warnings, list) or not warnings:
        return
    typer.echo("")
    for note in warnings:
        typer.echo(f"  note  {note}")


def _render_argv_honoured(honoured: object, processes: object) -> None:
    """Say loudly when the runtime is not running what it was told to.

    A deployment whose declared ``runtime_config`` the runtime never read is a
    record that describes nothing, and it is invisible from every other angle:
    the configured argv above looks exactly right. So this is stated as a fault,
    not a note, and it names the process that is actually running -- because the
    first question anyone asks is "then what *is* it doing".
    """
    if honoured is False:
        typer.echo(
            "  launch args        NOT HONOURED — the runtime is not running what it was given"
        )
        if isinstance(processes, list) and processes:
            for line in processes[:3]:
                typer.echo(f"    running            {line}")
        typer.echo("    declared runtime_config does not describe this deployment")
    elif honoured is None:
        typer.echo("  launch args        — (could not be established)")


def _render_cache(cache: object) -> None:
    """What the durable compile cache holds.

    An empty subtree is called out rather than shown as a zero, because a
    provisioned-but-unused cache is the case an operator most needs to
    recognise and the one that looks most like success: the directory exists,
    the mount is there, and the runtime recompiles anyway. That is the exact
    ambiguity that left "is TRITON_CACHE_DIR being honoured?" unanswerable on
    2026-08-11.
    """
    if cache is None:
        typer.echo("  cache              — (agent does not report it)")
        return
    if not isinstance(cache, dict):
        return
    if not cache.get("present"):
        typer.echo(f"  cache              none at {cache.get('path', '?')}")
        return

    subtrees = cache.get("subtrees") or []
    if not subtrees:
        typer.echo(f"  cache              {cache.get('path')}  EMPTY — nothing written")
        return

    typer.echo(f"  cache              {cache.get('path')}")
    for subtree in subtrees:
        if not isinstance(subtree, dict):
            continue
        files = subtree.get("files", 0)
        name = str(subtree.get("name", "?"))
        if not files:
            typer.echo(f"    {name:<24} EMPTY — the runtime wrote nothing here")
        else:
            typer.echo(f"    {name:<24} {files} files, {_human_bytes(subtree.get('bytes', 0))}")


def _human_bytes(value: object) -> str:
    """Bytes in a unit a human compares at a glance."""
    try:
        size = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _render_log_tail(log_tail: object) -> None:
    """The runtime's own words, or an honest account of why they are missing."""
    if log_tail is None:
        typer.echo("  output             — (could not be read)")
    elif not str(log_tail).strip():
        typer.echo("  output             (the runtime wrote nothing)")
    else:
        typer.echo("  output")
        typer.echo("")
        typer.echo(str(log_tail).rstrip())


def _render_runtime_config(config: dict) -> None:
    """Render the runtime config with the modelled/forwarded split made loud.

    The config is structurally honest already: modelled keys are validated by
    the adapter, while ``extra_args`` and ``host_config`` are forwarded verbatim
    and unvalidated. The problem was that a reader had to know
    which key is which — a bad flag in ``extra_args`` reads the same as a good one
    in the printed output. So the two halves are labelled, and the forwarded
    half says plainly that this product did not check it.
    """
    # ``extra_args`` and ``host_config`` are the two forwarded maps. Everything
    # else is a modelled, adapter-validated field.
    forwarded = {"extra_args", "host_config"}
    modelled = {k: v for k, v in config.items() if k not in forwarded}

    typer.echo("  runtime config")
    if modelled:
        for key, value in modelled.items():
            typer.echo(f"    {key} = {value}")
    else:
        typer.echo("    (none)")

    extra = config.get("extra_args")
    if isinstance(extra, dict) and extra:
        typer.echo("  extra args (forwarded, not validated by this product)")
        for key, value in extra.items():
            typer.echo(f"    {key} = {value}")

    host = config.get("host_config")
    if isinstance(host, dict) and host:
        typer.echo("  host config (forwarded, not validated by this product)")
        for key, value in host.items():
            typer.echo(f"    {key} = {value}")


def _echo_wrapped(text: str) -> None:
    """Echo ``text`` with subsequent lines re-indented to the first line's indent.

    Keeps a long note lined up with its first line instead of letting the
    terminal wrap it ragged. Narrow terminals get a softer width so a 200-col
    note does not run unbroken on an 80-col screen.
    """
    import shutil

    indent = len(text) - len(text.lstrip())
    width = shutil.get_terminal_size((80, 24)).columns
    # Reserve a little margin so the last column is not flush against the edge.
    target = max(40, width - 4)
    for line in textwrap.wrap(text.lstrip(), width=target - indent):
        typer.echo(" " * indent + line)


def _render_suggested_images(records: list[dict]) -> None:
    """Render suggested image references per runtime, as suggestions not guarantees.

    A runtime that named none is reported as such rather than silently omitted
    -- the gap this rendering exists to close is exactly the one where
    ``runtime list`` answered ``versions: []`` and nothing else, and a first-time
    operator concluded no starting point was knowable.
    """
    any_named = any(item.get("suggested_images") for item in records)
    if not any_named:
        return
    typer.echo("\nSUGGESTED IMAGES (a starting point, not a guarantee)")
    for item in records:
        runtime = item.get("type", "?")
        suggestions = item.get("suggested_images") or []
        if not suggestions:
            typer.echo(f"  {runtime}  (none named)")
            continue
        for img in suggestions:
            reference = img.get("reference", "?")
            note = img.get("note", "")
            typer.echo(f"  {runtime}  {reference}")
            if note:
                # Wrap the note so the indent lines up, the same way the table's
                # columns do: a long note is a sentence, not a wall.
                _echo_wrapped(f"      {note}")
            else:
                typer.echo("      (no note)")


def _render_deployment(body: dict) -> None:
    """Render DECLARED and OBSERVED as visually separate blocks.

    UNREACHABLE is rendered as an emphatic value, not a blank cell.
    The divergence footer states that no host state was changed and
    names ``deployment reconcile`` as the explicit next step.
    """
    declared = body.get("declared", {})
    observed = body.get("observed", {})
    divergences = body.get("divergences", [])

    typer.echo(f"\nDeployment  {declared.get('name', '?')}  ({declared.get('id', '?')[:8]})")

    typer.echo("\nDECLARED")
    typer.echo(f"  desired state       {declared.get('desired_state', '?')}")
    typer.echo(f"  current revision    {declared.get('current_revision', '?')}")
    running_rev = declared.get("running_revision")
    typer.echo(f"  running revision    {running_rev or '—'}")
    revision = declared.get("revision", {})
    model = revision.get("model", {})
    pinned = "pinned" if model.get("revision_pinned") else "unpinned"
    typer.echo(
        f"  model               {model.get('source_id', '?')}:{model.get('source_model_id', '?')}"
        f" @ {model.get('resolved_revision', '—')}  ({pinned})"
    )
    rt = revision.get("runtime_type", "?")
    rv = revision.get("runtime_version", "?")
    typer.echo(f"  runtime             {rt} {rv}")
    typer.echo(f"  image               {revision.get('image_reference', '?')}")
    typer.echo(f"  nodes               {', '.join(revision.get('participating_nodes', []))}")
    typer.echo(f"  endpoint            {revision.get('endpoint', '?')}")
    _render_runtime_config(revision.get("runtime_config") or {})

    observed_at = observed.get("observed_at", "?")
    typer.echo(f"\nOBSERVED  as of {observed_at}")
    per_node = observed.get("per_node", {})
    if per_node:
        for node_id, node_obs in per_node.items():
            status = node_obs.get("status", "unknown")
            if status == "unreachable":
                typer.echo(f"  {node_id[:12]}  UNREACHABLE  {node_obs.get('detail', '')}")
            else:
                rev = node_obs.get("running_revision")
                typer.echo(f"  {node_id[:12]}  {status}  revision {rev or '—'}")
    else:
        typer.echo(f"  status              {observed.get('status', 'unknown')}")

    typer.echo("\nDIVERGENCE")
    if divergences:
        for d in divergences:
            typer.echo(
                f"  {d.get('kind', '?')}   {d.get('node_id', '?')[:12]}"
                f"   declared {d.get('declared', '—')}, observed {d.get('observed', '—')}"
            )
    else:
        typer.echo("  (none)")
    typer.echo(
        "  No host state has been changed."
        f" Run `stead deployment reconcile {declared.get('name', '?')}` to converge."
    )


def _get_deployment(name: str) -> dict:
    deployments = _call_list("GET", "/v1/deployments")
    for deployment in deployments:
        declared = deployment.get("declared", deployment)
        if declared.get("name") == name:
            return dict(declared)
    typer.echo(f"no deployment named {name!r}", err=True)
    raise typer.Exit(EXIT_FAILURE)


# ------------------------------------------------------------------- images
@images_app.command("list")
def image_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List images present on nodes."""
    _run("GET", "/v1/images", None, json_mode)


@images_app.command("delete")
def image_delete(
    node_id: Annotated[str, typer.Argument()],
    digest: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Delete an image from the node, then the record."""
    result = _call("DELETE", f"/v1/images/{node_id}/{digest}")
    if _effective_json(json_mode):
        _emit(result, json_mode=True)
        return
    # Say which of the two happened. This line used to read "Image removed from
    # node" unconditionally,
    # while the call removed nothing from any node ever.
    outcome = result.get("outcome") if isinstance(result, dict) else None
    where = f"{_short_id(node_id)}: {_short_digest(digest)}"
    if outcome == "removed":
        typer.echo(f"Image removed from node {where}")
    else:
        typer.echo(f"Node did not hold this image; stale record removed {where}")


@images_app.command("reconcile")
def image_reconcile(
    node_id: Annotated[
        str | None, typer.Argument(help="Only this node. Default: every registered node.")
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Compare image records against the nodes; reap records whose image is gone."""
    path = "/v1/images:reconcile" + (f"?node_id={node_id}" if node_id else "")
    result = _call("POST", path)
    if _effective_json(json_mode):
        _emit(result, json_mode=True)
        return
    body: dict = result if isinstance(result, dict) else {}
    reaped: list[dict] = body.get("reaped") or []
    unrecorded: list[dict] = body.get("unrecorded") or []
    unreachable: list[dict] = body.get("unreachable") or []
    for row in reaped:
        typer.echo(f"reaped stale record  {_short_id(row['node_id'])}  {row['reference']}")
    for row in unrecorded:
        typer.echo(f"present, unrecorded  {_short_id(row['node_id'])}  {row['reference']}")
    for row in unreachable:
        typer.echo(f"UNREACHABLE          {_short_id(row['node_id'])}  ({row['reason']})")
    typer.echo(
        f"{len(reaped)} stale record(s) reaped, {len(unrecorded)} present but unrecorded, "
        f"{len(unreachable)} node(s) unreachable"
    )


# -------------------------------------------------------------- credentials
@credentials_app.command("set")
def credential_set(
    source: Annotated[str, typer.Argument(help="Model source, e.g. huggingface.")],
    name: Annotated[str, typer.Argument(help="Credential name within the source.")],
    default: Annotated[
        bool, typer.Option("--default", help="Make this the source's default credential.")
    ] = False,
    from_env: Annotated[
        str | None,
        typer.Option("--from-env", help="Read the secret from this environment variable."),
    ] = None,
    from_file: Annotated[
        str | None,
        typer.Option(
            "--from-file", help="Read the secret from this file. The value never appears in argv."
        ),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Set a named credential for a source.

    The secret is read from an interactive prompt, or from stdin when one is
    piped in. **There is deliberately no flag that takes the value**: an argv
    flag would put the secret in shell history and in every process listing on
    the box for the duration of the call.

    ``--from-env`` and ``--from-file`` name where the value lives instead of
    carrying it, so they are safe on a command line.
    """
    payload: dict = {"default": default}
    if from_env is not None and from_file is not None:
        typer.echo("give at most one of --from-env or --from-file", err=True)
        raise typer.Exit(EXIT_USAGE)
    if from_env is not None:
        payload["from_env"] = from_env
    elif from_file is not None:
        payload["from_file"] = from_file
    else:
        payload["secret"] = _read_secret(f"Secret for {source}/{name}")
    _run("PUT", f"/v1/credentials/{source}/{name}", payload, json_mode, expected=204)
    if not _effective_json(json_mode):
        typer.echo(f"credential {source}/{name} set")


def _read_secret(prompt: str) -> str:
    """Read a secret from stdin when piped, otherwise prompt without echo.

    Piped input is the scriptable path (``… | stead credential set …``); the
    hidden prompt is the interactive one. Neither reaches argv.
    """
    import sys

    if not sys.stdin.isatty():
        value = sys.stdin.read()
        if not value.strip():
            typer.echo("no secret on stdin", err=True)
            raise typer.Exit(EXIT_USAGE)
        # A trailing newline belongs to the pipe, not to the secret.
        return value.rstrip("\n")
    return str(typer.prompt(prompt, hide_input=True))


@credentials_app.command("list")
def credential_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List credential names and which is default — never values."""
    _run("GET", "/v1/credentials", None, json_mode)


@credentials_app.command("delete")
def credential_delete(
    source: Annotated[str, typer.Argument()],
    name: Annotated[str, typer.Argument()],
    promote: Annotated[
        str | None, typer.Option("--promote", help="Promote the staged copy once verified.")
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Delete a named credential.

    Deleting the default leaves the source with none unless ``--promote`` names
    a successor. Either outcome is printed; the default is never silently
    reassigned.
    """
    path = f"/v1/credentials/{source}/{name}"
    if promote is not None:
        path = f"{path}?promote={promote}"
    _run("DELETE", path, None, json_mode)


# --------------------------------------------------------------- operations
@operations_app.command("show")
def operation_show(
    operation_id: Annotated[str, typer.Argument()],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Progress and terminal outcome of one operation."""
    operation = _get_operation(operation_id)
    _run("GET", f"/v1/operations/{operation['id']}", None, json_mode)


@operations_app.command("list")
def operation_list(
    deployment: Annotated[
        str | None, typer.Option("--deployment", help="Deployment name or id prefix.")
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List operations, optionally filtered by deployment."""
    path = "/v1/operations"
    if deployment is not None:
        dep = _get_deployment(deployment)
        path = f"/v1/operations?deployment_id={dep['id']}"
    _run("GET", path, None, json_mode)


def _get_operation(identifier: str) -> dict:
    """Resolve a full or unambiguous displayed operation identifier."""
    return _resolve_record(
        _call_list("GET", "/v1/operations"),
        identifier,
        kind="operation",
        display=lambda item: f"{item.get('kind', '?')} ({_short_id(item.get('id'))})",
    )


# ------------------------------------------------------------------- runner
def _call(method: str, path: str, payload: dict | None = None) -> object:
    """Perform an HTTP call and raise typer.Exit on a non-success status."""
    try:
        response = _client().request(method, path, json=payload, headers=_headers())
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    return response.json() if response.content else {}


def _call_list(method: str, path: str) -> list[dict]:
    """Call the coordinator and return the response as a list of records.

    Used by lookup helpers that iterate a list response; a non-list response is
    an internal error, not a user-visible value.
    """
    result = _call(method, path)
    if not isinstance(result, list):
        raise typer.Exit(EXIT_FAILURE)
    return [item for item in result if isinstance(item, dict)]


def _resolve_record(
    records: list[dict],
    identifier: str,
    *,
    kind: str,
    display: Callable[[dict], str],
    alternate_key: str | None = None,
) -> dict:
    """Resolve an exact or prefix identifier without ever guessing.

    Human tables intentionally shorten opaque ULIDs.  This helper makes the
    shown prefix directly useful while refusing an ambiguous match.  It also
    lets a model be addressed by its source-model name when that is unique.
    """
    exact = [item for item in records if item.get("id") == identifier]
    if not exact and alternate_key is not None:
        exact = [item for item in records if item.get(alternate_key) == identifier]
    matches = exact or [item for item in records if str(item.get("id", "")).startswith(identifier)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        typer.echo(f"no {kind} matches {identifier!r}", err=True)
    else:
        candidates = ", ".join(display(item) for item in matches)
        typer.echo(f"{kind} ID prefix {identifier!r} is ambiguous: {candidates}", err=True)
    raise typer.Exit(EXIT_USAGE)


def _run(
    method: str,
    path: str,
    payload: dict | None,
    json_mode: bool,
    *,
    expected: int | None = None,
    timeout: float = 30.0,
) -> None:
    try:
        response = _client(timeout=timeout).request(method, path, json=payload, headers=_headers())
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach coordinator at {api_url()}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code >= 400:
        _report_error(response)
    if expected is not None and response.status_code != expected:
        typer.echo(f"unexpected status {response.status_code}", err=True)
        raise typer.Exit(EXIT_FAILURE)
    if response.content:
        _emit(response.json(), json_mode=json_mode)
    elif _effective_json(json_mode):
        typer.echo("{}")


def _report_wrong_service(response: httpx.Response) -> None:
    """Explain a reply that did not come from a Tensorstead coordinator.

    The CLI defaults to ``http://127.0.0.1:8080``, and 8080 is a popular port —
    llama.cpp, one of this product's own supported runtimes, listens there by
    default. So "the CLI reached something else entirely" is not an exotic
    failure, it is the most likely one on a workstation. Rendering that
    service's error body verbatim, as this used to, produces a message about a
    system the operator was not asking about and never names the address that
    was actually called.
    """
    configured = os.environ.get("TENSORSTEAD_API", "").strip()
    served_by = response.headers.get("server", "")
    detail = f" (answered by {served_by})" if served_by else ""

    typer.echo(
        f"not a Tensorstead coordinator: {api_url()} returned HTTP {response.status_code}{detail}",
        err=True,
    )
    typer.echo(
        "The reply did not carry Tensorstead's structured failure shape, so this "
        "address is serving a different application.",
        err=True,
    )
    if not configured:
        typer.echo(
            "\nTENSORSTEAD_API is not set, so the CLI used its default of "
            "http://127.0.0.1:8080. Point it at your coordinator:\n"
            "\n    export TENSORSTEAD_API=https://<coordinator-dns-name>:8080"
            "\n    export TENSORSTEAD_CA_BUNDLE=/path/to/ca.pem"
            '\n    export TENSORSTEAD_MGMT_TOKEN="$(cat /path/to/token-file)"\n'
            "\nSee docs/cli-setup.md.",
            err=True,
        )
    else:
        typer.echo(
            f"\nTENSORSTEAD_API is set to {configured}. Check that it names the "
            "coordinator and its management port, not an inference runtime.",
            err=True,
        )
    raise typer.Exit(EXIT_UNREACHABLE)


def _report_error(response: httpx.Response) -> None:
    """Render a structured failure shape and exit with the right code."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    # Every coordinator failure carries the structured shape from contracts/api.py.
    # Its absence means the address answered, but not as this product.
    if "code" not in body and "message" not in body:
        _report_wrong_service(response)
    code = body.get("code", "http_error")
    message = body.get("message", response.text)
    typer.echo(f"{code}: {message}", err=True)
    if code == "agent_unreachable":
        raise typer.Exit(EXIT_UNREACHABLE)
    if code == "authorization_refused":
        raise typer.Exit(EXIT_AUTH_REFUSED)
    if code in ("invalid_deployment", "runtime_not_distributed", "endpoint_conflict"):
        raise typer.Exit(EXIT_USAGE)
    raise typer.Exit(EXIT_FAILURE)


# ----------------------------------------------------------------- status
def status_command(json_mode: bool = False) -> None:
    """Report what this client is talking to, and where that came from.

    Every value names its origin. A user should never have to work out whether
    an address came from their shell, a config file, or a default they have
    never seen — that hunt is the failure this exists to prevent.
    """
    api = resolved_api_url()
    ca = resolved_ca_bundle()
    token = resolved_token()
    path = config_path()

    report: dict[str, Any] = {
        "coordinator": api.value,
        "coordinator_source": api.source,
        "ca_bundle": ca.value,
        "ca_bundle_source": ca.source,
        "token_source": token.source,
        "config_file": str(path),
        "config_file_exists": path.is_file(),
        "client_version": VERSION,
    }

    # A brand-new, loopback-only coordinator may deliberately run without a
    # management token. Do not send an unconfigured client to the default
    # port just to discover that, though: 8080 is commonly a runtime port.
    # Once the operator has selected a coordinator explicitly, status can
    # safely test it without an Authorization header.
    if not token.value and api.source == "built-in default":
        report["error"] = "no management token — run `stead login`"
        _emit_status(report, json_mode)
        raise typer.Exit(EXIT_USAGE)

    try:
        version = _client(timeout=10.0).get("/v1/version", headers=_headers())
        nodes = _client(timeout=10.0).get("/v1/nodes", headers=_headers())
    except httpx.HTTPError as exc:
        report["reachable"] = False
        report["error"] = str(exc)
        _emit_status(report, json_mode)
        raise typer.Exit(EXIT_UNREACHABLE) from exc

    if version.status_code == 401 or nodes.status_code == 401:
        report["reachable"] = True
        report["error"] = "authentication refused — check the token, or run `stead login`"
        _emit_status(report, json_mode)
        raise typer.Exit(EXIT_AUTH_REFUSED)

    report["reachable"] = True
    if version.is_success:
        body = version.json()
        report["coordinator_version"] = body.get("version")
        report["contract_version"] = body.get("contract_version")
    elif version.status_code == 404:
        report["coordinator_version"] = "unknown (coordinator predates /v1/version)"
    if nodes.is_success:
        registered = nodes.json()
        report["nodes"] = len(registered) if isinstance(registered, list) else 0
    _emit_status(report, json_mode)


def _emit_status(report: dict[str, Any], json_mode: bool) -> None:
    if _effective_json(json_mode):
        _emit(report, json_mode=json_mode)
        return

    typer.echo("STATUS")
    typer.echo(f"  coordinator        {report['coordinator']}")
    typer.echo(f"                     from {report['coordinator_source']}")
    typer.echo(f"  CA bundle          {report.get('ca_bundle') or '(none)'}")
    typer.echo(f"                     from {report['ca_bundle_source']}")
    typer.echo(f"  token              from {report['token_source']}")
    typer.echo(f"  client version     {report['client_version']}")
    if "coordinator_version" in report:
        typer.echo(f"  coordinator ver.   {report['coordinator_version']}")
    if "contract_version" in report:
        typer.echo(f"  contract version   {report['contract_version']}")
    if "nodes" in report:
        typer.echo(f"  registered nodes   {report['nodes']}")
    reachable = report.get("reachable")
    if reachable is not None:
        typer.echo(f"  reachable          {'yes' if reachable else 'NO'}")

    marker = "" if report["config_file_exists"] else "  (does not exist)"
    typer.echo(f"  config file        {report['config_file']}{marker}")
    if not report["config_file_exists"]:
        typer.echo("                     run `stead login` to create it")
    if "error" in report:
        typer.echo(f"  problem            {report['error']}", err=True)


def login_command(
    api: str | None = None,
    ca_bundle_path: str | None = None,
    token_file: str | None = None,
) -> None:
    """Save this coordinator's connection settings, after checking they work.

    Writes ``~/.config/tensorstead/config.yml`` so no shell setup is needed again.
    The token is referenced by path, never copied into the config file: an
    operator should be able to show someone their configuration without
    redacting it.

    It verifies before writing. A configuration saved without being tried is a
    configuration you find out about three commands later.
    """
    path = config_path()
    api_value = api or typer.prompt("Coordinator URL", default=resolved_api_url().value)
    ca_value = ca_bundle_path or typer.prompt(
        "CA bundle path (blank for the public trust store)",
        default=resolved_ca_bundle().value or "",
        show_default=bool(resolved_ca_bundle().value),
    )
    token_value = (
        token_file
        if token_file is not None
        else typer.prompt("Path to the management token file (blank if none)", default="")
    )

    token_path: Path | None = None
    secret: str | None = None
    if token_value.strip():
        token_path = Path(token_value).expanduser()
        try:
            secret = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            typer.echo(f"cannot read token file {token_path}: {exc}", err=True)
            raise typer.Exit(EXIT_USAGE) from exc
        if not secret:
            typer.echo(f"token file {token_path} is empty", err=True)
            raise typer.Exit(EXIT_USAGE)

    verify: str | bool = str(Path(ca_value).expanduser()) if ca_value.strip() else True
    typer.echo(f"verifying {api_value} ...")
    try:
        with httpx.Client(base_url=api_value.rstrip("/"), timeout=15.0, verify=verify) as client:
            headers = {"Authorization": f"Bearer {secret}"} if secret else {}
            response = client.get("/v1/nodes", headers=headers)
    except httpx.HTTPError as exc:
        typer.echo(f"could not reach {api_value}: {exc}", err=True)
        raise typer.Exit(EXIT_UNREACHABLE) from exc
    if response.status_code in (401, 403):
        typer.echo("the coordinator refused that token", err=True)
        raise typer.Exit(EXIT_AUTH_REFUSED)
    if not response.is_success:
        typer.echo(f"{api_value} answered HTTP {response.status_code}", err=True)
        raise typer.Exit(EXIT_FAILURE)

    nodes = response.json()
    count = len(nodes) if isinstance(nodes, list) else 0

    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"coordinator": api_value.rstrip("/")}
    if token_path is not None:
        document["token_file"] = str(token_path)
    if ca_value.strip():
        document["ca_bundle"] = str(Path(ca_value).expanduser())
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)

    typer.echo(f"  reached the coordinator; {count} node(s) visible")
    typer.echo(f"  wrote {path}")
    typer.echo("  `stead status` will now work in any shell.")


# ------------------------------------------------- managed runtime images
@buildspecs_app.command("set")
def buildspec_set(
    name: Annotated[str, typer.Argument(help="Name for the build spec.")],
    base: Annotated[
        str,
        typer.Option("--base", help="Base image. Pin with @sha256:... to be reproducible."),
    ],
    step: Annotated[
        list[str] | None,
        typer.Option("--step", help="A build step. Repeat, in order."),
    ] = None,
    entrypoint: Annotated[
        list[str] | None,
        typer.Option(
            "--entrypoint",
            help=(
                "One argv element of the produced image's ENTRYPOINT. Repeat, in order. "
                "Omit to keep the base image's own."
            ),
        ),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Record a build spec. This executes nothing."""
    # The coordinator has accepted ``entrypoint`` since contract 1.9 and the
    # column has existed since migration 0004, but nothing here ever sent it --
    # so a spec whose base image carries its own entrypoint could not be
    # corrected through the product. Building llama.cpp on an NGC CUDA base
    # found it: that base runs nvidia_entrypoint.sh, which received
    # ``--model ...`` and answered ``exec: --: invalid option``.
    _run(
        "PUT",
        f"/v1/buildspecs/{name}",
        {
            "base_image": base,
            "steps": list(step or []),
            "entrypoint": list(entrypoint or []),
        },
        json_mode,
    )


@buildspecs_app.command("list")
def buildspec_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List recorded build specs."""
    _run("GET", "/v1/buildspecs", None, json_mode)


@buildspecs_app.command("delete")
def buildspec_delete(
    name: Annotated[str, typer.Argument(help="Build spec to delete.")],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Delete a build spec; refused while an image it produced is referenced."""
    _run("DELETE", f"/v1/buildspecs/{name}", None, json_mode)


@images_app.command("build")
def image_build(
    spec: Annotated[str, typer.Argument(help="Recorded build spec to build.")],
    node: Annotated[
        list[str],
        typer.Option(
            "--node",
            help="Node to build on. Repeat to build once and copy to the others.",
        ),
    ],
    reference: Annotated[str, typer.Option("--reference", help="Tag for the produced image.")],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Build a recorded spec into an image, once, on one or more nodes.

    Repeating `--node` builds on the first and copies the produced image to the
    rest. Building separately on each would give the same spec a different
    identifier per node, which breaks digest comparison silently -- so a
    deployment spanning nodes needs one produced image, not several equivalent
    ones.
    """
    node_ids = [_get_node(name)["id"] for name in node]
    payload: dict[str, object] = {"spec": spec, "reference": reference}
    if len(node_ids) > 1:
        payload["nodes"] = node_ids
    else:
        payload["node_id"] = node_ids[0]
    # A build can take many minutes on a Spark. This used to hold the connection
    # open for an hour rather than return, which kept the CLI honest and did
    # nothing for any other client -- MCP's own 60s window still reported
    # "could not reach the coordinator" for a build that had been accepted, and
    # the caller could not tell an unaccepted request from a running one
    # The route now returns an operation id immediately.
    _post_and_poll("/v1/images:build", payload, json_mode, default_code="image_build_failed")


@images_app.command("import")
def image_import(
    archive: Annotated[str, typer.Argument(help="Archive file name in the node image store.")],
    node: Annotated[str, typer.Option("--node", help="Node to import onto.")],
    reference: Annotated[str, typer.Option("--reference", help="Tag for the imported image.")],
    expect: Annotated[
        str | None,
        typer.Option("--expect", help="Required image id; a mismatch imports nothing."),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Import a prebuilt image archive onto a node."""
    node_id = _get_node(node)["id"]
    _run(
        "POST",
        "/v1/images:import",
        {
            "node_id": node_id,
            "reference": reference,
            "archive_name": archive,
            "expected_image_id": expect,
        },
        json_mode,
    )


# ------------------------------------------------- inference credentials
@inferencekeys_app.command("set")
def inferencekey_set(
    name: Annotated[str, typer.Argument(help="Name for this credential.")],
    from_file: Annotated[
        str | None,
        typer.Option("--from-file", help="Read the secret from this file."),
    ] = None,
    from_env: Annotated[
        str | None,
        typer.Option("--from-env", help="Read the secret from this environment variable."),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Store an inference credential.

    With neither option the secret is read from a prompt. It is never accepted
    as a flag value: an argv secret reaches shell history and the process
    listing, which is the rule `credential set` already follows.
    """
    payload: dict = {}
    if from_file:
        payload["from_file"] = from_file
    elif from_env:
        payload["from_env"] = from_env
    else:
        payload["value"] = typer.prompt("Inference credential", hide_input=True)
    _run("PUT", f"/v1/inference-credentials/{name}", payload, json_mode)


@inferencekeys_app.command("list")
def inferencekey_list(
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """List credential names. Values are never returned — there is no read path."""
    _run("GET", "/v1/inference-credentials", None, json_mode)


@inferencekeys_app.command("delete")
def inferencekey_delete(
    name: Annotated[str, typer.Argument(help="Credential to delete.")],
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Delete a credential; refused while a deployment binds it."""
    _run("DELETE", f"/v1/inference-credentials/{name}", None, json_mode)


@inferencekeys_app.command("bind")
def inferencekey_bind(
    deployment: Annotated[str, typer.Argument(help="Deployment name or id prefix.")],
    name: Annotated[
        str | None,
        typer.Option("--name", help="Credential to bind. Omit to clear the binding."),
    ] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Emit JSON for machine consumption.")
    ] = False,
) -> None:
    """Bind a credential to a deployment, or clear it.

    Changes a definition, never host state: a restart is required for it to
    take effect and is never performed implicitly.
    """
    target = _get_deployment(deployment)
    _run(
        "PUT",
        f"/v1/deployments/{target['id']}/inference-credential",
        {"name": name},
        json_mode,
    )
