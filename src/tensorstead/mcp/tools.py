"""MCP tools.

One tool per registry operation, named ``<noun>_<verb>``, each a direct
projection of one coordinator API call. No management logic lives here.

The tools are written out **explicitly rather than generated from the operation
registry**, which is deliberate. Generating them would make the parity
test vacuously true — it would be comparing the registry against itself.
Written by hand, the parity test can actually fail when someone adds an
operation and forgets the agent surface, which is the failure it exists to
catch.

Two shapes of restriction matter here:

- **``credential_set`` accepts only reference forms**. It
  has no parameter capable of carrying a secret value, so a secret never enters
  an agent's context or any transcript its host retains — somewhere none of this
  product's guarantees reach. The CLI keeps prompt/stdin entry for humans; the
  asymmetry narrows the agent surface, never widens it. A reference is not
  automatically safe, though: ``from_env``/``from_file`` name *what* to read,
  and an unconstrained name/path made this tool a way to read anything the
  coordinator process could, including its own management token.
  ``secret_from`` (service/credentials.py),
  shared by the CLI and this tool, now refuses Tensorstead's own
  ``TENSORSTEAD_*`` variables and applies file-read hardening — this file has no
  separate copy of that restriction to keep in sync.
- **No shell, exec, file, or SSH tool exists**. Not filtered,
  not permission-gated: absent, so it cannot be reached for under pressure.

Long-running operations return an ``operation_id`` immediately; the agent polls
``operation_get``. Client polling is not the background polling this design
prohibits — that binds the product's own unprompted behaviour.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from tensorstead.mcp.client import ApiResult, CoordinatorClient


def register_tools(server: Any, client: CoordinatorClient) -> None:
    """Register every management tool on ``server``.

    ``server`` is an ``MCPServer`` from the official SDK; it is untyped here so
    this module imports no SDK internals beyond what the server passes in.
    """

    # ------------------------------------------------------------------ nodes
    @server.tool(name="node_register")
    def node_register(
        name: Annotated[str, Field(description="Name to register the host under.")],
        agent_endpoint: Annotated[
            str, Field(description="Agent base URL, e.g. https://host:8443.")
        ],
    ) -> ApiResult:
        """Register a host whose agent is already running."""
        return client.request(
            "POST", "/v1/nodes", json={"name": name, "agent_endpoint": agent_endpoint}
        )

    @server.tool(name="coordinator_version")
    def coordinator_version() -> ApiResult:
        """The coordinator's release and contract version.

        Worth asking before anything else when a call behaves unexpectedly: an
        agent and a coordinator can disagree about which operations exist, and
        the contract version is the only signal that says so. ``/health`` is the
        unauthenticated liveness check and deliberately does not disclose this.
        """
        return client.request("GET", "/v1/version")

    @server.tool(name="node_list")
    def node_list() -> ApiResult:
        """Inventory of registered nodes."""
        return client.request("GET", "/v1/nodes")

    @server.tool(name="node_get")
    def node_get(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
    ) -> ApiResult:
        """Detail of one node."""
        return client.request("GET", f"/v1/nodes/{node_id}")

    @server.tool(name="node_deregister")
    def node_deregister(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
    ) -> ApiResult:
        """Remove a node; refused while a deployment names it."""
        return client.request("DELETE", f"/v1/nodes/{node_id}")

    @server.tool(name="node_reachability")
    def node_reachability(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
    ) -> ApiResult:
        """On-demand reachability check. Never a background poll."""
        return client.request("GET", f"/v1/nodes/{node_id}/reachability")

    @server.tool(name="node_resources")
    def node_resources(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
    ) -> ApiResult:
        """On-demand accelerator, memory, and storage reading.

        Reported for judgement; never gates an operation.
        ``unknown`` and ``unreachable`` are ordinary values here, not errors.
        """
        return client.request("GET", f"/v1/nodes/{node_id}/resources")

    # --------------------------------------------------------------- runtimes
    @server.tool(name="runtime_list")
    def runtime_list() -> ApiResult:
        """Supported runtimes and their capabilities, incl. ``supports_distributed``."""
        return client.request("GET", "/v1/runtimes")

    # ----------------------------------------------------------------- models
    @server.tool(name="model_acquire")
    def model_acquire(
        source_id: Annotated[str, Field(description="Model source id, e.g. huggingface.")],
        source_model_id: Annotated[
            str, Field(description="Model identifier as the source knows it.")
        ],
        nodes: Annotated[list[str], Field(description="Node ids to acquire onto.")],
        revision: Annotated[
            str | None,
            Field(description="Immutable upstream revision to pin. Prefer over a moving tag."),
        ] = None,
        credential: Annotated[
            str | None, Field(description="Named credential; omit for the source default.")
        ] = None,
        file_selector: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Globs of files to acquire from the repository; omit for all of them. "
                    "A GGUF repository commonly ships many quantizations, and the "
                    "selection is part of the resulting model's identity."
                )
            ),
        ] = None,
    ) -> ApiResult:
        """Acquire a model onto nodes; returns an ``operation_id``.

        ``credential`` names a stored credential — it is a *name*, never a
        secret value. Omitting it applies the source's default.

        ``file_selector`` selects part of a repository. Two selections of one
        repository are two models with two store directories, because a record
        naming the repository while holding one of its files would describe
        something that does not exist.
        """
        payload: dict[str, Any] = {
            "source_id": source_id,
            "source_model_id": source_model_id,
            "nodes": nodes,
        }
        if revision is not None:
            payload["revision"] = revision
        if credential is not None:
            payload["credential"] = credential
        if file_selector:
            payload["file_selector"] = file_selector
        return client.request("POST", "/v1/models:acquire", json=payload)

    @server.tool(name="model_list")
    def model_list() -> ApiResult:
        """Models with identity, source, and resolved revision."""
        return client.request("GET", "/v1/models")

    @server.tool(name="model_get")
    def model_get(
        model_id: Annotated[str, Field(description="Model id from model_list.")],
    ) -> ApiResult:
        """Detail of one model, including per-node replica states."""
        return client.request("GET", f"/v1/models/{model_id}")

    @server.tool(name="model_delete")
    def model_delete(
        model_id: Annotated[str, Field(description="Model id from model_list.")],
    ) -> ApiResult:
        """Delete a model; refused while referenced."""
        return client.request("DELETE", f"/v1/models/{model_id}")

    # ----------------------------------------------------------------- images
    @server.tool(name="image_list")
    def image_list() -> ApiResult:
        """List container images present on nodes."""
        return client.request("GET", "/v1/images")

    @server.tool(name="image_delete")
    def image_delete(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
        digest: Annotated[str, Field(description="Image digest to delete.")],
    ) -> ApiResult:
        """Delete an image from the node, then the record.

        Returns ``outcome: removed`` when the node freed it and
        ``record_reaped`` when the node did not hold it.
        """
        return client.request("DELETE", f"/v1/images/{node_id}/{digest}")

    @server.tool(name="image_reconcile")
    def image_reconcile(
        node_id: Annotated[
            str | None, Field(description="Only this node. Default: every registered node.")
        ] = None,
    ) -> ApiResult:
        """Compare image records against the nodes; reap records whose image is gone.

        Reports images present with no record, and unreachable nodes, without
        acting on either.
        """
        path = "/v1/images:reconcile" + (f"?node_id={node_id}" if node_id else "")
        return client.request("POST", path)

    # ------------------------------------------------- inference credentials
    # Reference forms only. No parameter here can carry a secret value, so a
    # secret never enters an agent's context or any transcript its host keeps
    # -- somewhere none of this product's guarantees reach. The
    # CLI keeps prompt entry for humans; the asymmetry narrows the agent
    # surface, never widens it.
    @server.tool()
    def inferencekey_set(
        name: Annotated[str, Field(description="Name for this credential.")],
        from_env: Annotated[
            str | None,
            Field(description="Env var holding the secret. A reference, never the value."),
        ] = None,
        from_file: Annotated[
            str | None,
            Field(description="File holding the secret. A reference, never the value."),
        ] = None,
    ) -> ApiResult:
        """Store an inference credential from a reference."""
        return client.request(
            "PUT",
            f"/v1/inference-credentials/{name}",
            json={"from_env": from_env, "from_file": from_file},
        )

    @server.tool()
    def inferencekey_list() -> ApiResult:
        """List credential names. Values are never returned."""
        return client.request("GET", "/v1/inference-credentials")

    @server.tool()
    def inferencekey_delete(
        name: Annotated[str, Field(description="Credential to delete.")],
    ) -> ApiResult:
        """Delete a credential; refused while a deployment binds it."""
        return client.request("DELETE", f"/v1/inference-credentials/{name}")

    @server.tool()
    def inferencekey_bind(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
        name: Annotated[
            str | None,
            Field(description="Credential to bind. Null clears the binding."),
        ] = None,
    ) -> ApiResult:
        """Bind a credential to a deployment, or clear it.

        Changes a definition, never host state: a restart is required for it to
        take effect and is never performed implicitly.
        """
        return client.request(
            "PUT", f"/v1/deployments/{deployment_id}/inference-credential", json={"name": name}
        )

    # -------------------------------------------------- managed runtime images
    # These exist so an agent that researches a fix can apply it. Before them,
    # repairing a broken upstream image meant SSH -- outside the product, and
    # unrecorded.
    @server.tool()
    def buildspec_set(
        name: Annotated[str, Field(description="Name for the build spec.")],
        base_image: Annotated[
            str,
            Field(description="Base image. Pin with @sha256:... to be reproducible."),
        ],
        steps: Annotated[list[str], Field(description="Ordered build steps. Not a host command.")],
        entrypoint: Annotated[
            list[str] | None,
            Field(
                description=(
                    "The produced image's own ENTRYPOINT, in exec form. For an image "
                    "that must prepare itself before serving -- installing a "
                    "model-shipped file into the runtime, say -- since that work reads "
                    "the mounted model directory, which does not exist at build time. "
                    "Omit to leave the base image's entrypoint alone."
                )
            ),
        ] = None,
    ) -> ApiResult:
        """Record a build spec. Executes nothing.

        Reports whether the base is pinned; an unpinned base is not refused,
        only called unreproducible.
        """
        return client.request(
            "PUT",
            f"/v1/buildspecs/{name}",
            json={
                "base_image": base_image,
                "steps": steps,
                **({"entrypoint": entrypoint} if entrypoint else {}),
            },
        )

    @server.tool()
    def buildspec_list() -> ApiResult:
        """List recorded build specs."""
        return client.request("GET", "/v1/buildspecs")

    @server.tool()
    def buildspec_delete(
        name: Annotated[str, Field(description="Build spec to delete.")],
    ) -> ApiResult:
        """Delete a build spec; refused while referenced."""
        return client.request("DELETE", f"/v1/buildspecs/{name}")

    @server.tool()
    def image_build(
        spec: Annotated[str, Field(description="Recorded build spec to build.")],
        reference: Annotated[str, Field(description="Tag for the produced image.")],
        node_id: Annotated[
            str | None, Field(description="Single node to build on. Use nodes for several.")
        ] = None,
        nodes: Annotated[
            list[str] | None,
            Field(description="Build on the first, copy the produced image to the rest."),
        ] = None,
    ) -> ApiResult:
        """Build a recorded spec into an image.

        With ``nodes``, the image is produced **once** on the first and copied
        to the others. Building separately on each node gives the same spec a
        different identifier per node, which breaks digest comparison silently
        and makes divergence detection meaningless — so a deployment spanning
        nodes needs this, not several equivalent builds.

        This docstring previously claimed "produced once and distributed" while
        the tool accepted a single node and could do no such thing. The
        orchestration existed and was tested; nothing called it.

        **Returns an operation id, not a finished build**.
        A build routinely outlasts this client's read timeout, which reported
        ``coordinator_unreachable`` -- indistinguishable from a dead
        coordinator -- leaving the caller unable to tell an unaccepted request
        from one still running. Retrying blindly risked a duplicate concurrent
        build or a race with an accepted build's distribution phase. Poll
        ``operation_get`` to a terminal state; ``per_node`` carries which node
        produced the image and which received it.
        """
        payload: dict[str, Any] = {"spec": spec, "reference": reference}
        if nodes:
            payload["nodes"] = list(nodes)
        elif node_id:
            payload["node_id"] = node_id
        return client.request("POST", "/v1/images:build", json=payload)

    @server.tool()
    def image_import(
        node_id: Annotated[str, Field(description="Node id from node_list.")],
        reference: Annotated[str, Field(description="Tag for the imported image.")],
        archive_name: Annotated[
            str, Field(description="Archive file name in the node managed image store.")
        ],
        expected_image_id: Annotated[
            str | None,
            Field(description="Required image id; a mismatch imports nothing."),
        ] = None,
    ) -> ApiResult:
        """Import a prebuilt image archive onto a node."""
        return client.request(
            "POST",
            "/v1/images:import",
            json={
                "node_id": node_id,
                "reference": reference,
                "archive_name": archive_name,
                "expected_image_id": expected_image_id,
            },
        )

    # --------------------------------------------- code-execution approvals
    @server.tool()
    def code_approval_create(
        option: Annotated[str, Field(description="Option to authorize, e.g. trust_remote_code.")],
        runtime_type: Annotated[str, Field(description="Runtime, e.g. vllm or sglang.")],
        model_source_id: Annotated[str, Field(description="Model source, e.g. huggingface.")],
        source_model_id: Annotated[str, Field(description="Repository, e.g. nvidia/Model.")],
        model_revision: Annotated[
            str, Field(description="Immutable 40- or 64-char hex revision. Never a tag.")
        ],
        image_digest: Annotated[str, Field(description="Runtime image digest, sha256:...")],
        reason: Annotated[str, Field(description="Why this was approved. Required.")],
        approved_by: Annotated[str, Field(description="Who reviewed it. Required.")],
    ) -> ApiResult:
        """Authorize one code-loading option for one exact model/revision/image tuple.

        The authorization policy refuses ``trust_remote_code`` because "what
        executes on an appliance is not something a deployment record may decide
        **on its own**". This is the other authority: a separate, immutable,
        audited record. It is required for models whose vendor guidance needs the
        flag -- NVIDIA's for ``Qwen3.8-Flash-Next-NVFP4``, for one -- and without
        it the operator's only route is a hand-started container the coordinator
        knows nothing about.

        Binds to an **immutable** revision and an image **digest**. A tag is
        refused: an approval is granted for code that was read at one commit, and
        one that outlives the bytes it was granted for is not an approval. Change
        the model revision or the image and the tuple no longer matches, which is
        how authorization is lost -- not by anyone noticing, but by the key
        ceasing to fit.

        Only options classified ``loads-code`` can be approved. Every other
        refusal says the thing itself is wrong rather than unauthorized, and no
        review repairs that.

        Starts nothing and changes no deployment. Create the approval, then
        create or modify the deployment that needs it.
        """
        return client.request(
            "POST",
            "/v1/code-approvals",
            json={
                "option": option,
                "runtime_type": runtime_type,
                "model_source_id": model_source_id,
                "source_model_id": source_model_id,
                "model_revision": model_revision,
                "image_digest": image_digest,
                "reason": reason,
                "approved_by": approved_by,
            },
        )

    @server.tool()
    def code_approval_list() -> ApiResult:
        """Every reviewed code-execution approval, newest first."""
        return client.request("GET", "/v1/code-approvals")

    @server.tool()
    def code_approval_delete(
        approval_id: Annotated[str, Field(description="Approval id from code_approval_list.")],
    ) -> ApiResult:
        """Revoke an approval.

        Running deployments are not disturbed -- stopping inference is a larger
        action than withdrawing an authorization, and this product does not
        reverse work on a host without being asked. What revocation guarantees is
        that the next create, modify, or start finds nothing to match.
        """
        return client.request("DELETE", f"/v1/code-approvals/{approval_id}")

    # ------------------------------------------------------------ credentials
    @server.tool(name="credential_set")
    def credential_set(
        source_id: Annotated[str, Field(description="Model source id, e.g. huggingface.")],
        name: Annotated[str, Field(description="Unique name for the deployment.")],
        from_env: Annotated[
            str | None,
            Field(description="Env var holding the secret. A reference, never the value."),
        ] = None,
        from_file: Annotated[
            str | None,
            Field(description="File holding the secret. A reference, never the value."),
        ] = None,
        default: Annotated[
            bool, Field(description="Make this the source default credential.")
        ] = False,
    ) -> ApiResult:
        """Set a credential **by reference only**.

        There is deliberately no parameter that takes a secret value. Give
        ``from_env`` (an environment variable the coordinator reads) or
        ``from_file`` (a path the coordinator reads). The secret never passes
        through this tool call, so it never lands in an agent's context or in a
        transcript its host retains.

        Honest limitation: a human must still have placed that variable or
        file. What this buys is that an agent can finish a gated-model setup
        rather than halting.
        """
        if (from_env is None) == (from_file is None):
            return {
                "code": "invalid_request",
                "message": "give exactly one of from_env or from_file",
                "detail": {},
            }
        payload: dict[str, Any] = {"default": default}
        if from_env is not None:
            payload["from_env"] = from_env
        else:
            payload["from_file"] = from_file
        return client.request("PUT", f"/v1/credentials/{source_id}/{name}", json=payload)

    @server.tool(name="credential_list")
    def credential_list() -> ApiResult:
        """Credential names and default status — never values."""
        return client.request("GET", "/v1/credentials")

    @server.tool(name="credential_delete")
    def credential_delete(
        source_id: Annotated[str, Field(description="Model source id, e.g. huggingface.")],
        name: Annotated[str, Field(description="Credential name to delete.")],
        promote: Annotated[
            str | None, Field(description="Credential to promote to default instead.")
        ] = None,
    ) -> ApiResult:
        """Delete a credential; reports the consequence for the default."""
        params = {"promote": promote} if promote is not None else None
        return client.request("DELETE", f"/v1/credentials/{source_id}/{name}", params=params)

    # ------------------------------------------------------------ deployments
    @server.tool(name="deployment_create")
    def deployment_create(
        name: Annotated[str, Field(description="Unique name for the deployment.")],
        model_id: Annotated[str, Field(description="Id of an acquired model (see model_list).")],
        runtime_type: Annotated[str, Field(description="Runtime type from runtime_list.")],
        runtime_version: Annotated[
            str, Field(description="Runtime version label to record, e.g. latest.")
        ],
        image_reference: Annotated[
            str,
            Field(description="Runtime container image, e.g. registry/repo:tag."),
        ],
        participating_nodes: Annotated[
            list[str],
            Field(description="Node ids from node_list. Several need a distributed runtime."),
        ],
        endpoint: Annotated[
            str,
            Field(description="Where the runtime serves, host:port. Recorded, never proxied."),
        ],
        runtime_config: Annotated[
            dict[str, Any] | None,
            Field(description="Runtime-specific settings; bad keys list the accepted set."),
        ] = None,
    ) -> ApiResult:
        """Define a deployment; the host is untouched until start.

        **``restore_on_boot`` is deliberately absent from this surface**. The API
        and the CLI accept it; this tool does not,
        so an agent reaching the product only through MCP cannot express the
        intent at all -- not "is refused", cannot say it.

        The same narrowing ``credential_set`` already has, for the same reason.
        Boot restoration is the authority that turned a driver deadlock into a
        reimaged node: it decides what runs before anyone can log in, which makes
        it a machine authority wearing a deployment property's clothes.

        A surface that omits a field is only half the guard -- the other half is
        the node refusing to enable a unit its own provisioning did not permit,
        which is a credential no agent holds.
        """
        return client.request(
            "POST",
            "/v1/deployments",
            json={
                "name": name,
                "model_id": model_id,
                "runtime_type": runtime_type,
                "runtime_version": runtime_version,
                "image_reference": image_reference,
                "runtime_config": runtime_config or {},
                "participating_nodes": participating_nodes,
                "endpoint": endpoint,
            },
        )

    @server.tool(name="deployment_modify")
    def deployment_modify(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
        model_id: Annotated[
            str | None, Field(description="Id of an acquired model (see model_list).")
        ] = None,
        runtime_version: Annotated[
            str | None, Field(description="Runtime version label to record, e.g. latest.")
        ] = None,
        image_reference: Annotated[
            str | None,
            Field(description="Runtime container image, e.g. registry/repo:tag."),
        ] = None,
        endpoint: Annotated[
            str | None,
            Field(description="Where the runtime serves, host:port. Recorded, never proxied."),
        ] = None,
        participating_nodes: Annotated[
            list[str] | None,
            Field(description="Node ids from node_list. Several need a distributed runtime."),
        ] = None,
        runtime_config: Annotated[
            dict[str, Any] | None,
            Field(description="Runtime-specific settings; bad keys list the accepted set."),
        ] = None,
        replace_config: Annotated[
            bool,
            Field(
                description=(
                    "If true, replace the whole runtime_config map (keys not named "
                    "are dropped). If false (default), merge: only keys present "
                    "change, the rest are kept."
                ),
            ),
        ] = False,
        expected_revision: Annotated[
            int | None,
            Field(description="Refuse if the deployment moved since this revision."),
        ] = None,
    ) -> ApiResult:
        """Modify a definition — a new numbered revision.

        Reports ``restart_required`` without acting on it: nothing is restarted
        as an implicit consequence of a modification.

        ``runtime_config`` merges by default so a one-key modify keeps its
        neighbours; set ``replace_config`` to replace the whole map.
        """
        payload = {
            key: value
            for key, value in {
                "model_id": model_id,
                "runtime_version": runtime_version,
                "image_reference": image_reference,
                "endpoint": endpoint,
                "participating_nodes": participating_nodes,
                "runtime_config": runtime_config,
                "replace_config": replace_config,
                "expected_revision": expected_revision,
            }.items()
            if value is not None
        }
        return client.request("PATCH", f"/v1/deployments/{deployment_id}", json=payload)

    @server.tool(name="deployment_list")
    def deployment_list() -> ApiResult:
        """List declared deployments.

        ``name`` is a **stable handle chosen at creation, not a description of
        what is being served.** It is unique, it is what the CLI resolves a
        deployment by, and there is deliberately no rename: a deployment
        repointed to a different model keeps its name, the way a host keeps its
        hostname when it is reimaged.

        So do not read the name as the model. The model currently declared is
        at ``declared.revision.model.source_model_id``, and what the runtime
        advertises to clients is
        ``declared.revision.runtime_config.served_model_name``. Those three can
        legitimately disagree, and on a deployment that has been repointed they
        will.
        """
        return client.request("GET", "/v1/deployments")

    @server.tool(name="deployment_get")
    def deployment_get(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Declared and observed state as separate blocks, never merged.

        Declared is returned in full even when a node cannot be reached.
        """
        return client.request("GET", f"/v1/deployments/{deployment_id}")

    @server.tool(name="deployment_revisions")
    def deployment_revisions(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """List every retained revision."""
        return client.request("GET", f"/v1/deployments/{deployment_id}/revisions")

    @server.tool(name="deployment_status")
    def deployment_status(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Observed state, fetched now.

        ``unknown`` and ``unreachable`` are ordinary values. An agent that
        treats an unreachable node as a failed call would retry rather than
        report, which is the stale-state failure this design exists to prevent.
        """
        return client.request("GET", f"/v1/deployments/{deployment_id}/status")

    @server.tool(name="deployment_runtime")
    def deployment_runtime(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
        tail: Annotated[
            int,
            Field(
                description=(
                    "Lines of runtime output per node (1-2000). A runtime that logs "
                    "every HTTP request can bury a failure; raise this if what you "
                    "get back is all access logging."
                ),
                ge=1,
                le=2000,
            ),
        ] = 500,
    ) -> ApiResult:
        """The runtime's own account of itself: argv, relaunches, recent output.

        ``deployment_status`` answers whether a deployment is serving. This
        answers *why it is not* — the runtime's actual output, the arguments it
        was actually launched with, and how many times it has been relaunched.

        This is the tool to reach for when ``deployment_status`` reports
        ``inference_ready: false`` or ``not_running``. Before it existed the
        only way to that answer was a shell on the node, which is not something
        an agent has or should have.

        The log tail is the runtime's words, unparsed. Read it; do not pattern
        -match it into a verdict — a runtime that is still loading a large model
        looks much like one that has failed, and only its own output tells them
        apart.

        Nothing here is stored by the product. It is read from the node when
        asked and kept nowhere, so a second call after the runtime goes quiet
        returns nothing rather than a cached copy.
        """
        return client.request(
            "GET", f"/v1/deployments/{deployment_id}/runtime", params={"tail": tail}
        )

    @server.tool(name="deployment_export")
    def deployment_export(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
        revision: Annotated[
            int | None, Field(description="Revision to export. Omit for the current one.")
        ] = None,
    ) -> ApiResult:
        """Export a revision — the artifact that outlives this session.

        This is what makes an agent's work retrievable by a human who never saw
        the transcript.
        """
        params = {"revision": revision} if revision is not None else None
        return client.request("GET", f"/v1/deployments/{deployment_id}/export", params=params)

    @server.tool(name="deployment_revision")
    def deployment_revision(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
        revision: Annotated[int, Field(description="Revision number from deployment_revisions.")],
    ) -> ApiResult:
        """Fetch one retained revision in full.

        ``deployment_revisions`` lists what exists; this is how an agent reads
        the configuration a past revision actually held — which is what
        answering "what changed, and what was it before?" requires.
        """
        return client.request("GET", f"/v1/deployments/{deployment_id}/revisions/{revision}")

    @server.tool(name="deployment_create_from_export")
    def deployment_create_from_export(
        export: Annotated[
            dict[str, Any],
            Field(description="An export artifact from deployment_export, verbatim."),
        ],
    ) -> ApiResult:
        """Recreate a deployment from an export artifact.

        The inverse of ``deployment_export``, and its absence made that a
        one-way door: an agent could produce the artifact this product treats as
        the portable definition of a deployment and had no way to consume one.
        Reproducing a deployment on another estate — or restoring the one you
        exported before changing it — meant leaving the management plane.

        Returns an operation id. Comparability warnings, if any, arrive on the
        operation record rather than blocking the call: they inform, they do not
        gate.
        """
        return client.request("POST", "/v1/deployments:create-from-export", json={"export": export})

    @server.tool(name="deployment_start")
    def deployment_start(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Start a deployment."""
        return client.request("POST", f"/v1/deployments/{deployment_id}:start")

    @server.tool(name="deployment_stop")
    def deployment_stop(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Stop a deployment."""
        return client.request("POST", f"/v1/deployments/{deployment_id}:stop")

    @server.tool(name="deployment_restart")
    def deployment_restart(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Restart a deployment — stop then start."""
        return client.request("POST", f"/v1/deployments/{deployment_id}:restart")

    @server.tool(name="deployment_reconcile")
    def deployment_reconcile(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Converge toward declared state, explicitly.

        The only operation that mutates in response to divergence, and only
        because it was asked for.
        """
        return client.request("POST", f"/v1/deployments/{deployment_id}:reconcile")

    @server.tool(name="deployment_remove")
    def deployment_remove(
        deployment_id: Annotated[str, Field(description="Deployment id from deployment_list.")],
    ) -> ApiResult:
        """Remove a deployment; model artifacts and image are retained."""
        return client.request("DELETE", f"/v1/deployments/{deployment_id}")

    # ------------------------------------------------------------- operations
    @server.tool(name="operation_get")
    def operation_get(
        operation_id: Annotated[str, Field(description="Operation id from a long-running tool.")],
    ) -> ApiResult:
        """Progress and terminal outcome of one operation."""
        return client.request("GET", f"/v1/operations/{operation_id}")

    @server.tool(name="operation_list")
    def operation_list(
        deployment_id: Annotated[str | None, Field(description="Filter to one deployment.")] = None,
    ) -> ApiResult:
        """List operations, optionally filtered by deployment."""
        params = {"deployment_id": deployment_id} if deployment_id is not None else None
        return client.request("GET", "/v1/operations", params=params)
