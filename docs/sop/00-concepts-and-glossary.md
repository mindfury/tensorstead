---
title: Tensorstead concepts and glossary
sop_id: SOP-00
applies_to: Tensorstead 1.13
keywords: [glossary, architecture, coordinator, agent, node, deployment, revision, declared, observed, drift, operation, runtime, ports]
---

# SOP-00: Tensorstead concepts and glossary

## What Tensorstead is

Tensorstead is a management plane for self-hosted AI inference hosts. It
acquires model weights, places them on hosts, and starts, stops and restores
inference runtimes (vLLM, SGLang, llama.cpp, ExLlama, TensorRT-LLM,
TensorFold) as you declared them. It keeps an exportable record of every
change. It is **not in the inference data path**: clients send inference
requests directly to the runtime's own endpoint.

The tested target is a two-node NVIDIA DGX Spark cluster (GB10, arm64,
ConnectX-7 fabric). Other Linux hosts with an NVIDIA GPU, Docker and the NVIDIA
Container Toolkit should work but are untested.

## Components of a Tensorstead installation

| Component | What it does | Default port |
| --- | --- | --- |
| Coordinator (`tensorstead-coordinator`) | FastAPI service with a SQLite store. Holds nodes, models, images, deployments, revisions, credential references and the operation log. The single source of truth. | 8080 (HTTPS) |
| MCP server (`tensorstead-mcp`) | Streamable HTTP MCP endpoint for AI agents, beside the coordinator. Same operations as the CLI. | 8090, path `/mcp` |
| Node agent (`tensorstead-agent`) | One per inference host. Downloads models straight from the source, runs containers through Docker, writes systemd units for deployments. Runs as root. | 8443 (HTTPS) |
| Agent content port | Optional second agent port that serves model bytes to peer agents during node-to-node copies. | 8444 when `tensorstead_agent_content_port` is set |
| `stead` CLI | Human client of the coordinator API. | — |
| `tensorstead-mcp` (stdio) | Local MCP server; another client of the coordinator API. | — |
| Ansible playbooks | Optional external tooling that installs the services on already-provisioned Linux hosts. | — |

Default on-host paths: model store `/var/lib/tensorstead/models`, image store
`/var/lib/tensorstead/images`, agent state `/var/lib/tensorstead/state`,
coordinator database `/var/lib/tensorstead/coordinator/tensorstead.db`, config
and TLS files `/etc/tensorstead`, software `/opt/tensorstead/venv`.

## Glossary

- **Node**: a registered inference host running a node agent.
- **Model**: an acquired model artifact, identified by source (`huggingface`
  or `azureml`), source model id, resolved revision and optional file
  selection. A model has **replicas**, one per node that holds it.
- **Resolved revision**: the concrete, immutable revision the agent recorded
  for a model (a commit SHA for Hugging Face; `<version>@<manifest digest>` for
  Azure ML).
- **Content digest**: a hash of the model's file tree. Node-to-node copies are
  verified against it.
- **Runtime**: the inference server type (`vllm`, `sglang`, `llamacpp`,
  `exllama`, `trtllm`, `tensorfold`). `stead runtime list` shows them.
- **Image**: the container image a runtime runs from, held on a node.
- **Build spec**: a recorded recipe (pinned base image plus ordered steps) that
  Tensorstead can build into an image on a node. It is data and runs nothing
  until `image build`.
- **Deployment**: a named definition of model, runtime, image, nodes, endpoint
  and runtime config, plus a desired state (`running` or `stopped`).
- **Revision**: an immutable, numbered version of a deployment definition.
  Every accepted change inserts a new revision. None is ever edited or deleted.
- **RUNNING/CURRENT**: in `deployment list`, the revision running followed by
  the newest declared revision. `3/3` is current. `2/3` means revision 3 is
  declared but revision 2 still runs, which normally means a restart is
  needed. `—/3` means nothing is running.
- **Declared state**: what you asked for, from the coordinator's record.
- **Observed state**: what the node reports at the moment you ask. Never cached.
  An unreachable node shows `unreachable`, never a stale value.
- **Drift / divergence**: observed differs from declared. It is reported and
  left alone until you run `deployment reconcile`.
- **`inference_ready`**: per-node flag in observed state, true when the runtime
  answers inference requests, not just when its container is up.
- **Operation**: the record of a management action, with progress and a
  terminal outcome. Long actions return an operation id to poll.
- **Endpoint**: `host:port` the runtime serves on. Recorded, never proxied.
- **Served model name**: the name clients pass as `model`. Two running
  deployments cannot serve the same name (`served_name_conflict`).
- **Boot restoration**: a deployment's systemd unit enabled to start at boot.
  Requires both node permission (`TENSORSTEAD_ALLOW_BOOT_RESTORATION`) and the
  deployment's `--restore-on-boot`. Both default to off.
- **Reserved node**: a node declared off-limits to new work. Create, modify and
  start refuse it.
- **Management token**: the bearer token every coordinator API and MCP call
  carries. Whoever holds it has full control of the estate. There are no
  per-user permissions.
- **Replication token**: a separate token agents use to copy model bytes and
  images between themselves. It must differ from the management token.
- **Source credential**: a stored secret for downloading from a model source
  (for example a Hugging Face token). Stored by reference, never returned.
- **Inference key**: an optional bearer key the runtime endpoint requires from
  inference clients.
- **Code approval**: a recorded attestation that permits one code-loading or
  isolation-relaxing option for one exact model revision and image digest.

## Behaviours to expect from Tensorstead

- **Idempotent lifecycle**: starting a running deployment returns
  `already_in_state` and exit code 0.
- **Modify never restarts**: a change that needs a restart reports
  `restart_required: true`.
- **Multi-node partial failure is not rolled back**: if some nodes change and
  others do not, the result is `partial_failure`, the desired state is not
  recorded as changed, and the nodes that changed show as drift.
- **Interrupted operations**: operations in flight when the coordinator stops
  are marked `failed` with `outcome_unknown` on restart. They are not retried.
- **Inference survives a coordinator outage**: containers and their units do
  not depend on the coordinator.
- **Nothing comes back after a reboot by default**: services are not enabled
  at boot and deployments are not restored at boot unless configured.
- **No scheduler, no capacity planning**: you choose the node, and resource
  readings only inform you. One start-time check exists: a vLLM
  `gpu_memory_utilization` or SGLang `mem_fraction_static` larger than the
  share of node memory free right now is refused with `insufficient_memory`.
- **Deliberately absent**: deployment rollback command, operation cancellation,
  runtime-config redaction, automatic recovery of interrupted operations, RBAC,
  SSO, high availability, autoscaling. See `docs/not-in-v1.md`.
