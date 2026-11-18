---
title: Operate Tensorstead through an AI agent over MCP
sop_id: SOP-16
applies_to: Tensorstead 1.13
keywords: [MCP, AI agent, tools, operation_get, runbook_get, model-swap skill, hosted MCP, stdio, tool mapping, CLI equivalent, agent guardrails]
---

# SOP-16: Operate Tensorstead through an AI agent over MCP

## How an AI agent connects to Tensorstead

Use the hosted endpoint `https://<coordinator>:8090/mcp` with the private CA and
a bearer management token, or run `tensorstead-mcp` locally over stdio (SOP-02).
The MCP surface has the same operations as the CLI, plus runbooks, and no
shell, SSH, exec or arbitrary-path tool.

## Rules for an AI agent operating Tensorstead

1. Inspect before acting: `node_list`, `runtime_list`, `model_list`,
   `deployment_list`.
2. Long-running tools return an `operation_id`. Poll `operation_get` until it
   is terminal.
3. Report declared and observed separately. `unknown` and `unreachable` are
   answers to report, not errors to retry.
4. Read `runbook_get('model-swap')` before changing which model a deployment
   serves.
5. On a refusal, read `code` and `detail` and change the request. Do not loop.
6. Never put a token or key in a tool argument except through the documented
   reference mechanisms (`from_env`, import directory file names).
7. Stop for the operator's approval before anything that causes downtime, and
   never unreserve a node.
8. Run no host-changing playbook; those are operator actions outside MCP.

## CLI to MCP tool mapping

| CLI | MCP tool |
| --- | --- |
| `stead status` | `coordinator_version` |
| `node register` / `list` / `show` | `node_register` / `node_list` / `node_get` |
| `node reachability` / `resources` / `deregister` | `node_reachability` / `node_resources` / `node_deregister` |
| `node reserve` / `unreserve` / `rotate-management-token` | CLI and API only |
| `model acquire` / `list` / `show` / `digest` / `delete` | `model_acquire` / `model_list` / `model_get` / `model_digest` / `model_delete` |
| `runtime list` | `runtime_list` |
| `deployment create` | `deployment_create` |
| `deployment create --from-export` | `deployment_create_from_export` |
| `deployment modify` / `rename` | `deployment_modify` / `deployment_rename` |
| `deployment revisions` / `export` | `deployment_revisions`, `deployment_revision` / `deployment_export` |
| `deployment start` / `stop` / `restart` | `deployment_start` / `deployment_stop` / `deployment_restart` |
| `deployment reconcile` / `remove` | `deployment_reconcile` / `deployment_remove` |
| `deployment list` / `show` / `status` / `runtime` | `deployment_list` / `deployment_get` / `deployment_status` / `deployment_runtime` |
| `image list` / `pull` / `build` / `import` | `image_list` / `image_pull` / `image_build` / `image_import` |
| `image delete` / `reconcile` | `image_delete` / `image_reconcile` |
| `buildspec set` / `list` / `delete` | `buildspec_set` / `buildspec_list` / `buildspec_delete` |
| `credential set` / `list` / `delete` | `credential_set` / `credential_list` / `credential_delete` |
| `inferencekey set` / `list` / `delete` / `bind` | `inferencekey_set` / `inferencekey_list` / `inferencekey_delete` / `inferencekey_bind` |
| `operation list` / `show` | `operation_list` / `operation_get` |
| (none) | `code_approval_create` / `code_approval_list` / `code_approval_delete` |
| (none) | `runbook_list` / `runbook_get` |

## Typical agent sequence: deploy a model

1. `node_list`, `node_resources` on the target node.
2. `model_acquire` with a pinned revision; poll `operation_get`.
3. `deployment_create` with an unused endpoint and a distinct served model name.
4. `deployment_start`; poll `operation_get`.
5. Poll `deployment_status` until `inference_ready` is true; on trouble read
   `deployment_runtime`.
6. Report the endpoint, served name, and declared and observed state.

## Repairing a runtime image as an agent

`buildspec_set` (records only), then `image_build` naming every node the
deployment uses, then `deployment_modify` with the new image, then
`deployment_restart` after operator agreement. `image_import` takes an archive
file name inside the node's managed store, never a path.

## Skills for agent hosts

`skills/tensorstead-model-swap/SKILL.md` is the short form of the model-swap
runbook for agent hosts that load skills. It sends the agent to `runbook_get`
for the full procedure.

## What the audit trail records about agents

Agent actions are recorded exactly like human actions, as the same operation
records. They are not attributed to a person or agent, because authentication
is one shared token. Code approvals created by an agent are attestations, not
independent human sign-off (`docs/adr/0001-trust-boundaries.md`).
