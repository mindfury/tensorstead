---
title: Tensorstead SOP manual - index
sop_id: SOP-INDEX
applies_to: Tensorstead 1.13
keywords: [index, standard operating procedure, manual, runbook, how to]
---

# Tensorstead SOP manual

Standard operating procedures for deploying and operating inference models with
Tensorstead. Each file is one procedure and can be read on its own. Use these
when you need the steps for a task. Use the reference docs one directory up
(`docs/operations.md`, `docs/builds.md` and so on) when you need the reasoning
behind a behaviour.

## How the procedures are written

- Every SOP starts with YAML front matter: `sop_id`, `title`, `applies_to`, and
  `keywords`. Every section restates what it is about, so a section still makes
  sense when it is retrieved without the rest of its file.
- Commands use the `stead` CLI. Every CLI operation has an MCP equivalent with
  the same behaviour; [SOP-16](16-operate-through-an-ai-agent.md) lists them.
- `<angle brackets>` are placeholders. Example host names (`spark-01`,
  `spark-alpha.internal`) and ports are illustrative.
- **Host-changing** marks a step that changes a machine. **Read-only** marks a
  step that does not.

## Procedures

| SOP | Title | Use it when |
| --- | --- | --- |
| [SOP-00](00-concepts-and-glossary.md) | Concepts and glossary | You need to know what a term means or how the parts fit |
| [SOP-01](01-install-the-control-plane.md) | Install the control plane | Putting the coordinator, MCP server and agents on hosts |
| [SOP-02](02-connect-the-cli-and-mcp.md) | Connect the CLI and MCP clients | Pointing `stead` or an agent at a coordinator |
| [SOP-03](03-manage-nodes.md) | Manage nodes | Registering, inspecting, reserving or removing a host |
| [SOP-04](04-acquire-models.md) | Acquire and place models | Downloading weights, copying them to more nodes, deleting them |
| [SOP-05](05-manage-runtime-images.md) | Manage runtime images | Pulling, building, importing or deleting a container image |
| [SOP-06](06-deploy-a-model.md) | Deploy a model | Creating, starting and verifying a deployment |
| [SOP-07](07-change-a-deployment.md) | Change a deployment | Modifying config or image, restarting, renaming, reverting |
| [SOP-08](08-swap-the-served-model.md) | Swap the served model | Replacing the model behind an endpoint |
| [SOP-09](09-credentials-and-inference-keys.md) | Credentials and inference keys | Gated models, protecting an endpoint with a key |
| [SOP-10](10-code-approvals.md) | Code approvals | `trust_remote_code` or `allow_io_uring` is refused |
| [SOP-11](11-monitor-inspect-and-reconcile.md) | Monitor, inspect and reconcile | Checking health, finding drift, reading history |
| [SOP-12](12-services-reboots-and-cold-start.md) | Services, reboots and cold start | Restarting services, rebooting a node, recovering after power loss |
| [SOP-13](13-upgrade-tensorstead.md) | Upgrade Tensorstead | Shipping a new Tensorstead release to the estate |
| [SOP-14](14-decommission.md) | Decommission | Removing deployments, models, images and nodes |
| [SOP-15](15-troubleshooting-and-error-codes.md) | Troubleshooting and error codes | A command failed or a deployment is not serving |
| [SOP-16](16-operate-through-an-ai-agent.md) | Operate through an AI agent | Driving Tensorstead over MCP |

## Rules that apply to every procedure

1. **Nothing happens unless you ask.** Tensorstead does not poll, auto-restart,
   auto-reconcile, schedule, or retry. If a step needs a restart, you run it.
2. **Declared and observed are separate.** Declared is what you asked for.
   Observed is what the host reports now. Drift is reported, not corrected.
3. **Secrets never go on a command line.** Tokens, keys and passwords come from
   a prompt, stdin, a protected file, or an environment variable named by
   reference. Never put them in Git, chat, an inventory or `--config`.
4. **Pin everything.** Model revisions are commit SHAs, not `main`. Images are
   `repo@sha256:...`, not a moving tag.
5. **Read the refusal.** Failures carry a `code` and a `detail` that say what to
   change. Do not retry a refusal in a loop.
6. **Stop for approval before host-changing playbooks** (`make deploy`,
   `make reboot`, `make prepare-nodes`, `site.yml`).
