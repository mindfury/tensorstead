---
name: tensorstead-model-swap
description: Use before changing which model a Tensorstead deployment serves — swapping, upgrading or replacing the model behind an endpoint or a served model name. Covers creating a new deployment instead of modifying the running one, checking capacity, cutting clients over, and rolling back.
---

# Swapping the model a Tensorstead deployment serves

**A different model is a new deployment.** Never change `model_id` on a
running deployment: the coordinator refuses it (`model_change_refused`), and
even on a stopped one the old name, endpoint, runtime config and host config
carry over to a model they were not chosen for.

1. **Read the full runbook first:** call the MCP tool `runbook_get` with
   `name: "model-swap"`. It is the authoritative procedure and is kept in step
   with the server; this skill is the short form.
2. **Record what is serving:** `deployment_get` and `deployment_export` on the
   current deployment.
3. **Acquire the new model** at a pinned revision: `model_acquire`, then poll
   `operation_get`.
4. **Check room on the node:** `node_resources`. If both models will not fit,
   stop and tell the operator the swap needs downtime.
5. **Create a new deployment:** new name, free endpoint, a runtime config
   chosen for the new model (do not copy `host_config`), and a served model
   name that differs from the running one's (`served_name_conflict` otherwise).
6. **Start it, wait until `inference_ready` is true** (`deployment_status`);
   read `deployment_runtime` if it does not settle rather than retrying.
7. **Move clients, then stop the old deployment.** Keep it as the rollback;
   remove it only when the operator accepts the new one.

Rollback: stop the new deployment, start the old one.

On a refusal, read its `code` and `detail` — they name the conflicting
deployment and the runbook — and change the request; do not loop.
