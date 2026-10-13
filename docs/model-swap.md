# Model swap

Serve a different model in place of the one a deployment serves now, without
breaking the running one and with a way back.

**The rule: a different model is a new deployment.** A deployment's name,
endpoint, runtime config and host config were chosen for its model. Repointing
it at another model keeps all of them. That is how an agent left a deployment
crashing on warmup with the previous model's `host_config`. The coordinator
therefore refuses to change the model of a running deployment
(`model_change_refused`), and refuses to start a second deployment under a
model name another running deployment already serves (`served_name_conflict`).

## Before you start

1. **Write down what is serving now.** `deployment_get` on the current
   deployment. Note its node, endpoint, served model name, runtime, image,
   `runtime_config` and `host_config`. `deployment_export` it so the exact
   definition is on record.
2. **Get the new model onto the node.** `model_acquire` with a pinned
   `revision` (a commit, not `main`). Poll `operation_get` until it is
   terminal, then find its id in `model_list`.
3. **Check there is room for both.** `node_resources` on the node. On a unified
   memory node (DGX Spark) the old and new models share one pool. If both
   cannot fit, the swap needs downtime: say so to the operator and get their
   agreement before stopping anything. Do not discover it by starting and
   failing.

## Swap

4. **Create a new deployment.** `deployment_create` with:
   - a name that says what it serves (for example the model and the date), not
     the old deployment's name with a suffix;
   - a free endpoint, a port nobody else uses on that node. The old one's
     endpoint is refused while the old one exists (`endpoint_conflict`);
   - a `runtime_config` chosen for the **new** model. Start from the runtime's
     defaults and the model's own recommendations. Do not copy `host_config`,
     speculative decoding, parsers or memory settings from the old deployment
     unless you know they apply to this model;
   - a `served_model_name` that differs from the old one's, unless the old one
     will be stopped first (see step 7).
5. **Start it.** `deployment_start`, then poll `operation_get`. The operation
   finishing means containers were started, not that the model is serving.
6. **Wait until it serves.** Poll `deployment_status` (or `deployment_get`)
   until the node reports `inference_ready: true`. If it does not settle in a
   few minutes for a small model, or about 20 for a large one, read
   `deployment_runtime` for the runtime's own output instead of retrying.
7. **Move clients.** Either point clients at the new endpoint and served name,
   or keep the old served name: stop the old deployment first
   (`deployment_stop`), then `deployment_modify` the new one's
   `served_model_name` to the old name and `deployment_restart` it. That is a
   short outage; say so before doing it.
8. **Stop the old deployment.** `deployment_stop` once nothing needs it. Keep
   it: it is the rollback. `deployment_remove` only when the operator says the
   new one is accepted.

## Roll back

`deployment_stop` the new deployment, then `deployment_start` the old one,
in that order when they share a served name. The old definition is unchanged
because nothing modified it.

## Do not

- `deployment_modify` a running deployment's `model_id`. It is refused. Stopping
  it first so the change is accepted is still the in-situ swap this runbook
  replaces; it keeps the old config and the old name (one deployment was
  called `qwen36-27b` while it served Nemotron for the rest of its life).
- Copy `host_config` from another model's deployment.
- Remove the old deployment before the new one is verified.
- Retry a refused or failed start in a loop. Read the code and `detail`; they
  say what to change.

## Refusals

| Code | Meaning | What to do |
|---|---|---|
| `model_change_refused` | The modify changed the model of a running deployment | Create a new deployment (step 4) |
| `served_name_conflict` | Another running deployment serves under this model name | Choose another `served_model_name`, or stop the other one first (step 7) |
| `endpoint_conflict` | The node+endpoint is held by another deployment | Pick a free port |
| `node_reserved` | The operator reserved the node | Ask the operator; do not unreserve it yourself |
