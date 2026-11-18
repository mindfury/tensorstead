---
title: Swap the model a Tensorstead endpoint serves
sop_id: SOP-08
applies_to: Tensorstead 1.13
keywords: [model swap, upgrade model, replace model, new deployment, model_change_refused, served_name_conflict, endpoint_conflict, node_reserved, cutover, rollback, runbook model-swap]
---

# SOP-08: Swap the model a Tensorstead endpoint serves

## The rule for swapping models

**A different model is a new deployment.** Do not change the model of an
existing deployment. Its name, endpoint, runtime config and `host_config` were
chosen for the old model. Changing the model of a running deployment is refused
with `model_change_refused`. The authoritative version of this procedure is
`docs/model-swap.md`, also served to AI agents by MCP `runbook_get('model-swap')`.

## Step 1: Record what is serving now (read-only)

```sh
stead deployment show <old>
stead deployment export <old> -o <old>.yaml
```

Note the node, endpoint, served model name, runtime, image, runtime config and
host config.

## Step 2: Acquire the new model at a pinned revision

```sh
stead model acquire --source huggingface --id <org/new-model> --revision <sha> --node <node>
stead operation show <operation id>
stead model list
```

## Step 3: Check both models fit (read-only)

```sh
stead node resources <node>
```

On a DGX Spark both models share one memory pool. If both cannot fit, the swap
needs downtime. Tell the operator and get agreement before stopping anything.
Do not find out by starting and failing.

## Step 4: Create the new deployment

```sh
stead deployment create --name <new-model-and-date> --model <new model> \
  --runtime <runtime> --image <image> --node <node> \
  --endpoint <node>:<free port> --config served_model_name=<new served name> ...
```

- Use a new name that says what it serves, not the old name plus a suffix.
- Use a free port. The old endpoint is refused (`endpoint_conflict`) while the
  old deployment exists.
- Choose runtime config for the new model, starting from the runtime defaults
  and the model's own recommendations. Do not copy `host_config`, speculative
  decoding, parsers or memory settings from the old deployment unless you know
  they apply.
- Use a different `served_model_name` unless you will stop the old deployment
  first (Step 7).

## Step 5: Start it and wait for inference_ready

```sh
stead deployment start <new>
stead deployment show <new>      # until inference_ready: true
stead deployment runtime <new>   # if it does not settle; do not retry blindly
```

## Step 6: Move clients

Either point clients at the new endpoint and served name, or keep the old
served name by stopping the old deployment, then:

```sh
stead deployment modify <new> --config served_model_name=<old served name>
stead deployment restart <new>
```

That variant causes a short outage. Announce it first.

## Step 7: Stop the old deployment and keep it

```sh
stead deployment stop <old>
```

Keep it as the rollback. Remove it (SOP-14) only after the operator accepts the
new deployment.

## Roll back a model swap

```sh
stead deployment stop <new>
stead deployment start <old>
```

Stop the new one first when the two share a served name. The old definition is
unchanged because nothing modified it.

## Refusals during a model swap

| Code | Meaning | Action |
| --- | --- | --- |
| `model_change_refused` | Tried to change the model of a running deployment | Create a new deployment |
| `served_name_conflict` | Another running deployment serves that name | Choose another name, or stop the other one first |
| `endpoint_conflict` | That node and port are taken | Pick a free port |
| `node_reserved` | The operator reserved the node | Ask the operator; do not unreserve it yourself |
