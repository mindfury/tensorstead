---
title: Tensorstead troubleshooting and error code reference
sop_id: SOP-15
applies_to: Tensorstead 1.13
keywords: [error code, refusal, troubleshooting, agent_unreachable, partial_failure, endpoint_conflict, served_name_conflict, model_change_refused, code_execution_not_authorized, replication_failed, digest_mismatch, insufficient_memory, outcome_unknown, coordinator_unreachable, not serving, inference_ready false]
---

# SOP-15: Tensorstead troubleshooting and error code reference

## First steps for any failure

1. Read the refusal's `code` and `detail`. They name what failed, where, and
   usually what to change.
2. `stead operation show <id>` for the full record, including per-node outcomes.
3. `stead deployment show <name>` for declared and observed state.
4. `stead deployment runtime <name>` when a runtime will not serve.
5. Do not retry a refusal unchanged, and do not go to the host over SSH before
   reading these.

## Deployment started but is not serving

| Symptom | Check | Typical fix |
| --- | --- | --- |
| `inference_ready: false` for minutes | `deployment runtime` output | Large models can take about 20 minutes to load |
| Out of memory in runtime output | `node resources` | Lower `gpu_memory_utilization` / `mem_fraction_static`, or stop another deployment |
| Unknown or invalid flag | runtime output | Fix `--config`; check the image's runtime version |
| Missing Python module | runtime output | Build an image that adds it (SOP-05) |
| Tool calls return 200 but no `tool_calls` | `tool_call_parser` | Use the parser matching the model's format |
| Multi-node group does not form | `distributed_interface`, both nodes reachable | Set the fabric interface; acquire the model on every node |
| io_uring "Operation not permitted" | runtime output | `allow_io_uring` with a code approval (SOP-10) |

## Error code reference

| Code | Meaning | Action |
| --- | --- | --- |
| `already_in_state` | The deployment is already in the requested state | None; this is success |
| `agent_unreachable` | The node's agent did not answer | `make services-status`; check DNS and the registered URL |
| `agent_timeout` | The agent did not answer in time | Check the operation later; large transfers are long |
| `agent_version_incompatible` | Agent and coordinator contract versions differ | Upgrade the older side (SOP-13) |
| `coordinator_unreachable` | The client could not reach or verify the coordinator | Check URL and CA bundle (SOP-02) |
| `unauthorized` / `authorization_refused` / HTTP 401 | Bad management token, or upstream refused a model download | Fix the token file; or set a source credential and accept the model's terms |
| `not_found` / `node_not_found` | No such record | Check the name or id prefix |
| `invalid_request` | Malformed request, or JSON nested too deep | Fix the request |
| `request_too_large` (413) | Body over 16 MiB | Reduce the request |
| `busy` (429) | Eight background operations already running | Retry when one finishes |
| `invalid_runtime` | Unknown runtime type | See `stead runtime list` |
| `invalid_runtime_config` | Unknown or invalid config key | Fix `--config` (SOP-06) |
| `runtime_not_distributed` | Multi-node requested on a single-node runtime | Use one node or vllm/sglang/tensorfold |
| `distributed_interface_unusable` | The named fabric interface cannot be used | Check the interface name on each node |
| `endpoint_conflict` | Node and port already held by another deployment | Choose a free port |
| `served_name_conflict` (409) | Another running deployment serves that model name | Use another name, or stop the other first |
| `model_change_refused` (409) | Tried to change the model of a running deployment | Create a new deployment (SOP-08) |
| `node_reserved` | The node is reserved | Ask the operator |
| `code_execution_not_authorized` | Gated option without a matching approval | Create a code approval (SOP-10) |
| `credential_not_enforceable` | An inference key is bound to a runtime that cannot enforce it | Unbind the key or use vLLM, llama.cpp or TensorFold |
| `insufficient_memory` (422) | vLLM `gpu_memory_utilization` or SGLang `mem_fraction_static` exceeds the share of node memory free now | Lower the fraction to fit what is free, stop another deployment, or use another node |
| `still_referenced` | Delete refused; a retained deployment revision uses it | Remove the referring deployment first |
| `image_in_use` | Image referenced by a deployment revision | Remove the referring deployment first |
| `image_not_present` | Image not on the node | `image pull` or `image build` (SOP-05) |
| `image_staging_failed` / `image_distribution_failed` | Copying an image between nodes failed | Check node-to-node reachability; read the per-node reason |
| `image_pull_failed` / `image_build_failed` | Registry pull or build failed | Read the operation detail; pin the base; fix the failing step |
| `acquisition_in_progress` (409) | Same repository already being acquired | Wait, then re-run |
| `model_acquire_failed` | Download failed | Read the detail; check credential and revision |
| `unknown_model_source` | Source id not supported | Use `huggingface` or `azureml` |
| `replication_failed` | Node-to-node model copy failed | Fix reachability; or `--allow-upstream-fallback` |
| `digest_mismatch` | Bytes differ from the recorded digest | Investigate the replica; re-acquire |
| `model_digest_failed` | Digest computation failed | Read the detail |
| `partial_failure` | Some nodes changed, others did not | Read per-node outcomes; fix and reconcile (SOP-11); nothing was rolled back |
| `start_failed` / `stop_failed` / `restart_failed` / `remove_failed` | Lifecycle step failed on a node | Read the per-node reason and `deployment runtime` |
| `group_workers_unavailable` / `rendezvous_unavailable` | Multi-node group could not form | Check every node is up and on the fabric |
| `outcome_unknown` | The coordinator restarted mid-operation | Inspect the deployment; reconcile if needed |

## Connection problems

See SOP-02 for CLI and MCP connection errors (`certificate verify failed`,
`not a Tensorstead coordinator`, empty MCP lists).

## A node that does not answer at all

See SOP-12, "When a node is wedged".
