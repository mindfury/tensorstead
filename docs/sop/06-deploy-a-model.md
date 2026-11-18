---
title: Deploy a model with Tensorstead
sop_id: SOP-06
applies_to: Tensorstead 1.13
keywords: [deployment create, deployment start, deployment show, inference_ready, endpoint, served_model_name, runtime config, --config, vllm, sglang, llamacpp, exllama, trtllm, tensorfold, multi-node, tensor parallel, distributed_interface, restore-on-boot, speech-to-text, OpenAI client]
---

# SOP-06: Deploy a model with Tensorstead

## Purpose of this procedure

Serve an acquired model: create a deployment definition, start it, confirm it
answers inference, and hand the endpoint to clients.

## Prerequisites for a deployment

- The node is registered and not reserved (SOP-03).
- The model is acquired on every node the deployment will use (SOP-04).
- You know the runtime image (SOP-05). Staging it with `image pull` is
  recommended.
- You have a free port on the node. `endpoint_conflict` means another
  deployment already holds that `host:port`.
- There is enough memory: run `stead node resources <node>`. For vLLM and
  SGLang, `gpu_memory_utilization` / `mem_fraction_static` is a share of the
  node's **total** memory. A start whose share is larger than the memory free
  right now is refused with `insufficient_memory`, so on a node already serving
  another deployment set it to fit what is left. Other runtimes are not
  checked, and nothing stops you overcommitting them.

## Step 1: Create the deployment (no host change)

```sh
stead deployment create --name <name> \
  --model <model id prefix or source name> \
  --runtime vllm --image 'nvcr.io/nvidia/vllm@sha256:<digest>' \
  --node spark-01 --endpoint spark-01.internal:8000 \
  --config tensor_parallel_size=1 --config max_model_len=32768 \
  --config gpu_memory_utilization=0.5 --config served_model_name=<client-facing-name>
```

- This creates revision 1 with desired state `stopped`. Nothing on the host
  changes.
- `--config` takes repeated `key=value` pairs (values are strings) or one JSON
  object. Nested values such as `extra_args` or `host_config` need the JSON
  form: `--config '{"ctx_size": 32768, "extra_args": {"alias": "m"}}'`.
- Names must be unique. Pick a name that says what the deployment serves.
- `--restore-on-boot` is off by default. Turn it on only after the deployment
  is known good (SOP-12).
- `--from-export <file.yaml>` creates from an exported definition instead.

## Runtime configuration reference

Unknown keys are refused (`invalid_runtime_config`). Values are then checked by
the runtime itself at start.

| Runtime | Common keys |
| --- | --- |
| `vllm` | `tensor_parallel_size`, `pipeline_parallel_size`, `distributed_interface`, `distributed_executor_backend`, `max_model_len`, `gpu_memory_utilization`, `dtype`, `quantization`, `served_model_name`, `max_num_seqs`, `max_num_batched_tokens`, `kv_cache_dtype`, `enable_prefix_caching`, `enable_chunked_prefill`, `reasoning_parser`, `tool_call_parser`, `enable_auto_tool_choice`, `speculative_config`, `load_format`, `enforce_eager`, `seed`, `disable_log_requests`, `use_image_entrypoint`, `trust_remote_code`*, `allow_io_uring`*, `extra_args`, `host_config` |
| `sglang` | `tp_size`, `dp_size`, `mem_fraction_static`, `context_length`, `max_running_requests`, `chunked_prefill_size`, `kv_cache_dtype`, `reasoning_parser`, `tool_call_parser`, `speculative_algorithm`, `speculative_num_draft_tokens`, `speculative_draft_model_path`, `enable_metrics`, `trust_remote_code`*, `extra_args`, `host_config` |
| `llamacpp` | `model_file`, `mmproj_file`, `n_gpu_layers`, `ctx_size`, `threads`, `batch_size`, `parallel`, `flash_attn`, `draft_model`, `draft_model_file`, `extra_args` |
| `trtllm` | `tp_size`, `pp_size`, `ep_size`, `max_batch_size`, `max_num_tokens`, `max_seq_len`, `kv_cache_free_gpu_memory_fraction`, `kv_cache_dtype`, `enable_chunked_prefill`, `reasoning_parser`, `tool_parser`, `served_model_name`, `chat_template`, `trust_remote_code`*, `extra_args` |
| `exllama` | `served_model_name` or `model_subdir` (not both), `max_seq_len`, `cache_size`, `cache_mode`, `chunk_size`. EXL3 models only; EXL2 and GPTQ are refused |
| `tensorfold` | `served_model_name`, `context`, `parallel`, `max_tokens`, `thinking`, `reasoning_effort`, `drafts`, `mtp_drafts`, `drafter`, `kv_cache`, `prompt_cache_gib`, `vision`, `precision`, `distributed_interface`, `extra_args` |

\* Requires a code approval (SOP-10).

Notes:

- `extra_args` forwards flags Tensorstead does not model, unvalidated. Flags
  that load code (`--chat-template`, `--tool-parser-plugin`,
  `--reasoning-parser-plugin`, `--logits-processors`, `--worker-cls`,
  `--worker-extension-cls`) are refused in vLLM and SGLang.
- `host_config` (vLLM, SGLang) accepts `shm_size`, `ipc_mode`, `extra_ports`,
  `ulimits` and `environment`. Privileged mode, host networking and arbitrary
  bind mounts are refused.
- Never put a secret in `--config`. Runtime config is exported and displayed as
  given, with no redaction.
- Check parsers against the model's output format. A `hermes` tool parser
  returns no `tool_calls` for a model that emits XML tool calls.

## Multi-node deployments

Only `vllm`, `sglang` and `tensorfold` distribute across nodes. Other runtimes
are refused with `runtime_not_distributed`.

- Repeat `--node` for each node and acquire the model on every one of them.
- Set `distributed_interface` to the fabric interface (for example the
  ConnectX-7 port). It has no default. Without it the runtime may pick the
  general LAN.
- For vLLM, set `tensor_parallel_size` (and optionally
  `distributed_executor_backend`) to match the group.
- TensorFold two-node groups need the identical checkpoint on both nodes. Rank
  1 serves no API.

## Step 2: Start the deployment (host-changing)

```sh
stead deployment start <name>
stead operation show <operation id>
```

The agent pulls the image if needed, starts the container, and writes a systemd
unit (enabled at boot only when both boot-restoration switches are on). The
operation finishing means the container started, not that the model serves.

## Step 3: Wait until the deployment serves (read-only)

```sh
stead deployment show <name>      # declared and observed blocks side by side
stead deployment status <name>    # observed only
```

Wait for `inference_ready: true` on every node. Allow a few minutes for a small
model and about 20 minutes for a large one. If it does not settle, read the
runtime's own output instead of retrying:

```sh
stead deployment runtime <name> --tail 500
```

## Step 4: Hand the endpoint to clients

Clients call the runtime directly, not Tensorstead. For OpenAI-compatible
runtimes the base URL is `http://<endpoint>/v1`, and the model name is the
`served_model_name`.

```sh
curl http://spark-01.internal:8000/v1/models
curl http://spark-01.internal:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "<served name>", "messages": [{"role": "user", "content": "Say hello"}]}'
```

If an inference key applies (SOP-09), add
`-H "Authorization: Bearer $(< <key file>)"`. `deployment show` reports
`endpoint_authenticated: false` when the endpoint is open.

## Special case: speech-to-text

Speech models (Whisper, Voxtral, Parakeet and others) are served by vLLM with
an image that adds the `vllm[audio]` dependencies. Clients call
`POST /v1/audio/transcriptions`. The full procedure is in
`docs/speech-to-text.md`. It has been checked against vLLM's source but not yet
run on hardware.
