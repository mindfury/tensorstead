# Changelog

All notable changes to Tensorstead are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The agent/coordinator wire contract
is versioned separately (`stead status` reports both).

## [Unreleased]

## [1.5.0] - 2026-10-14

### Added
- **Speculative decoding drafters for llama.cpp.** `draft_model` (an acquired
  drafter's store directory) and `draft_model_file` (the .gguf within it) are
  mounted by the agent the way vLLM and SGLang drafters are, and reach
  llama-server as `--model-draft`. The type and length stay in `extra_args`
  (`spec-type`, `spec-draft-n-max`), as llama.cpp's published recipes put them.
  A drafter with no drafting type, or a drafting type (`draft-dflash`,
  `draft-eagle3`, `draft-dspark`, `draft-simple`) with no drafter, is refused
  before a container starts. A drafter passed in `extra_args` (`model-draft`)
  is refused, because the agent would not mount it.

### Documentation
- **llama.cpp on a DGX Spark.** The suggested image is now
  `ghcr.io/ggml-org/llama.cpp:server-cuda`, which runs on a GB10 as published,
  replacing the CPU-only `light` tag under the project's former organisation.
  `docs/builds.md` shows acquiring one GGUF quantization with `--file` and
  deploying it. Verified with Unsloth's Qwen3.8-27B UD-Q4_K_XL.

## [1.4.0] - 2026-10-14

### Added
- **`deployment rename`.** Available on the API (`POST /v1/deployments/{id}:rename`),
  the CLI (`stead deployment rename NAME NEW_NAME`) and MCP (`deployment_rename`).
  It changes the name and nothing else: id, revisions, containers, served model
  name and observed state are untouched, so it is safe while running and needs
  no restart. Names were write-once so that a deployment repointed at another
  model kept its handle. Since 1.3.0 refuses that repointing, names can say what
  a deployment serves. A rename target is 1-128 letters, digits, `.`, `_` or `-`.

### Fixed
- **Creating a deployment under a name another deployment holds is refused**
  with `already_exists` (409). Before, the request reached the store, whose
  `INSERT OR REPLACE` on the unique name asked SQLite to delete the other
  deployment's row. Only that row's revisions' foreign keys prevented it, and
  the request failed as a bare integrity error.

## [1.3.0] - 2026-10-13

### Added
- **Model-swap guardrails.** Changing `model_id` on a running deployment is
  refused (`model_change_refused`, 409): a different model is a new deployment.
  On a stopped deployment it is still accepted, with a warning that the name,
  endpoint and configs carry over. Starting or restarting a deployment whose
  declared served model name another running deployment already serves is
  refused (`served_name_conflict`, 409). A restart is checked before it stops
  anything, and `modify` warns when a rename would collide. The served name is
  read from `served_model_name` or from `extra_args`
  (`served-model-name`, `alias`).
- **Runbooks over MCP.** `runbook_list` and `runbook_get` serve procedures to
  agents. The first is `model-swap` (also `docs/model-swap.md`), and the MCP
  server's instructions point to it. `skills/tensorstead-model-swap/SKILL.md`
  is the short form for agent hosts that load skills.

### Fixed
- **The release smoke test no longer counts `coordinator_version` as an
  unexpected MCP tool.** Tools that are not catalogued operations are declared
  in `NON_CATALOGUE_TOOLS`.

## [1.2.0] - 2026-10-12

### Added
- **ExLlama `served_model_name`.** TabbyAPI names a model after its directory,
  so ExLlama deployments were served as the store's encoded directory token
  (`aHVnZ2luZ2ZhY2U6...`). Setting `served_model_name` mounts the model at
  `/models/<name>` and serves it under that name. Adapters can now choose where
  a model is mounted inside the container (`ContainerRequirements.model_mount_target`);
  the mounted host path is still the one the agent resolved and checked.

### Fixed
- **Resource reports read the configured model store.** The fallback for
  managed paths read `TENSORSTEAD_MODEL_STORE`, but the install sets
  `TENSORSTEAD_MODEL_STORE_PATH`. The fallback now reads the installed name
  first.


## [1.1.0] - 2026-10-12

### Added
- **ExLlama runtime** (`exllama`, single-node) — serves EXL3-quantized and
  unquantized models through TabbyAPI. EXL2 and GPTQ models are refused before
  the container starts, since TabbyAPI no longer loads them. TabbyAPI can only
  be given an API key through a file whose contents it logs on every start, so
  binding an inference credential to an `exllama` deployment refuses the start.
  Without one, the endpoint serves unauthenticated, and that includes TabbyAPI's
  admin endpoints. The suggested upstream image is x86_64 only.
- **TensorRT-LLM runtime** (`trtllm`, single-node) — serves Hugging Face
  checkpoints with `trtllm-serve`. TensorRT-LLM's usage telemetry to NVIDIA is
  off unless `telemetry: true` is set. `trtllm-serve` has no inference key, so
  deployments serve unauthenticated, and `extra_args` options that would load
  code or replace the recorded configuration are refused. Spanning nodes is not
  supported, because TensorRT-LLM forms multi-node groups through MPI.
- **Speech-to-text guide** (`docs/speech-to-text.md`) — a local transcription
  endpoint served by the existing vLLM runtime (Whisper, Voxtral, Qwen3-ASR and
  others), including building an image with vLLM's audio extra.

### Fixed
- **Models acquired on a node with a non-default model store now start.** The
  coordinator records a model's path under `/var/lib/tensorstead/models`
  regardless of the node's `TENSORSTEAD_MODEL_STORE_PATH`, and the agent
  refused any path outside its own store, so on such a node every newly
  acquired model downloaded correctly and then failed to start ("does not
  resolve to a directory beneath the managed store"). This hit every install
  upgraded in place from Project X, which keeps its old store. The agent now
  finds a model it holds under its own store when the recorded root differs;
  containment is unchanged.
- **TensorRT-LLM deployments start.** The adapter replaced the release image's
  entrypoint with a bare `trtllm-serve`, which skips the setup NVIDIA's
  entrypoint performs; on release 1.2.1 `import tensorrt` then failed
  (`libnvonnxparser.so.10`) and every deployment exited at startup. It now runs
  `trtllm-serve` through `/opt/nvidia/nvidia_entrypoint.sh`.
- **TensorRT-LLM 1.2.x (the current stable line) starts.** The adapter always
  passed `--no-telemetry`, which exists only from 1.3, so every 1.2.x start
  failed at argument parsing. Telemetry is now switched off with
  `TRTLLM_NO_USAGE_STATS=1`, which 1.3 honours and 1.2.x (which has no
  telemetry) never reads. `served_model_name` and `kv_cache_dtype` remain 1.3+
  and now raise a config advisory saying so. Verified serving on a DGX Spark
  with 1.2.1 and 1.3.0rc29.

### Documentation
- **Verified arm64 runtime images** (`docs/builds.md`): pinned recipes for
  TensorRT-LLM and for ExLlama/TabbyAPI built on a DGX Spark, including the
  exllamav3 `sm_121` build and TabbyAPI's undeclared `uvloop` dependency on
  arm64.

### Changed
- **Inference keys are optional everywhere.** The Ansible install no longer
  generates a node-wide inference API key unless
  `tensorstead_inference_api_key_enabled: true` is set. Where a node does carry
  one, it now applies only to runtimes that can enforce it (vLLM, llama.cpp).
  SGLang, ExLlama and TensorRT-LLM deployments start unauthenticated instead of
  being refused. A key bound to a specific deployment still refuses the start on
  a runtime that cannot enforce it. **Upgrading:** an existing installation that relies on the
  node key must set `tensorstead_inference_api_key_enabled: true` before
  re-running `site.yml`. Otherwise the key is removed from the agents, and vLLM
  and llama.cpp deployments serve unauthenticated after their next start. The
  key file on the controller is kept and reused.

## [1.0.1] - 2026-09-30

### Fixed
- `pip install "tensorstead[cli]"` produced a `stead` command that failed on
  start: the `cli` extra did not declare `httpx`, the HTTP client the CLI is
  built on, and the CLI imported the coordinator's web server (`uvicorn`) at
  startup even though only `stead coordinator serve` uses it. Installs with all
  extras (the Ansible path) were unaffected. CI now installs the built wheel one extra at a time so a
  missing dependency fails the build.

## [1.0.0] - 2026-09-30

First public release.

### Management
- **Nodes** — register a host whose agent is already running, inspect it, check
  reachability and read accelerator, memory, and storage on demand. Readings
  inform; they never gate an operation.
- **Models** — acquire an immutable revision from Hugging Face onto one or more
  nodes (fetched once upstream, then replicated between nodes), with gated-model
  credentials stored by reference and never echoed.
- **Deployments** — declare model, runtime, image, nodes, endpoint, and
  configuration as a numbered revision; every revision is retained and
  exportable, and an export can recreate the deployment.
- **Lifecycle** — explicit start, stop, restart, reconcile, and remove. Status
  shows declared and observed state side by side, fetched live.
- **Boot restoration** — a started deployment installs a self-sufficient
  systemd unit on each node, so inference returns after a reboot without the
  coordinator.
- **Runtimes** — vLLM and SGLang (single- and multi-node) and llama.cpp
  (single-node), each with its own configuration validation.
- **Runtime images** — record a build spec as data and build it on a node, or
  import a prebuilt archive, to repair a broken upstream image without SSH.
- **Inference credentials** — Bearer-token protection for runtime endpoints,
  kept out of the database and every export.

### Interfaces
- `stead` CLI with guided `login`, a `status` command that names the source of
  every setting, and operator-friendly tables.
- MCP server (`tensorstead-mcp`) over stdio and as a TLS-protected Streamable
  HTTP service, exposing exactly the management operation set — enforced by a
  parity test — and no shell, exec, or path-taking tool.
- Coordinator HTTP API with TLS and token authentication; open "dev mode" is
  refused on any non-loopback address.

### Installation
- Ansible playbooks to install the coordinator, agents, and hosted MCP on
  already-provisioned Linux hosts, with an automatically managed private CA,
  a read-only readiness check, service control, orderly reboot, and optional
  DGX Spark node preparation.
- `make build` produces a wheel with a build number and SHA-256 provenance
  manifest.

[1.5.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.5.0
[1.4.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.4.0
[1.3.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.3.0
[1.2.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.2.0
[1.1.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.1.0
[1.0.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.1
[1.0.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.0
