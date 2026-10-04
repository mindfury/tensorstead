# Changelog

All notable changes to Tensorstead are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The agent/coordinator wire contract
is versioned separately (`stead status` reports both).

## [Unreleased]

### Added
- **ExLlama runtime** (`exllama`, single-node) — serves EXL3-quantized and
  unquantized models through TabbyAPI. EXL2 and GPTQ models are refused before
  the container starts, since TabbyAPI no longer loads them. TabbyAPI can only
  be given an API key through a file whose contents it logs on every start, so
  binding an inference credential to an `exllama` deployment refuses the start.
  Without one, the endpoint serves unauthenticated, and that includes TabbyAPI's
  admin endpoints. The suggested upstream image is x86_64 only.

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

[1.0.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.1
[1.0.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.0
