# Changelog

All notable changes to Tensorstead are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The agent/coordinator wire contract
is versioned separately (`stead status` reports both).

## [1.12.0] - 2026-10-24

### Changed
- **Node-to-node model copies are several times faster.** A copy ran at about
  4 Gbps on a 200 Gb fabric. It made four passes over the model, one after
  another, over a single TLS stream: the source wrote a whole tar before
  sending a byte, then the destination wrote the tar, unpacked it, and re-read
  everything to hash it. Each pass ran at roughly 4-5 Gbps. Now the
  destination reads the source's manifest and pulls the files as byte ranges
  over several connections, writing each range straight into place. It hashes
  each file as its pieces land, so verification finishes moments after the
  last byte. On a 4 GiB loopback benchmark, end to end went from 25.9 s to
  4.5 s.
  - The digest, the verification and the staging-then-promote rules are
    unchanged. The manifest is checked as untrusted input before anything is
    written, and free space is now checked against the model's real size.
  - A peer that predates this is still copied over the old tar protocol, so
    agents can be upgraded one at a time.
  - `TENSORSTEAD_REPLICATION_CONNECTIONS` (default 8) and
    `TENSORSTEAD_REPLICATION_HASHERS` (default 4) tune the parallelism.

### Added
- Two agent-to-agent routes under the model content endpoint, gated by the
  replication token like it: `.../content/manifest` and
  `.../content/file?path=` (honours a byte `Range`).

### Security
- Dependency updates (#58, #57): `fastapi` 0.156.4, `ruff` 0.17.1, and `astral-sh/setup-uv` 10.3.1.

## [1.11.1] - 2026-10-23

### Fixed
- **`start` and `stop` no longer report the record as if it were the node.**
  "Already running" and "already stopped" used to be read off the declared
  state. After a reboot, crash or OOM, `start` refused with "is already
  running" while the coordinator's own observation said `not_running`. Both
  now ask the nodes first:
  - **A deployment declared running whose containers are gone** is started
    again; the dead container is cleared first.
  - **A deployment recorded stopped that is still running** is stopped.
  - **A node that cannot be observed** refuses the start with
    `agent_unreachable` and makes no claim either way.
- **A model download that outlasts the one-hour hop is no longer recorded as
  failed while it is still running.** The coordinator now asks the agent
  again, up to 5 more hours. The repeat waits on the agent's per-model lock
  for the transfer already in flight and returns its real outcome; it cannot
  start a second download. Real failures are still reported at once.
- **A failed attempt's placeholder record is removed when the same acquire
  later succeeds.** It was filed under the unresolved revision (`2`, `main`)
  and used to stay beside the real model, reading "failed, unpinned, 0 bytes".
  It is removed only if it was never pinned, holds only failed replicas, and
  no deployment names it.

### Security
- Dependency updates (#54): `opentelemetry-api` 1.45.1, `filelock` 4.2.1, `black` 26.10.1.

## [1.11.0] - 2026-10-23

### Added
- **Azure ML registry model source** (`source_id: "azureml"`). Some models are
  published only to the Azure ML public registry, such as Microsoft's
  MAI-Code-1.1-Flash-On-Device. Fetching one used to mean a hand download
  outside Tensorstead.
  - **Model ids** are `<registry>/<name>`, and an explicit version is required.
  - **The resolved revision** is `<version>@<manifest SHA-512>`, so files that
    change under the same version are refused, not silently fetched.
  - **Downloads** use the registry's short-lived read-only link. Each file is
    checked against its published MD5 and size, `file_selector` is honoured,
    and the link's signature is never logged.
  - **Agents** now route acquires by `source_id`; an unknown source is refused
    (`unknown_model_source`).

### Fixed
- **Copying a large model between nodes no longer reports failure while it is
  still copying.** The coordinator waited only the default 30 s for
  `models:replicate`. Copying a 95 GB checkpoint was marked failed while both
  agents carried on, and the failure told the operator to check `image_list`.
  Replicate now gets the same one-hour bound as acquire and digest, and a
  timed-out model call points at `model_list`.
- **Re-running an acquire after such a timeout is free.** A destination that
  already holds an available replica with the expected digest answers at once.
  It no longer wipes the finished copy and transfers it again.
- **A copied replica keeps its revision.** The coordinator sent
  `resolved_revision` with every replicate call, but the agent dropped it, so a
  replica that arrived by copy had no revision. Anything bound to a revision,
  such as an `allow_io_uring` approval, refused to start on it. The revision is
  now recorded when the copy is promoted, and a retried acquire fills it in on
  a replica copied earlier, without moving a byte.

## [1.10.0] - 2026-10-22

### Added
- **`stead model digest` / `POST /v1/models/{id}:digest` / MCP
  `model_digest`.**
  - **What it does:** computes and records the content digest and size of a
    replica the estate already holds.
  - **Why:** models acquired before downloads were digested could not be
    copied between nodes. Replication refused, correctly, with nothing to
    verify against, so placing one on another node meant downloading it from
    upstream again. Found placing Nemotron Bolt on a second node.
  - **After digesting:** the acquire copies node to node, verified.
  - **A model that already has a digest** is verified, never overwritten.
    A difference is reported as `digest_mismatch`.
  - **Model records also gain `size_bytes`**, which every pre-digest model
    lacked.

### Fixed
- **llama.cpp deployments bind the port Tensorstead publishes.** `llama-server`
  changed its default port from 8080 to 9931 on 2026-10-09 (ggml-org/llama.cpp
  #30159). With any build since then, the server loaded and listened on 9931
  while Tensorstead published 8080, so the deployment never became ready. Found
  deploying Recursive Flint on llama.cpp b11804. The adapter now passes
  `--port 8080` explicitly; older builds are unaffected.

### Dependencies
- `certifi` 2026.7.22 → 2026.10.19 (#47, security group): the CA bundle used for
  TLS verification.

## [1.9.0] - 2026-10-20

### Added
- **TensorFold runtime** (`runtime_type: tensorfold`). TensorFold is
  [ashhart/TensorFold](https://github.com/ashhart/TensorFold)'s native serving
  binary, with exact draft-and-verify decoding.
  - **Launch:** the adapter runs `tensorfold-native serve` on an acquired
    checkpoint and delivers the inference key as `TENSORFOLD_API_KEY`, never
    argv.
  - **Config:** the served name, context, parallelism, thinking and reasoning
    effort, drafts, MTP depth, a mounted drafter, KV-cache dtype, prompt cache,
    vision and precision.
  - **Refused in `extra_args`:**
    - the product-owned keys;
    - `--speed-up`, which loads a native library the record does not describe;
    - `--cortheon`, which makes the server issue outbound web requests.
  - **Two nodes:** Nemotron Super VL and GLM-5.3-Flash run as a two-node
    group, with ranks over NCCL/RDMA on the declared `distributed_interface`;
    rank 1 serves no API.
  - **Schema probe:** reads the binary's `capabilities --json`.
  - **Image:** a build-spec recipe for the image is in `docs/builds.md`.

  Measured on the estate's GB10s against the vLLM deployments of the same
  checkpoints:
  - Nemotron Lightning NVFP4: 140 vs 103 tok/s single-stream decode, loaded in
    14 s vs 176 s.
  - Nemotron Super VL NVFP4: 50 vs 48 tok/s on one node, 61 tok/s on two.
  - Nemotron Bolt: 39 vs 81 tok/s. TensorFold has no drafter for it.

### Changed
- `SchemaProbe.script` may be `None`, for a runtime that reports its own
  surface without a probe script.

### Fixed
- **Adding a node to a model already on the estate copies it instead of
  downloading it again.** `acquire` looked for an existing replica only among
  the nodes named in the request, so `nodes=[new node]` found no source and
  fetched the whole checkpoint from upstream while another node held a verified
  copy. Measured on the estate: ~70 GB of Nemotron Super VL downloaded to
  spark-6d42 beside spark-471b's replica. A verified replica on any node now
  serves as the source, and when that copy cannot be made the acquire is
  refused (`replication_failed`) instead of downloading again, as for a holder
  named in the request.

## [1.8.2] - 2026-10-19

### Dependencies
- **`huggingface-hub` 1.27.0 → 2.2.0** (#32), the library behind model
  downloads. Validated against the live hub before release: Tensorstead's real
  Hugging Face source gave identical results under both versions for a whole
  tiny-model download with verification, a filtered (`allow_patterns`)
  download, and gated and anonymous requests mapping to
  `authorization_refused`. It now uses `httpx2` instead of `httpx`; nothing new
  enters the dependency set.

## [1.8.1] - 2026-10-19

### Fixed
- **Peer model replication preserves the complete digested tree.** Empty
  directories were omitted from the archive, causing digest mismatches, and
  hardlinked regular files became tar links that the receiver refused. The
  archive now includes each directory and file once and writes hardlinked
  files as independent regular entries. Symlinks and special files are refused
  before serving, matching the digest and extraction rules.

### Dependencies
- `types-pyyaml` 6.0.12.20260724 → 6.0.12.20260906 (type stubs, development
  only).
- Build backend range widened to `uv_build>=0.8,<0.14`.

## [1.8.0] - 2026-10-18

### Changed
- **`deployment_list` (MCP) returns a short summary per deployment and takes
  filters.** It returned every deployment's full declared/observed document:
  ~70 KB for 42 deployments from the API and ~190 KB once MCP carried it as
  both text and structured content. An agent's harness spilled that to a file,
  so "which deployment serves model X?" could not be answered with the tools
  alone. It now returns `id`, `name`, `desired_state`, revisions,
  `source_model_id`, `served_model_name`, `runtime_type`, `endpoint`,
  `participating_nodes` and `observed_status`, about 770 bytes each, and takes
  `name`, `model`, `desired_state`, `endpoint` and `observe`. The full record is
  still `deployment_get`. Its description no longer says that deployments
  cannot be renamed.
- **`GET /v1/deployments` takes the same filters, plus `view=summary`.** The
  default stays the full document, so existing callers are unchanged. Filters
  apply before any node is asked, so a filtered listing observes only the
  deployments it returns. An unfiltered full listing observes all of them
  (~3 s for 42). Summaries observe nothing unless `observe=true`.
- **CLI lookups by name ask only for that deployment.** Commands that resolve a
  deployment by name used to fetch and observe every deployment.

## [1.7.0] - 2026-10-17

### Security
Fixes from an independent Project CodeGuard review (IDs CG-xx refer to it).
- **CG-01: a bound inference credential reaches every start.** An initial
  `deployment start` resolved no credential, so the agent fell back to the
  node's default key or none; only `restart` honoured the binding. Both now go
  through one path that resolves the binding *before* any image or node call.
  A missing, unreadable or empty binding refuses the start
  (`credential_resolution_failed`) with every node untouched.
- **CG-02: legacy model-path migration can't move anything from outside the
  store.** The agent renamed a colon-bearing legacy path after a lexical check,
  so `<store>/../outside:x` was moved into the store and then accepted.
  Migration now requires a replica this agent recorded under that id. It walks
  the path with `O_NOFOLLOW` directory handles and renames relative to them,
  so traversal, symlinked parents and symlinked final components are never
  followed or moved.
- **CG-03: credential file imports are confined to an import directory.**
  `--from-file` read any coordinator-readable file, so an authenticated caller
  could import Tensorstead's own operational secrets as a credential. Now only
  files inside `TENSORSTEAD_CREDENTIAL_IMPORT_DIR` are read, opened
  descriptor-relative with no symlinks anywhere on the path. The directory must
  not overlap the credential store, the coordinator state directory or
  `/etc/tensorstead`, and `--from-file` is refused when it is unset. The
  coordinator role creates `/var/lib/tensorstead/credential-imports`.
- **CG-04: image archives are authorized as a whole before loading.**
  `import_image` loaded an archive and checked only the first image, so extra
  images and tags (from a compromised peer, say) survived verification. The
  archive's `index.json`/`manifest.json` must now declare exactly one image,
  identified as expected, carrying no name the caller didn't allow, or nothing
  is loaded. A load that differs from its declaration removes only what it
  introduced.
- **CG-05: the model-tree digest is framed and versioned.** v1 hashed
  `path NUL bytes` with no length framing, so `{a: "x", b: "y"}` and
  `{a: "xb\0y"}` hashed identically, letting a peer substitute a differently
  shaped tree. v2 (`tsdir2-sha256:`) hashes a canonical manifest of type, framed
  path, size and per-file SHA-256, counts empty directories, and refuses links
  and special files. Replication refuses to verify against a v1 digest rather
  than reinterpreting it.
- **CG-06: replication extraction is budgeted.** The transfer ceiling bounded
  wire bytes only, so a sparse member could expand far past it. Extraction is
  now one streaming pass that, before writing each entry, refuses sparse
  entries, links and special files, and enforces the member count, the
  cumulative logical size (the same ceiling), the free-space reserve and a
  deadline. Hashing is bounded as a result.
- **CG-08: definition changes can't be lost to a concurrent lifecycle
  operation.** `modify` didn't share the lifecycle lock, and lifecycle saves
  rewrote the whole deployment row from a stale snapshot, so a modify accepted
  mid-start was rolled back and its revision orphaned. Concurrent modifies
  could also hit a raw uniqueness error. Lifecycle, modify and rename now share
  one per-deployment lock. Lifecycle and rename write only their own columns.
  Modify inserts the revision and advances the pointer in one `BEGIN IMMEDIATE`
  compare-and-swap, answering a lost race with `concurrent_modification`.
- **CG-09: stop, restart and remove reach every node the deployment ran on.**
  After a placement change while running, stop drove only the new nodes,
  recorded `stopped`, and left the old node serving. Stop and restart now
  target the running revision's nodes plus the current revision's, and remove
  targets every retained revision's nodes. An unreachable old node fails the
  stop instead of reporting success.
- **CG-10: an image a retained revision ran can't be deleted.** The guard
  compared revision digests the service never populated, so nothing was ever
  protected. Each start now records, per revision and node, the digest the
  agent ran (`revision_images`, migration 0010). The guard checks those
  bindings and, for revisions without evidence, protects any image whose
  reference they name.
- **CG-07: build, probe and runtime output is bounded while captured.**
  buildx output was captured whole, a classic build's error log accumulated in
  full, and `logs(tail=N)` returned one unbounded string for a newline-free
  record. All three now stream into a bounded tail (1 MiB, with long lines
  cut), and buildx is killed at its deadline. Containers Tensorstead creates
  get a rotated `json-file` log driver (3 × 50 MB).
- **CG-11: every route body is a strict typed model.** The credential, binding,
  approval, buildspec, image-build, image-import and create-from-export routes
  took a raw `dict`, ignoring unknown fields and turning wrong types into 500s.
  They now reject unknown fields, wrong types, conflicting sources and
  oversized lists with 422 and write nothing. The export document is validated
  section by section. This also fixes create-from-export ignoring the
  `endpoint` override the CLI sent; MCP's `deployment_create_from_export` now
  takes `endpoint` too.
- **CG-12: errors never echo what the client sent.** Validation errors keep
  location, rule and type, but never the rejected input. An unexpected field's
  name is shown as `<unexpected field>`. Credential-source errors are a typed
  422 (`invalid_credential_source`) that withholds the env-var name or file
  path given, so a secret mis-pasted there doesn't bounce back. Unexpected
  500s return the exception type and an `error_id`; the message and traceback
  go to the log under that id. The CLI now renders 422s instead of reporting
  "not this product".
- **CG-15: admission budgets.** Request body size and JSON depth are limited
  on every surface; background operations take one of a fixed number of slots
  (429 `busy` when full); and connections are capped per server. See
  docs/operations.md, "Admission budgets".
- **CG-18: no unauthenticated API docs.** `/docs`, `/redoc` and
  `/openapi.json` are disabled on coordinator and agent. The schema is served
  at `GET /v1/openapi.json` behind the management token.
- **CG-19: malformed Authorization headers are a plain 401.** Comparing
  non-ASCII `str` tokens raised `TypeError`. All three servers now compare
  bytes, so this can't happen, and refuse duplicate Authorization headers.

- **CG-13: the server-side push gate checks every ref on its own.** The
  `pre-receive` hook gathered changed files from all pushed refs but then
  checked them against only the *last* ref's tree, so a violation in one ref
  could pass on another ref's tree. It also split filenames on whitespace, and
  it treated a failed `git diff` as "no Python changed". Each ref is now diffed
  and checked against its own tree with NUL-delimited names, under a per-ref
  time budget. Any diff, archive, ruff or timeout failure refuses the push. The
  expected ruff version is read from the pushed `uv.lock` instead of a
  hard-coded pin that had drifted (0.16.2 against a 0.17.0 lock).
  `make hooks-install-server` now installs the locked version even when an older
  ruff is already present.
- **CG-14: appliances install the reviewed dependency set, offline.** Deploys
  let pip resolve the wheel's lower-bounded dependencies fresh against an index
  every time. `make build` now also produces
  `dist/tensorstead-<version>-wheelhouse/`: the `uv.lock` pins for the target
  platform, each with its hashes, plus the wheels themselves, each re-checked
  against the lock. Ansible installs from it with
  `--no-index --require-hashes --only-binary=:all: --no-deps`, then installs
  Tensorstead with `--no-deps`. A missing or altered wheel fails the deploy, and
  a wheelhouse built for another platform is refused. The new
  `tensorstead_package_wheelhouse` variable is required (`make deploy` passes
  it). The upgrade procedure is in `docs/builds.md`.
- **CG-16: release publishing is separated from building.** The release
  workflow ran checkout, dependency install, tests and the build with
  `contents: write` and a persisted token. Building now runs under
  `contents: read` with `persist-credentials: false`. A separate `publish` job,
  the only one with write access, runs in the `release` environment and only
  attests and uploads the build job's artifacts.
- **CG-17: releases carry signed provenance, verified before install.**
  Release assets (wheel, sdist, build manifest, wheelhouse) and a `SHA256SUMS`
  are attested with `actions/attest-build-provenance` (Sigstore).
  `make deploy-release VERSION=x.y.z` downloads a release and installs nothing
  unless every asset verifies:
  - signed by this repository's `release.yml`;
  - built from the matching tag;
  - matching the attested `SHA256SUMS`;
  - with the wheelhouse matching its manifest.

  `--bundle-dir` verifies offline. Releases before this one carry no
  attestation and are refused by this path.

### Documentation
- **ADR 0001** (`docs/adr/0001-trust-boundaries.md`) records three positions the
  review asked to make explicit:
  - code approvals are operator attestations, not an independent human
    authority;
  - node registration trusts the destination the administrator names;
  - the audit trail records operations, not per-person identity.

  The MCP approval tool and the README no longer imply more.

## [1.6.0] - 2026-10-17

### Added
- **`allow_io_uring` for vLLM deployments: an approval-gated seccomp exception.**
  NVIDIA's dual-DGX-Spark recipe for `nvidia/GLM-5.3-Flash-FP4-Local` loads its
  checkpoint with `--load-format b12x`, which reads through io_uring, and gets
  io_uring by running the container `--privileged`. Docker's default seccomp
  profile has denied the io_uring syscalls since Docker 25, and Tensorstead
  refuses privileged containers, so the published launch died loading its
  weights (`io_uring bounce initialization failed: Operation not permitted`).
  With `allow_io_uring: true`, the container gets Docker's default profile
  (shipped from Docker 29.9.1) plus `io_uring_setup`, `io_uring_enter` and
  `io_uring_register`, and nothing else. It is never `privileged` or
  `seccomp=unconfined`, and the flag never reaches vLLM's argv.
- A new approvable effect class, `relaxes-isolation`, holding that one option.
  Like `trust_remote_code`, it is refused unless a reviewed approval
  (`code_approval_create`) names the exact runtime, model revision and image
  digest, and both the coordinator and the agent check it. It can't arrive
  through `extra_args`. Authorization policy is now `2026-10-16.1`.
- **`stead image pull REF --node A --node B`** (`POST /v1/images:pull`, MCP
  `image_pull`) stages a registry image on nodes before a planned start, so
  the download isn't part of the downtime. It pulls once and copies to the
  rest (see Fixed). Recorded as an `image_acquire` operation.

### Changed
- **Failed replication is refused, not papered over.** When a node already
  holds the model and copying it to another node fails, the acquire fails with
  `replication_failed`, naming both nodes and the reason. It no longer falls
  back to a second upstream download. `allow_upstream_fallback`
  (`--allow-upstream-fallback`, MCP `model_acquire`) opts back in.
- **One download per repository at a time.** An acquire for a repository that
  is already being acquired is refused at once with `acquisition_in_progress`
  (HTTP 409), naming the running operation and its nodes. Before, it waited
  behind the first on a lock, holding the request open. The check keys on
  source, model and file selection but not on revision, so `main` and the
  commit it names can't download side by side.
- Approval refusals name the option's own reason instead of always saying it
  "loads code".

### Fixed
- **A model is downloaded from the Internet once, not once per node.**
  Multi-node acquires were meant to fetch on the first node and replicate to
  the rest, but replication needs a content digest, and the real Hugging Face
  adapter never reported one (only the test fake did). So every additional node
  silently downloaded the whole model again: 165 GiB twice for GLM-5.3. The
  agent now computes the digest itself after verifying a download.
- **A multi-node start pulls its image from the registry once.** Each rank's
  agent used to resolve the image alone, so every node without it pulled the
  same bytes (9.7 GB per node for the GLM-5.3 runtime). The coordinator now
  stages the image first. A node that already holds it is the source;
  otherwise the first node pulls it, and the rest copy it agent-to-agent,
  verified by id. A failed copy fails the start (`image_staging_failed`)
  naming the node, rather than letting it pull. Unreachable nodes are left to
  the start, which reports them as before.
- **A peer-copied, digest-pinned image is usable.** `docker save`/`load` drops
  the `name@sha256:` reference, so a copied image is held only by its id
  (measured on the Sparks). The agent now resolves such a reference to a local
  image whose id is exactly the pinned digest. That match is exact on Docker's
  containerd store, where the id is the manifest digest, and never occurs on
  the classic store. Container create, image remove, the peer content
  endpoint and the schema probe all hand the daemon what the image is held as
  (`agent/local_images.py`), so a copied image starts, is removed rather than
  orphaned, and can be the source for a further copy.

### Security
- GitHub Actions in `ci.yml` and `release.yml` are pinned by commit SHA.
  `release.yml` publishes the package, so a moved tag would have run
  unreviewed code there.
- `.github/dependabot.yml`: weekly grouped version updates for Actions and
  `uv`, and grouped security updates. `SECURITY.md` points reporters at
  private vulnerability reporting.
- Dependency updates, among them httpx2 2.12, httpcore2 2.10, mcp 2.3,
  pydantic 2.14, fastapi 0.143, urllib3 2.8 and pyjwt 2.15. Tensorstead does
  not use OpenTelemetry, so fastapi's change making its exporter setup opt-in
  has no effect here.

## [1.5.1] - 2026-10-15

### Security
- **`extra_args` can no longer load code into vLLM or SGLang.** The passthrough
  now refuses every option the authorization policy classifies as
  `loads-code` (`--tool-parser-plugin`, `--reasoning-parser-plugin`,
  `--chat-template`, `--logits-processors`, `--worker-cls`,
  `--worker-extension-cls`), in any spelling. Code approvals only ever covered
  modelled fields, and the design assumed the passthrough refused these; it
  did not, so a deployment could load code that no approval had reviewed.
  NVIDIA's Nemotron 3.5 Bolt recipe passes `--reasoning-parser-plugin` this
  way. Its plugin also ignores the server's `default-chat-template-kwargs`
  (with thinking off by default, every reply landed in `reasoning`), and the
  built-in `nemotron_v3` parser serves Bolt correctly instead.
- `reasoning_parser_plugin` is classified as `loads-code`. Authorization
  policy is now `2026-10-15.1`. Existing approvals are unaffected, because the
  policy version is not part of an approval's fingerprint.

### Documentation
- Starting a deployment does not make it come back after a reboot. Since
  contract 1.13 the agent enables boot restoration only when the node permits
  it, the deployment asks for it, and the container came up.
  `docs/first-deployment.md`, `docs/operations.md` and the agent route's
  docstring still described the old unconditional behaviour.

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

[1.12.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.12.0
[1.11.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.11.1
[1.11.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.11.0
[1.10.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.10.0
[1.9.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.9.0
[1.8.2]: https://github.com/mindfury/tensorstead/releases/tag/v1.8.2
[1.8.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.8.1
[1.8.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.8.0
[1.7.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.7.0
[1.6.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.6.0
[1.5.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.5.1
[1.5.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.5.0
[1.4.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.4.0
[1.3.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.3.0
[1.2.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.2.0
[1.1.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.1.0
[1.0.1]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.1
[1.0.0]: https://github.com/mindfury/tensorstead/releases/tag/v1.0.0
