# Builds and versions

Tensorstead has two identifiers for every deployable artifact:

- **Release version** (the value in `pyproject.toml`) identifies the software release. It is
  recorded in `pyproject.toml`, reported by the coordinator and agent, and is
  part of the wheel filename.
- **Build number** identifies one exact rebuild of that release. It is local to
  the build controller and is paired with the wheel's SHA-256 checksum.

Use the project build command rather than calling `uv build` directly:

```sh
make build
```

Each run builds the wheel and appends one JSON record to the controller-local,
Git-ignored ledger `.tensorstead/build-history.jsonl`. It also writes a matching
manifest beside the wheel in `dist/`. The record contains the build number,
UTC time, release version, wheel name, SHA-256 checksum, Git revision, and
whether the source tree had uncommitted changes.

Before deploying with Ansible, record the wheel filename, build number, and
SHA-256 from the command output in the deployment change record. The manifest
is safe to share: it contains no tokens, private keys, or configuration
secrets.

The build number is evidence, not a substitute for a release version. Bump the
release version before a deliberate software release; keep the build manifest
to distinguish repeated development builds of the same release. The underlying
command is `uv run python scripts/build_package.py`; use it only if `make` is
unavailable.

Each build also leaves an **independent, never-overwritten copy** of its wheel,
sdist, and manifest in `dist/builds/<build_number>/`. The canonical wheel in
`dist/` is refreshed to the latest build (the version-pinned deploy pointer),
but a prior build's wheel survives in its own directory — the build number now
also indexes a recoverable artifact, not only a ledger row. The directory is
never overwritten: a reused build number fails loudly rather than destroy
history.


## Rollback

If a deploy goes wrong, roll back to a prior build's wheel without rebuilding
from source. List the archived builds and the revision each was built from:

```sh
make builds
```

Then deploy that build by pointing the artifact at its archived wheel:

```sh
make deploy TENSORSTEAD_DEPLOY_ARTIFACT=dist/builds/3/tensorstead-1.12.0-py3-none-any.whl
```

### Dependency upgrades (requalification)

Appliances install exactly the dependency set in `uv.lock`: offline, every
wheel hash-checked, nothing resolved, no source builds. A dependency therefore
changes on an appliance only through this procedure:

1. Change the dependency (`uv lock --upgrade-package <name>`, or a Dependabot
   PR). Review the lock diff.
2. `make check` must pass against the new lock. CI runs the same check with
   `uv sync --locked`.
3. `make build` regenerates the wheelhouse from the new lock. It refuses any
   pin without a hash, any downloaded file whose hash is not in the lock, and
   any pin with no wheel for the target (`linux-aarch64-cp312`). A dependency
   that publishes no aarch64 wheel fails here, not on an appliance.
4. Deploy the build. Reinstalling the same build cannot move a dependency,
   because pip is given no index to resolve against.

A new appliance platform needs a new entry in `TARGETS` in
`scripts/build_wheelhouse.py`. The deploy refuses a wheelhouse whose target does
not match the host venv.

A deploy also needs the version's dependency wheelhouse
(`dist/tensorstead-<version>-wheelhouse/`, built by `make build`); a rollback to
a build of the same version reuses it. Rolling back across a `uv.lock` change
means pointing `TENSORSTEAD_DEPLOY_WHEELHOUSE` at that build's wheelhouse too.

The default `make deploy` path is unchanged — it is still the latest build of
the checked-in version — so this override is the only thing that changes. The
archived wheel is a real, installable wheel (the same one `uv build` produced),
so the deploy playbook needs nothing new; it takes a file path either way.


## Runtime images

The versions above are **Tensorstead's own** releases. A deployment's *runtime*
image is separate, and the product can now produce one — which matters when an
official image is broken.

```sh
stead buildspec set vllm-xgrammar \
  --base 'nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268' \
  --step 'python3 -m pip install --no-deps --no-cache-dir --root-user-action=ignore xgrammar==0.2.1 apache-tvm-ffi==0.1.9' \
  --step 'python3 -c '"'"'from xgrammar import normalize_tool_choice; from importlib.metadata import version; assert version("xgrammar") == "0.2.1"; assert version("apache-tvm-ffi") == "0.1.9"'"'"''

stead buildspec list
stead image build vllm-xgrammar --node spark-alpha.internal \
  --reference local/vllm:26.07-xgrammar-0.2.1
```

Three things about that recipe are deliberate, and each was learned the hard
way:

- **Exact versions, not a range.** `xgrammar>=0.2.1,<1.0.0` is what vLLM
  declares, but as a *build step* it resolves to whatever is newest that day,
  so the same recorded spec produces different images over time. That is the
  ad hoc behaviour the build spec exists to remove.
- **`apache-tvm-ffi` is named explicitly.** `xgrammar` 0.2.x loads its native
  binding through `tvm_ffi`, which 0.1.x did not use. With `--no-deps`, pip
  will not supply it. It happens to be present in this base image because vLLM
  0.24.0 pins `apache-tvm-ffi==0.1.9` itself — so the shorter step would have
  worked here by luck. Naming it makes the requirement true by construction
  rather than by coincidence.
- **`--no-deps` stays.** Without it, pip is free to resolve `torch`,
  `transformers`, and `numpy`, which in a runtime image is how a working
  deployment becomes a broken one.

The second step is the build verifying itself. `docker build` fails on a
non-zero step, so an image that does not import the symbol is never produced
and never tagged. Prefer an assertion that fails loudly over one that reports:
an earlier version of this step printed `xgrammar.__version__`, which the
package does not define, and the build failed for a reason that had nothing to
do with the repair.

Pin `--base` with `@sha256:...`. An unpinned base is reported as
unreproducible and never refused — pinning is your judgement, not the
product's.

To bring an image the estate cannot pull, place the archive in the node's
managed image store and name it:

```sh
stead image import vllm-vendor.tar --node spark-alpha.internal \
  --reference local/vllm:vendor --expect sha256:...
```

`import` takes a **file name inside the managed store**, never a path. `--expect`
makes a mismatched archive import nothing.

### What a built image's identifier is, and is not

A build returns an `image_id`: a content-addressable identifier. It is **not** a
registry digest — a locally produced image has none. The product records which
it is (`origin`), and never presents one as the other, because that would make
an export look portable when it is not.

Why this exists: when `nvcr.io/nvidia/vllm:26.07-py3` shipped an `xgrammar`
too old for its own tool-calling path, repairing it meant SSH — outside the
product, and unrecorded. That is the workflow Tensorstead exists to replace.

### Worked example: the xgrammar repair, as performed

Run against both nodes. The whole sequence, in order:

```sh
# 1. Record the spec (executes nothing), then build on each node that runs it.
stead buildspec set vllm-xgrammar --base '...' --step '...' --step '...'
stead image build vllm-xgrammar --node spark-alpha.internal \
  --reference local/vllm:26.07-xgrammar-0.2.1
stead image build vllm-xgrammar --node spark-beta.internal \
  --reference local/vllm:26.07-xgrammar-0.2.1

# 2. Point the deployment at it. Pass the WHOLE runtime config — see below.
stead deployment modify qwen36-27b --expect-revision 7 \
  --image local/vllm:26.07-xgrammar-0.2.1 \
  --config tensor_parallel_size=1 --config max_model_len=262144 \
  --config gpu_memory_utilization=0.4 --config quantization=modelopt \
  --config reasoning_parser=qwen3 --config tool_call_parser=qwen3_xml \
  --config enable_auto_tool_choice=true --config served_model_name=qwen36-27b

# 3. Modify never restarts anything. Do it explicitly.
stead deployment restart qwen36-27b
```

Three things that cost time, so they are written down:

- **`--config` replaces the entire runtime config, it does not merge.** One
  `--config tool_call_parser=...` produced a revision that had lost
  `gpu_memory_utilization`, `max_model_len`, and four other settings. Always
  pass the full set, and read back the declared revision before restarting.
  `deployment modify` changes no host state, so there is a free window in which
  to check.
- **A new image may not be the whole repair.** With xgrammar fixed, tool calls
  returned 200 but produced no parsed `tool_calls`: the model emits Qwen's XML
  form and the deployments were set to `tool_call_parser: hermes`, which
  expects JSON. The parser had never been correct; the xgrammar fault simply
  failed earlier. Check that a fix reaches the *outcome* you wanted, not just
  that the previous error stopped appearing.
- **Build on every node that runs the deployment.** The produce-once-and-
  distribute is not implemented, so the same spec built on two nodes yields two
  different image identifiers. Fine for a repair; not fine if you are relying
  on identifiers matching across nodes.

### Runtime images for a DGX Spark (arm64, GB10)

Recipes verified on a DGX Spark in October 2026. Pin by digest as shown; a
newer tag is a new recipe, not an edit to this one.

**TensorRT-LLM.** NVIDIA's release image runs as published. Both the current
stable line (1.2.x) and 1.3 serve. The adapter's `served_model_name` and
`kv_cache_dtype` need 1.3 or later, and `config_advisories` says so when either
is set.

```sh
stead buildspec set trtllm-1.2.1-arm64 \
  --base 'nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9'
stead image build trtllm-1.2.1-arm64 --node spark-alpha.internal \
  --reference local/trtllm:1.2.1
```

**llama.cpp (GGUF).** The project's `server-cuda` image is published for
arm64 and runs on a GB10 as published: pull it, pin it by digest, and use it as
the deployment's image directly. No build spec is needed. A source build for
`sm_121` served the same GGUF no faster (12.1 vs 11.6 tokens/s, Qwen3.8-27B
UD-Q4_K_XL). Acquire one quantization from a GGUF repository with `--file`,
since a repository ships twenty or more:

```sh
stead model acquire --source huggingface --id unsloth/Qwen3.8-27B-GGUF \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --file 'Qwen3.8-27B-UD-Q4_K_XL.gguf' --node spark-alpha.internal
stead deployment create --name qwen3.8-27b-ud-q4_k_xl-llamacpp \
  --model <model id> --runtime llamacpp --node spark-alpha.internal \
  --image 'ghcr.io/ggml-org/llama.cpp@sha256:b0baccbe2e00b5d74e63f85f9f023c93d61d0f329475269e1e7002e87eb875ea' \
  --endpoint spark-alpha.internal:8133 \
  --config '{"ctx_size": 32768, "flash_attn": true, "extra_args": {"alias": "qwen3.8-27b-ud-q4kxl"}}'
```

A nested setting such as `extra_args` needs the single-JSON form of `--config`;
`key=value` pairs carry strings. llama.cpp enforces a node inference key when
the node has one, so clients send `Authorization: Bearer <key>`.

**llama.cpp with a speculative-decoding drafter.** A single stream on a GB10
reads every weight once per token, so a 17.6 GB GGUF tops out near 15 tokens/s
on its own. A DFlash2 drafter more than doubles that. Acquire the drafter like
any model, then name its store directory as `draft_model`; the agent mounts it
read-only beside the target. The type and length go in `extra_args`, as in the
drafter's own instructions:

```sh
stead model acquire --source huggingface --id z-lab/Qwen3.8-27B-DFlash2-GGUF \
  --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 \
  --file 'Qwen3.8-27B-DFlash2-Q4_K_M.gguf' --node spark-alpha.internal
stead deployment modify qwen3.8-27b-ud-q4_k_xl-llamacpp --config '{
  "draft_model": "/var/lib/tensorstead/models/<drafter token>",
  "extra_args": {"alias": "qwen3.8-27b-ud-q4kxl",
                 "spec-type": "draft-dflash", "spec-draft-n-max": 7}}'
```

`<drafter token>` is the store's directory name for the drafter: the URL-safe
base64 of `huggingface:<repo>` (plus `#<digest>` for a file selection), as
`ls` of the model store shows. `extra_args` is replaced as a whole on modify,
so repeat any keys that should stay, such as the alias. Measured on a GB10 with
Unsloth's Qwen3.8-27B UD-Q4_K_XL: 11.6 tokens/s without the drafter, and with
it 26.9 on prose (29% of drafted tokens accepted) and 51.1 on code (70%). That
matches the NVFP4 model on SGLang with the same drafter.

**ExLlama (TabbyAPI).** TabbyAPI's published image is x86_64 only, and its
dependency list pins exllamav3 and torch wheels for x86_64, so on arm64 the
image is built on the node: the TensorRT-LLM image above supplies a CUDA 13
toolchain and a torch that targets GB10, exllamav3 is compiled for `sm_121`
(the longest step), and TabbyAPI is copied into the image's working directory,
because the adapter runs `python3 main.py` from there. `uvloop` is installed
explicitly: TabbyAPI declares it for x86_64 only but imports it on every
non-Windows platform.

```sh
stead buildspec set exllama-tabby-arm64 \
  --base 'nvcr.io/nvidia/tensorrt-llm/release@sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9' \
  --step 'git clone https://github.com/theroyallab/tabbyAPI.git /tmp/tabbyAPI && git -C /tmp/tabbyAPI checkout 2636ef602b55b5343cf3740dd55a77f21e0a8003 && cp -a /tmp/tabbyAPI/. /app/tensorrt_llm/ && rm -rf /tmp/tabbyAPI' \
  --step 'TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=16 python3 -m pip install --no-cache-dir --no-build-isolation --root-user-action=ignore git+https://github.com/turboderp-org/exllamav3.git@f45b84512ba65bca77f7d394399d4b68821fc71b' \
  --step 'cd /app/tensorrt_llm && python3 -m pip install --no-cache-dir --root-user-action=ignore --no-deps . && python3 -c "import tomllib; d = tomllib.load(open(\"pyproject.toml\", \"rb\"))[\"project\"][\"dependencies\"]; open(\"/tmp/req.txt\", \"w\").write(chr(10).join(x for x in d if not x.split()[0].startswith((\"torch\", \"exllamav3\", \"flash\"))))" && python3 -m pip install --no-cache-dir --root-user-action=ignore -r /tmp/req.txt uvloop' \
  --step 'cd /app/tensorrt_llm && python3 -c "import exllamav3, torch, uvloop" && python3 main.py --help >/dev/null'
stead image build exllama-tabby-arm64 --node spark-alpha.internal \
  --reference local/exllama-tabby:1.6.1
```

The build cannot import exllamav3's CUDA extension against a GPU -- a Docker
build has none -- so the last step checks that the packages import and that
TabbyAPI parses its arguments; whether it loads a model is the deployment's
first start. Serve an EXL3 model (`turboderp/Qwen3-8B-exl3`, for example, at
a pinned `4.0bpw` revision); EXL2 and GPTQ are refused before the container
starts.

#### TensorFold

[TensorFold](https://github.com/ashhart/TensorFold) publishes a native
`linux-aarch64` binary rather than an image. It needs glibc, the driver's
`libcuda` and, for a two-node group, NCCL and the CUDA 13 runtime. The base below
is the CUDA 13 image the estate already holds for vLLM, so the build downloads
only the 23 MB release. The archive's SHA-256 is written into the step, so a
different file fails the build.

```sh
stead buildspec set tensorfold-1.1.1 \
  --base 'eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595' \
  --step 'curl -fsSL -o /tmp/tf.tar.gz https://github.com/ashhart/TensorFold/releases/download/v1.1.1/tensorfold-1.1.1-linux-aarch64.tar.gz && echo "2e0103a544e64354bfec65bd8a052ede17aa9be1f6a56d8e37c868af9139cb39  /tmp/tf.tar.gz" | sha256sum -c - && mkdir -p /opt/tensorfold && tar -xzf /tmp/tf.tar.gz -C /opt/tensorfold --strip-components=1 && rm /tmp/tf.tar.gz && ln -s /opt/tensorfold/bin/tensorfold-native /usr/local/bin/tensorfold-native' \
  --step 'test "$(tensorfold-native --version)" = "tensorfold-native 1.1.1"' \
  --entrypoint tensorfold-native
stead image build tensorfold-1.1.1 --node spark-alpha.internal --node spark-beta.internal \
  --reference local/tensorfold:1.1.1-cuda13
```

A deployment then uses `runtime_type: tensorfold` and that image. The adapter
runs `tensorfold-native serve <model directory>`. It delivers the inference key
as `TENSORFOLD_API_KEY` and probes `/v1/models` for readiness. The model is
NVIDIA's NVFP4 export or another checkpoint TensorFold lists for CUDA. Its
README's model table is the authority; not every family it serves on Metal
has a CUDA path.

**Two nodes.** For the families TensorFold runs as a group on CUDA (Nemotron
Super VL, GLM-5.3-Flash), declare both nodes and `distributed_interface`, the
ConnectX-7 port. Every node needs the same checkpoint, because each rank hashes
its copy and the group refuses to serve if they differ. Rank 1 serves no API.

What the estate measured with 1.1.1, on the same stored checkpoints and
requests as the vLLM deployments it was compared with:

| Model (NVFP4) | Nodes | TensorFold decode | vLLM decode | Prefill (TF / vLLM) | Load (TF / vLLM) |
| --- | --- | --- | --- | --- | --- |
| Nemotron 3.5 Lightning 30B-A3B | 1 | 140 tok/s | 103 (MTP-1) | 8,190 / 6,980 tok/s | 14 s / 176 s |
| Nemotron 3.5 Super VL 120B-A12B | 1 | 50 tok/s | 48 (DFlash-7) | 2,340 / 2,740 tok/s | 65 s / 536 s |
| Nemotron 3.5 Super VL 120B-A12B | 2 | 61 tok/s | — | 1,350 tok/s | 300 s |
| Nemotron 3.5 Bolt 74B-A9B | 1 | 39 tok/s | 81 (DFlash) | 2,550 / 1,510 tok/s | 8 s / 227 s |

Bolt is the counter-example. TensorFold has no drafter for it, so vLLM's DFlash
wins single-stream decode by 2×. TensorFold's greedy output was identical
across repeats in every run. With eight concurrent requests it reached 262 tok/s
in total on Lightning, 90 on Super VL (one node), 125 on Super VL (two nodes)
and 151 on Bolt.
