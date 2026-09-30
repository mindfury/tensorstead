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
make deploy TENSORSTEAD_DEPLOY_ARTIFACT=dist/builds/3/tensorstead-1.0.1-py3-none-any.whl
```

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
