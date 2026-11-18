---
title: Manage runtime container images in Tensorstead
sop_id: SOP-05
applies_to: Tensorstead 1.13
keywords: [image pull, image build, buildspec, image import, image reconcile, image delete, digest, image_id, arm64, DGX Spark, TensorRT-LLM, llama.cpp, ExLlama, TabbyAPI, TensorFold, xgrammar, vLLM audio]
---

# SOP-05: Manage runtime container images in Tensorstead

## Purpose of this procedure

Get the runtime container image a deployment needs onto its nodes: pull a
registry image, build one from a recorded recipe, or import an archive. Also
covers listing and deleting images.

## Choose an image

```sh
stead runtime list      # supported runtimes, multi-node support, suggested images
stead image list        # what each node already holds
```

Suggested images are starting points, not guarantees. Pin images by digest
(`repo@sha256:...`) rather than a moving tag.

| Runtime | Multi-node | Image notes for DGX Spark (arm64, GB10) |
| --- | --- | --- |
| `vllm` | yes | `nvcr.io/nvidia/vllm:<tag>-py3` (NGC) |
| `sglang` | yes | `lmsysorg/sglang`; arm64 availability not established |
| `llamacpp` | no | `ghcr.io/ggml-org/llama.cpp:server-cuda` runs as published; pin a digest |
| `trtllm` | no | `nvcr.io/nvidia/tensorrt-llm/release` runs as published |
| `exllama` | no | TabbyAPI's image is x86_64 only; build on the node (recipe in `docs/builds.md`) |
| `tensorfold` | yes (2 nodes, some families) | No image published; build from the `tensorfold-1.1.1` spec in `docs/builds.md` |

## Stage a registry image before a planned start

```sh
stead image pull 'nvcr.io/nvidia/vllm@sha256:<digest>' --node spark-01 --node spark-02
stead operation show <operation id>
```

The registry is contacted once. A node that already holds the image is the
source; otherwise the first node pulls and the rest copy agent to agent.
Staging keeps the download out of the downtime window. A multi-node start does
the same staging itself and fails with `image_staging_failed` if a copy fails.

## Build an image from a recorded build spec

1. Record the spec. This executes nothing:

   ```sh
   stead buildspec set vllm-audio \
     --base 'nvcr.io/nvidia/vllm@sha256:<digest>' \
     --step 'python3 -m pip install --no-cache-dir --root-user-action=ignore av==<v> scipy==<v> soundfile==<v> soxr==<v>' \
     --step 'python3 -c "import av, scipy, soundfile, soxr"'
   stead buildspec list    # shows whether each base is pinned
   ```

2. Build once and copy to every node that will run it:

   ```sh
   stead image build vllm-audio --node spark-01 --node spark-02 --reference local/vllm:audio
   ```

Build spec rules:

- Pin `--base` by `@sha256:`. An unpinned base is reported as unreproducible
  but not refused.
- Pin exact package versions in steps. A range resolves differently over time.
- Use `pip install --no-deps` when patching one package in a runtime image, and
  name every dependency explicitly, so pip cannot replace `torch` or
  `transformers`.
- End with a step that asserts the result (an import or version check). A
  failing step fails the build, so a broken image is never tagged.
- `--entrypoint` sets the image entrypoint when the runtime needs it
  (TensorFold uses `tensorfold-native`).
- Build with every node named in one command. Separate builds give each node a
  different `image_id`.
- A build returns an `image_id`, not a registry digest. A locally built image
  has no registry digest.

Verified DGX Spark recipes (TensorRT-LLM, ExLlama/TabbyAPI, TensorFold, the
vLLM xgrammar repair) are in `docs/builds.md`. For speech-to-text audio support
see `docs/speech-to-text.md`.

## Import an image archive

Place the archive in the node's managed image store
(`/var/lib/tensorstead/images` by default), then:

```sh
stead image import vllm-vendor.tar --node spark-01 \
  --reference local/vllm:vendor --expect sha256:<digest>
```

`import` takes a file name inside the store, never a path. With `--expect`, a
mismatched archive imports nothing.

## Tidy and delete images

```sh
stead image reconcile               # drop records whose image is gone from the node
stead image delete <node-id> <digest>
stead buildspec delete <spec>       # refused while an image it produced is referenced
```

`image delete` is refused (`image_in_use`) while any retained deployment
revision references the image. Removing a deployment never deletes its image.
