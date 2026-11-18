---
title: Acquire, copy and delete models in Tensorstead
sop_id: SOP-04
applies_to: Tensorstead 1.13
keywords: [model acquire, Hugging Face, Azure ML, revision, commit SHA, GGUF, file selection, replication, node to node copy, digest, content port, allow-upstream-fallback, acquisition_in_progress, replication_failed, authorization_refused, model delete]
---

# SOP-04: Acquire, copy and delete models in Tensorstead

## Purpose of this procedure

Put model weights onto one or more nodes from Hugging Face or the Azure ML
registry, copy an existing model to more nodes, and delete models you no longer
need. Model bytes go straight from the source to the agent. They never pass
through the coordinator.

## Before acquiring: gated models need a source credential

If the model requires an account or accepted licence, store a credential once
(SOP-09):

```sh
stead credential set huggingface personal --default    # value from a prompt
```

If the provider's access terms are not accepted, the acquire is refused with
`authorization_refused`. Accept the terms on the provider's website.
Tensorstead never accepts them for you.

## Acquire a Hugging Face model

```sh
stead model acquire --source huggingface --id <org/model> \
  --revision <40-char commit SHA> --node spark-01
```

- Always pin `--revision` to a commit SHA, not `main`. Tensorstead records the
  resolved revision either way, but a moving label is not reproducible.
- The command returns an operation id. Poll it:
  `stead operation show <operation id>`. Large models take a long time. The
  coordinator keeps checking a download for up to 6 hours.
- When it finishes: `stead model list`, then `stead model show <model>` for the
  resolved revision, digest and size. Models can be named by id prefix or by
  their unique source name (for example `nvidia/Qwen3.6-27B-NVFP4`).
- `--credential <name>` uses a named credential instead of the source default.

## Acquire one file from a GGUF repository

GGUF repositories ship many quantizations. Select one with `--file`
(repeatable glob). The selection becomes part of the model's identity.

```sh
stead model acquire --source huggingface --id unsloth/Qwen3.8-27B-GGUF \
  --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  --file 'Qwen3.8-27B-UD-Q4_K_XL.gguf' --node spark-alpha.internal
```

## Acquire an Azure ML registry model

No Azure account is needed for publicly published models. The id is
`<registry>/<model name>` and a version is required:

```sh
stead model acquire --source azureml --id azureml/MAI-Code-1.1-Flash-On-Device \
  --revision 2 --node spark-alpha.internal
```

The recorded revision is `<version>@<manifest SHA-512>`. A later acquire is
refused if the registry's files for that version have changed. An unknown
source is refused with `unknown_model_source`.

## Place a model on several nodes

Name every node with a repeated `--node`. The model downloads from upstream
once, onto the first node, and every other node copies it from a node that
already holds a verified replica. The copy is checked against the content
digest. Re-running `model acquire` with an extra `--node` for a model already on
the estate copies it to the new node.

```sh
stead model acquire --source huggingface --id <org/model> --revision <sha> \
  --node spark-01 --node spark-02
```

- A failed copy is refused with `replication_failed`, naming both nodes. It is
  not silently downloaded again. `--allow-upstream-fallback` permits a second
  upstream download.
- A second acquire of the same repository while one runs is refused with
  `acquisition_in_progress`. Wait for the first to finish, then re-run.
- Re-running an acquire after a timeout is cheap: a destination that already
  holds a verified copy answers at once.

## Copy a model acquired before digests existed

Copies need a recorded content digest. Older models may lack one, and their
copy is refused with "no content digest is recorded to verify a copy against".

```sh
stead model digest <model>                 # hashes the whole tree; takes minutes
stead operation show <operation id>
stead model acquire --source huggingface --id <org/model> --revision <sha> --node <new node>
```

A model that already has a digest is verified rather than overwritten. If the
bytes no longer match, `digest_mismatch` reports both values.

## Speed up node-to-node copies

With `tensorstead_agent_content_port` set (8444 in the example group vars),
agents serve model bytes on a second port, one thread per connection, which
roughly doubled copy speed between two DGX Sparks. Peers must be able to reach
that port. If they cannot, the copy falls back to the main agent port after a
10-second connect timeout, which is slower but does not fail. Agent environment
`TENSORSTEAD_REPLICATION_CONNECTIONS` (default 8) and
`TENSORSTEAD_REPLICATION_HASHERS` (default 4) tune parallelism.

## Delete a model

```sh
stead model list
stead model delete <model>
```

Refused (`still_referenced`) while any retained deployment revision, including
an old one, names the model. Remove those deployments first (SOP-14). Removing
a deployment never deletes its model.

## Cancelling an acquisition

There is no cancel. An acquisition of the wrong model runs to completion or
failure. Delete it afterwards.
