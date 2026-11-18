---
title: Approve code-loading or isolation-relaxing options in Tensorstead
sop_id: SOP-10
applies_to: Tensorstead 1.13
keywords: [code approval, trust_remote_code, allow_io_uring, io_uring, seccomp, code_execution_not_authorized, loads-code, relaxes-isolation, code_approval_create, approved_by, extra_args refused, GLM-5.3 b12x]
---

# SOP-10: Approve code-loading or isolation-relaxing options in Tensorstead

## When this procedure applies

A deployment start is refused with `code_execution_not_authorized` because its
runtime config sets an option that runs code from the model repository or
loosens container isolation:

| Option | Runtimes | Effect class | Why it is gated |
| --- | --- | --- | --- |
| `trust_remote_code` | vllm, sglang, trtllm | loads-code | Runs Python shipped in the model repository |
| `allow_io_uring` | vllm | relaxes-isolation | Adds `io_uring_setup`, `io_uring_enter` and `io_uring_register` to Docker's default seccomp profile; needed for loaders such as `--load-format b12x` |

Tensorstead never runs a container privileged or with `seccomp=unconfined`.
These options cannot be passed through `extra_args`.

## What an approval is

A code approval is a separate, immutable record that permits one option for
one exact tuple: runtime, model source, model, model revision, and image
digest. Both the coordinator and the agent check it at start. It stops matching
when the model revision or image changes.

It is an **operator attestation**, not a second person's sign-off: any holder
of the management token, including an AI agent, can create one, and
`approved_by` is recorded as written, not authenticated.

## Step 1: Review before approving

- Read the code the option will run (for `trust_remote_code`, the model
  repository's Python files at the pinned revision).
- Confirm the image is pinned and note its digest: `stead image list`.
- Confirm the model revision: `stead model show <model>`.

## Step 2: Create the approval (API or MCP only)

There is no `stead` command for approvals. Use MCP `code_approval_create` or
the coordinator API:

```sh
curl --cacert "$TENSORSTEAD_CA_BUNDLE" \
  -H "Authorization: Bearer $(< ~/.local/share/tensorstead/management-token)" \
  -H 'Content-Type: application/json' \
  https://spark-alpha.internal:8080/v1/code-approvals -d '{
    "option": "trust_remote_code",
    "runtime_type": "vllm",
    "model_source_id": "huggingface",
    "source_model_id": "<org/model>",
    "model_revision": "<40- or 64-char hex revision>",
    "image_digest": "sha256:<digest>",
    "reason": "<what was reviewed>",
    "approved_by": "<reviewer>"
  }'
```

`model_revision` must be an immutable hex revision, never a tag.

## Step 3: Start the deployment

```sh
stead deployment start <name>
```

## List and revoke approvals

MCP `code_approval_list` / `GET /v1/code-approvals` lists approvals, newest
first. MCP `code_approval_delete` / `DELETE /v1/code-approvals/{id}` revokes
one. Revoking deletes the row, so no record of the revocation remains.

## Flags refused in extra_args

In vLLM and SGLang, `extra_args` refuses every code-loading flag, in any
spelling: `--tool-parser-plugin`, `--reasoning-parser-plugin`,
`--chat-template`, `--logits-processors`, `--worker-cls`,
`--worker-extension-cls`. Prefer a runtime's built-in parser (for example
vLLM's `nemotron_v3` reasoning parser for Nemotron 3.5 Bolt) over a plugin.
