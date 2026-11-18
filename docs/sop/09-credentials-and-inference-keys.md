---
title: Manage source credentials and inference keys in Tensorstead
sop_id: SOP-09
applies_to: Tensorstead 1.13
keywords: [credential set, Hugging Face token, gated model, default credential, promote, inferencekey, API key, Bearer, VLLM_API_KEY, LLAMA_API_KEY, TENSORFOLD_API_KEY, endpoint_authenticated, credential import directory, from-file, from-env, node-wide inference key, credential_not_enforceable]
---

# SOP-09: Manage source credentials and inference keys in Tensorstead

Tensorstead has two kinds of secret, and they do different jobs:

- A **source credential** lets an agent **download** a model from a source
  such as Hugging Face.
- An **inference key** is what **clients** must present to a runtime endpoint.

Neither can be passed as a flag value, and no command ever returns a stored
value.

## How to supply a secret safely

Every `set` command reads the secret from one of:

- an interactive prompt (default), or stdin when piped;
- `--from-env <VAR>`: an environment variable in the CLI's environment;
- `--from-file <name>`: a file **inside the coordinator's credential import
  directory** (`/var/lib/tensorstead/credential-imports`, owned by root, group
  `tensorstead`, mode 0750). Place the file on the coordinator host:

  ```sh
  sudo install -m 0640 -g tensorstead ./hf-token /var/lib/tensorstead/credential-imports/hf-token
  ```

  Paths outside that directory, traversing names and symlinks are refused.

## Source credentials for gated models

```sh
stead credential set huggingface personal --default     # prompt
stead credential set huggingface team --from-file hf-token
stead credential list                                    # names and default only
```

- A source can hold several named credentials and exactly one default. An
  acquisition with no `--credential` uses the default.
- Deleting the default leaves the source with **no** default and says so. To
  hand over deliberately: `stead credential delete huggingface personal --promote team`.
- Credentials are stored outside the database and appear in no export, record
  or log.

## Protect a deployment endpoint with an inference key

Inference keys are optional. Without one a deployment serves unauthenticated,
and `deployment show` reports `endpoint_authenticated: false`.

```sh
stead inferencekey set prod-key            # prompt; or --from-env / --from-file
stead inferencekey bind <deployment> --name prod-key
stead deployment restart <deployment>      # the binding takes effect only on restart
```

- Binding creates a new revision and reports `restart_required`. The runtime
  keeps the old key until you restart.
- The key reaches the runtime as an environment variable, never on the command
  line: `VLLM_API_KEY` (vLLM), `LLAMA_API_KEY` (llama.cpp),
  `TENSORFOLD_API_KEY` (TensorFold).
- SGLang, ExLlama and TensorRT-LLM cannot enforce a key in a way Tensorstead
  can use safely. Binding a key to one of those refuses the start rather than
  serving unauthenticated.
- Clients send `Authorization: Bearer <key>`.

Clear and delete:

```sh
stead inferencekey bind <deployment>       # no --name clears the binding
stead inferencekey delete prod-key         # refused while a deployment binds it
stead inferencekey list                    # names only
```

## Node-wide default inference key

If the installation set `tensorstead_inference_api_key_enabled: true`, every
agent has a default key, generated on the Ansible controller at
`~/.local/share/tensorstead/inference-api-key`. It applies to deployments with
no binding of their own, **only where the runtime can enforce it**. Runtimes
that cannot still start, unauthenticated. Use it from the file:

```sh
curl http://spark-01.internal:8000/v1/models \
  -H "Authorization: Bearer $(< ~/.local/share/tensorstead/inference-api-key)"
```

## Management and replication tokens

The management token (every API and MCP call) and replication token
(agent-to-agent copies) live in Ansible Vault and are installed by
`make deploy`. Changing them means editing the Vault and re-running
`make deploy`, then updating every client's token file. To give one node its
own management token, see SOP-03.
