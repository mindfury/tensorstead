# First Deployment

This guide is the reference walkthrough for the first deployment. It walks a
first-time operator through serving inference on one node — register → acquire
→ create → start — with **no manual host mutation** after the node agent is in
place.

Everything below is driven entirely through the CLI. No shell command is run
against the inference host for model, runtime, deployment, or lifecycle
management. Inference clients reach the runtime's own endpoint directly.

## Prerequisites

1. **The node agent is already installed and running** on the inference host.
   Installation and software distribution are external to this product. This
   repository provides an operator-run [Ansible installation
   playbook](../ansible/README.md) for already-provisioned Linux hosts; it does
   not register nodes or manage workloads. Registration *verifies* the agent is
   running; it never installs it.
2. A running **coordinator**. For a local evaluation, initialise the store and
   start it in a second terminal:

   ```bash
   stead coordinator init
   stead coordinator serve --insecure-dev-mode
   ```

   `--insecure-dev-mode` runs an open coordinator with no management token. It
   is refused on anything but a loopback address, so it cannot be exposed by
   accident; an installed coordinator always has a token (set
   `TENSORSTEAD_MGMT_TOKEN`, or let Ansible provision it).

   Then, in your first terminal, save the connection once:

   ```bash
   stead login
   ```

   Accept the displayed loopback URL and leave the token-file answer blank for
   this local coordinator. For an installed appliance, use the coordinator URL,
   CA bundle, and token file supplied during installation. `stead login`
   verifies the connection before saving it.

## Step 1 — Register the node

Registration captures the agent's reported platform facts and records the host
in the inventory. It installs nothing.

```bash
stead node register --name spark-01 --agent https://10.0.0.11:8443
```

On success you get the node's id, contract version, and platform facts. If the
agent is unreachable or version-incompatible, the command reports
`agent_unreachable` / `agent_version_incompatible` — it never tries to
bootstrap the agent.

## Step 2 — Acquire the model

### If the model is gated

Some models require an upstream account and an accepted licence. Supply a
credential once; it is applied automatically from then on:

```bash
stead credential set huggingface personal --default   # prompts; never echoed
```

The secret is read from a prompt or from stdin — **there is no flag that takes
the value**, so it cannot land in your shell history or in a process listing.
It is stored outside the database and appears in no export, operation record,
or log.

If the model's *access terms* have not been accepted, the acquisition is
refused and says so. Accept them with the provider on their website; this
product will never accept or work around them on your behalf.


Acquire a model revision onto the node. The coordinator delegates retrieval to
`huggingface_hub`; no model byte transits the coordinator.

```bash
stead model acquire --source huggingface --id org/model --revision <immutable-commit-sha> --node spark-01
```

This returns an operation id to poll. Use the upstream immutable commit SHA
rather than a moving label such as `main`. Tensorstead also records the concrete
revision returned by the agent, so `model list` and exports can identify the
exact artifact that was acquired. When it completes, the model is listed:

```bash
stead model list
```

## Step 3 — Create the deployment

Define the deployment at revision 1 with `desired_state: stopped`. This
**alters no host state**.

```bash
stead deployment create --name llama-70b \
    --model <model-id> --runtime vllm --image repo/vllm:tag \
    --node spark-01 --endpoint 10.0.0.11:8000 \
    --config tensor_parallel_size=1
```

`--config` accepts repeatable `key=value` settings, or one JSON object when a
configuration is easier to read as a block. For example,
`--config '{"tensor_parallel_size": 1, "max_model_len": 32768}'`. A malformed
setting is rejected before it reaches the coordinator; runtime values then go
to the runtime's own validator (vLLM here).

## Step 4 — Start the deployment

Start it. The coordinator drives the agent to pull the image, start the
container, and install+enable a self-sufficient boot-restoration unit so a
reboot restores the running deployment.

```bash
stead deployment start llama-70b
```

## Step 5 — Confirm inference is serving

The deployment reports serving at its recorded endpoint:

```bash
stead deployment show llama-70b
```

Point an inference client **directly at the runtime's own endpoint**
(`http://10.0.0.11:8000`) — the product never stands in the inference data path.

That is the whole first deployment. No shell command was run against the
inference host; every management step went over the coordinator's contract.

## When something goes wrong

Failures name what failed and where, so the first move is always to read the
reason rather than to go looking on the host.

| Reason | What it means | What to do |
|---|---|---|
| `agent_unreachable` | The host's agent did not answer | Check the agent process and the endpoint recorded at registration |
| `agent_version_incompatible` | Agent and coordinator disagree on the contract | Upgrade whichever is older; nothing is attempted across the gap |
| `authorization_refused` | Upstream refused the model | Set a credential, or accept the model's terms with the provider |
| `runtime_not_distributed` | Multi-node asked of a runtime that cannot | Use one node, or a runtime that distributes |
| `endpoint_conflict` | Another deployment already holds that node+port | Choose a different endpoint |
| `still_referenced` | Deletion refused; something retained still names it | Remove the referring deployment first |
| `partial_failure` | Some nodes changed, others did not | Read the per-node outcomes; nothing was rolled back |

`stead operation list --deployment <name>` shows the full history, including
the reason for anything that failed.

## Next steps

- **Day-two operations** — modify, restart, reconcile, remove, and what is
  retained: [operations.md](./operations.md)
- **What v1 deliberately does not do**: [not-in-v1.md](./not-in-v1.md)
- **More than one node** — a deployment can span several nodes when the runtime
  itself distributes (vLLM does; llama.cpp does not). Pass `--node` more than
  once. The product coordinates; the runtime distributes.
- **Driving this from an AI agent** — the same operations are available over
  MCP with no shell and no SSH. Run `tensorstead-mcp` and point your agent at it.
