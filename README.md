# Tensorstead

[![CI](https://github.com/mindfury/tensorstead/actions/workflows/ci.yml/badge.svg)](https://github.com/mindfury/tensorstead/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A steady home for your models.**

Tensorstead is a management plane for self-hosted AI inference appliances. It
acquires model weights, places them on your machines, and starts, stops, and
restores inference runtimes (vLLM, SGLang, llama.cpp) exactly as you declared
them — and it keeps a complete, exportable record of every change. It is
equally usable by a human at a CLI and by an AI agent over MCP.

It never sits in the inference data path. Your clients talk to the runtime
directly; Tensorstead only manages what is running and why.

```text
$ stead deployment list
DEPLOYMENTS  (* has declared/observed drift; run `deployment show NAME`)
NAME       DESIRED  HEALTH   RUNNING/CURRENT  ENDPOINT                MODEL
chat       running  running  3/3              spark-01.internal:8000  huggingface:org/chat-model
coder      running  running  1/2              spark-02.internal:8001  huggingface:org/coder-model
```

`1/2` means revision 2 is declared but revision 1 is still running: that change
needs an explicit restart, which Tensorstead will never do on its own.

## Why it exists

Running your own inference hardware tends to drift into a pile of shell
history: a `docker run` with forty flags that only one person remembers, a
model downloaded by hand to a path nobody wrote down, a runtime image patched
over SSH at 2 a.m. and never recorded. It works right up until a reboot, a
second machine, or a colleague — human or AI — needs to reproduce it.

Tensorstead replaces that with a small, strict contract:

- **Declared, then applied.** A deployment is a numbered revision — model,
  runtime, image, nodes, endpoint, configuration. Every revision is retained and
  exportable, so "what was running last Tuesday" has an exact answer.
- **Declared and observed, never blurred.** Status is fetched live from the
  hosts and shown *beside* the declaration, not merged into it. Drift is
  visible; reconciling it is an explicit command.
- **Nothing happens unless you ask.** No background polling, no auto-restart,
  no scheduler, no autoscaling. Leave it alone for a month and it will have done
  nothing you did not ask for.
- **Survives a reboot on its own.** A started deployment installs a
  self-sufficient boot unit on each host, so a power cycle brings inference back
  even if the coordinator is down.
- **Safe to hand to an agent.** The MCP surface exposes exactly the same
  operations as the CLI and API — and no shell, no SSH, no arbitrary paths.
  Anything an agent can do is recorded like anything a human does.
- **Honest about its limits.** Readings are reported for your judgement and
  never block an operation; failures name what failed and where; features that
  cannot be done truthfully (like "rollback") are [deliberately
  absent](docs/not-in-v1.md).

## How it works

```text
            you (stead CLI)        your AI agent (MCP)
                     \                  /
                      v                v
                 ┌──────────────────────────┐
                 │       coordinator        │  records declarations, revisions,
                 │  (API + SQLite ledger)   │  operations; drives agents
                 └────────────┬─────────────┘
               TLS + token-authenticated management calls
              ┌───────────────┴────────────────┐
              v                                v
     ┌──────────────────┐             ┌──────────────────┐
     │   node agent     │             │   node agent     │  acquires models,
     │  (inference box) │             │  (inference box) │  runs containers,
     │  vLLM / SGLang / │◄───────────►│  installs boot   │  installs boot units
     │  llama.cpp       │ runtime's   │  units           │
     └────────▲─────────┘ own fabric  └────────▲─────────┘
              │                                │
              └──── inference clients talk to the runtime directly ────
```

- **Coordinator** — the single source of truth. A FastAPI service with a
  SQLite store: nodes, models, images, deployments and their revisions,
  credentials (by reference), and an operation log.
- **Node agent** — one per inference host. Pulls model weights straight from
  the source (no model byte transits the coordinator), manages containers via
  Docker, and installs systemd units that restore deployments at boot.
- **CLI (`stead`) and MCP server (`tensorstead-mcp`)** — two clients of the same
  API. Every management operation is reachable from both; a contract test fails
  the build if one ever is not.
- **Ansible** — optional, external installation tooling for putting the
  coordinator and agents onto already-provisioned Linux hosts, with an automatic
  private CA for management TLS.

Multi-node deployments are supported when the runtime itself distributes
(vLLM and SGLang do; llama.cpp does not). Tensorstead coordinates; the runtime
distributes.

## Status and hardware

Tensorstead **1.0** is feature-complete for its scope (see [Not in
v1](docs/not-in-v1.md) for what is deliberately out of it) and is developed
against a two-node **NVIDIA DGX Spark** cluster (GB10, ConnectX-7 fabric,
DGX OS / Ubuntu) — that is the tested target. Any Linux host with an NVIDIA
GPU, Docker, and the NVIDIA Container Toolkit should work but is untested; the
DGX Spark-specific pieces (`prepare-node.yml`, fabric setup) are optional.
Reports from other hardware are very welcome.

## Quick start (no hardware needed)

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/mindfury/tensorstead.git
cd tensorstead
uv sync --all-extras
```

Start a local coordinator in one terminal:

```sh
uv run stead coordinator init
uv run stead coordinator serve --insecure-dev-mode   # loopback only, no token
```

And talk to it from another:

```sh
uv run stead login          # accept the loopback URL; leave the token file blank
uv run stead status         # what you are connected to, and where each setting came from
uv run stead runtime list   # supported runtimes and suggested images
uv run stead --help         # every command group
```

`--insecure-dev-mode` is refused on anything but a loopback address, so this
cannot be exposed by accident. To go further you need a node agent on a real
GPU host — see below.

## Deploying on real hardware

The full walkthrough is [First deployment](docs/first-deployment.md). In
outline:

1. **Build a traceable wheel** — `make build` (records a build number and
   SHA-256; see [builds](docs/builds.md)).
2. **Install the services with Ansible** — describe your hosts in a private
   inventory, put the two management tokens in Ansible Vault, and run the
   playbooks. TLS certificates are generated for you. See the [Ansible
   guide](ansible/README.md); on a freshly imaged DGX Spark, run
   `make prepare-nodes` first.
3. **Connect the CLI** — `stead login` with the coordinator URL, CA bundle, and
   token file from step 2 ([CLI setup](docs/cli-setup.md)).
4. **Serve a model:**

   ```sh
   stead node register --name spark-01 --agent https://spark-01.internal:8443
   stead model acquire --source huggingface --id org/model --revision <commit-sha> --node spark-01
   stead deployment create --name my-model --model <model-id> --runtime vllm \
       --image nvcr.io/nvidia/vllm:26.07-py3 --node spark-01 --endpoint spark-01.internal:8000
   stead deployment start my-model
   stead deployment show my-model   # declared and observed, side by side
   ```

5. **Point your inference clients at the runtime's own endpoint**
   (`http://spark-01.internal:8000/v1`). Day-two work — modify, restart,
   reconcile, remove, retention, credentials — is in
   [operations](docs/operations.md).

## Using it from an AI agent

Tensorstead was built to be driven by agents as comfortably as by people. Run
the MCP server locally over stdio (`tensorstead-mcp`), or use the TLS-protected
Streamable HTTP endpoint that Ansible installs beside the coordinator. Either
way the agent gets the full management surface — inspect, acquire, deploy,
start, stop, reconcile, even repair a broken runtime image from a recorded
build spec — and nothing that could run an arbitrary command on a host. See
[hosted MCP](docs/mcp.md).

If you are an agent reading this to deploy Tensorstead: follow
[First deployment](docs/first-deployment.md) and the [Ansible
guide](ansible/README.md), never put a token or private key in a command line
or a file under Git, and stop for the operator's approval before running any
playbook that changes a host.

## Documentation

| Doc | When you need it |
| --- | --- |
| [First deployment](docs/first-deployment.md) | Register a node, acquire a model, serve inference |
| [CLI setup](docs/cli-setup.md) | Pointing the CLI at a coordinator; any connection failure |
| [Ansible guide](ansible/README.md) | Installing the coordinator and agents on your hosts |
| [Operations](docs/operations.md) | Lifecycle, reconcile, retention, credentials, TLS, reboots |
| [Hosted MCP](docs/mcp.md) | Giving an AI agent access over the network |
| [Builds](docs/builds.md) · [Release candidate](docs/release-candidate.md) | Producing, tracking, and approving a deployable wheel; building a runtime image |
| [Smoke tests](docs/smoke-tests.md) | Read-only verification of a live installation |
| [Not in v1](docs/not-in-v1.md) | Why rollback, cancellation, and scheduling are absent |

## Development

```sh
uv sync --all-extras
make check            # lint, typecheck, tests, coverage — about 90 seconds
make ansible-check    # playbook syntax + ansible-lint; contacts no host
```

```text
src/tensorstead/   domain, service layer, coordinator, node agent, CLI, MCP, adapters
tests/             unit, contract, integration, and opt-in hardware tiers
ansible/           installation playbooks and roles
ops/githooks/      optional local gate: pre-commit (lint) and pre-push (make check)
```

GitHub Actions runs `make check` and `make ansible-check` on every push and pull
request (Python 3.12 and 3.14). `make hooks-install` points Git at the tracked
hooks in `ops/githooks/` to run the same gate locally before you push. Hardware tests are opt-in (`make test-hardware`) and
the Docker integration tests skip without a local daemon.

Contributions and bug reports are welcome — please open an issue first for
anything larger than a small fix.

## License

[MIT](LICENSE) © 2026 Philip Lincoln
