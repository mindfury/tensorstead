---
title: Install the Tensorstead control plane
sop_id: SOP-01
applies_to: Tensorstead 1.13
keywords: [install, ansible, site.yml, readiness, make deploy, vault, tokens, TLS, private CA, prepare-nodes, DGX Spark, wheel, build, inventory]
---

# SOP-01: Install the Tensorstead control plane

## Purpose of this procedure

Install the Tensorstead coordinator, hosted MCP server and node agents on
already-provisioned Linux hosts with the bundled Ansible playbooks, then
confirm the installation is ready. After this, models are deployed with
SOP-03, SOP-04 and SOP-06.

The Ansible tooling does **not** install the OS, Docker, GPU drivers or the
NVIDIA Container Toolkit. It does not download models, register nodes or
start inference.

## Prerequisites for installing Tensorstead

- Controller machine (where you run Ansible) with Python 3.12 or newer, `uv`,
  and a clone of the repository.
- Each inference host runs Linux with systemd, Python 3.12, Docker, an NVIDIA
  driver and the NVIDIA Container Toolkit.
- A stable DNS name for each host that resolves from the coordinator. TLS
  certificates are issued for the inventory hostname, so do not substitute an
  IP address.
- Non-interactive SSH from the controller to every host, as a user that can
  `sudo` (a password is fine; Ansible prompts for it).

## Step 1: Prepare freshly imaged DGX Spark nodes (DGX Spark only, host-changing)

Skip this step for hosts that are not DGX Sparks or are already prepared.

1. A reimage empties `authorized_keys`. Install the deploy key once with your
   password: `ssh-copy-id -i ~/.ssh/<deploy-key>.pub <user>@<host>`. Without
   it, the playbook fails at "Gathering Facts" with
   `Permission denied (publickey,password)`.
2. Get the operator's approval, then run `make prepare-nodes`. It is
   idempotent. It does not apply OS updates or rewrite EFI boot entries.

## Step 2: Build a traceable wheel (read-only for hosts)

```sh
uv sync --all-extras
make release-local     # lint, typecheck, tests, Ansible checks, then make build
```

`make build` writes the wheel and a dependency wheelhouse to `dist/`, archives
a copy in `dist/builds/<build_number>/`, and appends a record to
`.tensorstead/build-history.jsonl`. Record the wheel filename, build number,
SHA-256 and Git revision in your change record. A release candidate should show
`source_tree_dirty: false`.

To install a published release instead of a local build, see SOP-13
(`make deploy-release VERSION=<x.y.z>`).

## Step 3: Write the private inventory, settings and Vault (read-only for hosts)

1. Copy `ansible/inventories/example/hosts.yml` and
   `ansible/group_vars/all.yml.example` to files outside Git, for example under
   `~/.config/tensorstead/`.
2. In the inventory, put coordinator hosts in `tensorstead_coordinators` and
   inference hosts in `tensorstead_agents`. One host may be both. Set
   `ansible_user` and, if needed, `ansible_ssh_private_key_file`.
3. In settings, set `tensorstead_package_artifact` and
   `tensorstead_package_wheelhouse` to the paths from Step 2. Review the other
   values. The important ones:

   | Variable | Default | Meaning |
   | --- | --- | --- |
   | `tensorstead_coordinator_port` | 8080 | Coordinator HTTPS port |
   | `tensorstead_mcp_port` / `tensorstead_mcp_path` | 8090 / `/mcp` | Hosted MCP endpoint |
   | `tensorstead_agent_port` | 8443 | Agent management port |
   | `tensorstead_agent_content_port` | 8444 in the example | Faster node-to-node model copies; remove to disable |
   | `tensorstead_agent_model_store` | `/var/lib/tensorstead/models` | Where model weights live |
   | `tensorstead_managed_tls` | true | Ansible creates the private CA and certificates |
   | `tensorstead_inference_api_key_enabled` | false | Node-wide default inference key (SOP-09) |
   | `tensorstead_allow_boot_restoration` | false | Whether deployments may restart at boot (SOP-12) |
   | `tensorstead_services_enabled_at_boot` | false | Whether the three services start at boot |

   Bind hosts are IPv4 (`0.0.0.0`). Do not use `::`, which would stop IPv4
   clients connecting.
4. Create an Ansible Vault file holding `tensorstead_management_token` and
   `tensorstead_replication_token`. They must be different random values.
   Never put them in the inventory, settings, Git or chat.
5. Copy `ansible/deploy.mk.example` to `.tensorstead/deploy.mk` and set the
   three **paths** (inventory, settings, Vault). The file holds no secrets.

## Step 4: Validate without touching hosts (read-only)

```sh
make ansible-check
```

This runs playbook syntax checks and `ansible-lint`. It contacts no host.

## Step 5: Deploy the services (host-changing; get approval first)

```sh
make deploy        # prompts for the Vault password and the sudo password
```

What it does: copies the wheel and wheelhouse, installs them offline into
`/opt/tensorstead/venv` (hash-checked against `uv.lock`), writes environment
files and systemd units, creates the private CA and per-agent certificates when
`tensorstead_managed_tls` is true, and starts the coordinator, MCP and agent
services.

TLS files it places: CA bundle `/etc/tensorstead/ca.pem` on every host, agent
certificate `/etc/tensorstead/agent.crt` and key `/etc/tensorstead/agent.key`
on each agent. The controller keeps the CA private key under
`~/.local/share/tensorstead/ansible-tls/`. To use an existing organisation
certificate system instead, set `tensorstead_managed_tls: false` and follow
`ansible/tls-private-ca.md`.

Read the per-host recap. A failure names the host. A healthy host is not rolled
back because another failed.

## Step 6: Verify readiness (read-only)

```sh
make readiness
```

It checks Linux and systemd, Python, Docker, the NVIDIA Container Toolkit,
service state, HTTP health, coordinator-to-agent TLS reachability, and that the
management port no longer answers plain HTTP.

## Step 7: Connect a client and register nodes

Follow SOP-02 to configure `stead login`, then SOP-03 to register each agent.

## Verification checklist for a new installation

- `make readiness` passes on every host.
- `stead status` reports the coordinator version you built (a clean deploy from
  a checkout with an older `pyproject.toml` version installs the older wheel).
- `make test-smoke` passes once a deployment exists (see SOP-13).
