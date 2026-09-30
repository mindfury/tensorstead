# Tensorstead Ansible Automation

This is external installation tooling for already-provisioned Linux hosts. It installs the Tensorstead coordinator and node-agent services, then provides a read-only readiness check. Tensorstead itself does not invoke Ansible and Ansible does not register nodes or manage Tensorstead workloads.

## What it does

- Installs a controller-built Tensorstead wheel in `/opt/tensorstead/venv`.
- Creates and enables `tensorstead-coordinator` and/or `tensorstead-agent` systemd units according to inventory membership.
- Supports a coordinator host that is either separate from, or the same as, one or more agent hosts.
- Checks Linux/systemd, Python, Docker, NVIDIA Container Toolkit, services, HTTP health, and coordinator-to-agent TLS reachability.

## What it does not do

- Install an operating system, Docker, GPU drivers, NVIDIA Container Toolkit, certificates, networking, or vendor cluster configuration.
- Download model artifacts or runtime images.
- Register a node, create a deployment, start inference, route traffic, or alter any Tensorstead workload.

## TLS is automatic in the normal setup

You do **not** need to create certificates, keys, or a CA bundle before your first normal `site.yml` run. TLS is how the coordinator confirms it is talking to *your* Spark rather than an impostor on the network, and the playbook now prepares that private trust automatically.

In plain English:

- The **agent certificate** is the Spark's network ID card. It says, “I am `spark-alpha.internal`.” Ansible creates it in normal mode.
- The **agent private key** is the secret half of that ID card. Ansible creates it on the agent Spark and it stays there.
- The **CA bundle** is the trust file. Ansible creates its private source on the controller and copies only the public trust file to the selected hosts.

For the default paths, the playbook creates or installs the files as follows:

| File | Default path | Required on |
|---|---|---|
| CA bundle | `/etc/tensorstead/ca.pem` | Coordinator host **and** every agent host |
| Agent certificate | `/etc/tensorstead/agent.crt` | Each agent host only |
| Agent private key | `/etc/tensorstead/agent.key` | Each agent host only |

The automatically-created certificate name is the inventory hostname. For example, if the host is `spark-alpha.internal`, the certificate includes `spark-alpha.internal` as a DNS name. Make sure that name resolves from the coordinator; do not substitute a different hostname just because it resolves to the same IP address.

Most private hostnames such as `spark-alpha.internal` need a private/internal certificate arrangement, not a public website certificate.

### Advanced: use an existing certificate system

Only choose this path when you already have internal or organization-managed certificates. Set `tensorstead_managed_tls: false`, then identify the CA-bundle, agent-certificate, and agent-private-key paths. Confirm the certificate includes the agent DNS name and install those files before `site.yml`.

If you need to understand the manual alternative, [Private CA and Spark certificate](./tls-private-ca.md) remains available as background and an advanced override. It is not required for normal setup.

Never paste a private key or either Tensorstead token into Git, chat, terminal history, or an Ansible inventory. Keep tokens in Ansible Vault. Keep the private key on the agent host with owner-only permissions.

## Inference API key

Normal `site.yml` runs create one random inference API key on the Ansible
controller at `~/.local/share/tensorstead/inference-api-key` (owner-only). Agent
services use it to require a Bearer token for vLLM inference requests; it is
not stored in the coordinator database or an exported deployment definition.

Use the key from a local file rather than pasting it into a command:

```sh
curl http://spark-alpha.internal:8000/v1/models \
  -H "Authorization: Bearer $(<~/.local/share/tensorstead/inference-api-key)"
```

## SSH prerequisite

Before a host-changing Ansible run, the Ansible controller must be able to sign in to every Spark without an interactive password. Normally this means:

1. Create or choose an SSH key on the controller.
2. Authorize its **public** key for the chosen remote user on each Spark.
3. Set `ansible_user` and, if needed, `ansible_ssh_private_key_file` in your private inventory.
4. Confirm a non-interactive test works, for example:

   ```sh
   ssh -i ~/.ssh/tensorstead_spark_ed25519 <remote-user>@spark-alpha.internal 'hostnamectl --static'
   ```

The remote user needs permission to use `sudo` because `site.yml` creates system services and protected files. It does **not** need password-free `sudo`: when you eventually run a real deployment, Ansible will ask for your normal Spark password once. Do not put the SSH private key itself in the repository or inventory.

## Prepare a local profile

1. Build a wheel and record its provenance: `make build`.
2. Copy `inventories/example/hosts.yml` and `group_vars/all.yml.example` into local files outside Git.
3. Put `tensorstead_management_token` and `tensorstead_replication_token` in Ansible Vault. They must be distinct.
4. Keep `tensorstead_managed_tls: true` (the default). No certificate paths need changing for normal setup.

The coordinator environment file is readable only by the restricted `tensorstead` account. The agent requires root because it manages Docker and systemd units; its environment file is root-only. Neither template writes secret values to the repository, and secret-bearing template tasks use `no_log`.

## Safe local validation versus deployment

Safe local validation (does **not** connect to or change a Spark):

- Build the wheel.
- Run `uv sync --all-extras`.
- Run both `--syntax-check` commands and `ansible-lint` below.

Host-changing deployment (only after you deliberately choose to proceed):

- `site.yml` connects over SSH and installs/configures services.
- `readiness.yml` connects over SSH but is read-only: it checks prerequisites, services, and HTTPS reachability. It does not register a node or manage workloads.

## Validate without hardware

```sh
uv sync --all-extras
uv run ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/site.yml --syntax-check
uv run ansible-playbook -i ansible/inventories/example/hosts.yml ansible/playbooks/readiness.yml --syntax-check
uv run ansible-lint ansible/
```

The example inventory uses documentation-only IP addresses. Syntax checking does not contact them.

## Run later, when you choose to touch hosts

```sh
uv run ansible-playbook -i /path/to/hosts.yml ansible/playbooks/site.yml --ask-vault-pass --ask-become-pass
uv run ansible-playbook -i /path/to/hosts.yml ansible/playbooks/readiness.yml --ask-vault-pass --ask-become-pass
```

Review the per-host Ansible recap. A partial failure identifies the host that needs attention; a healthy host is not automatically removed or rolled back. Once readiness succeeds, register agents using the normal `stead node register` CLI/API workflow.
