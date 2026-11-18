---
title: Register and manage Tensorstead nodes
sop_id: SOP-03
applies_to: Tensorstead 1.13
keywords: [node register, node list, node show, reachability, resources, reserve, unreserve, deregister, rotate-management-token, agent_unreachable, agent_version_incompatible, node_reserved]
---

# SOP-03: Register and manage Tensorstead nodes

## Purpose of this procedure

Add an inference host to the Tensorstead inventory, inspect it, mark it
off-limits, give it its own credential, or remove it.

## Register a node

Prerequisite: the node agent is installed and running (SOP-01). Registration
checks the agent; it never installs it.

```sh
stead node register --name spark-01 --agent https://spark-01.internal:8443
```

On success you get the node id, contract version, and platform facts. Use the
agent's DNS name so TLS verification succeeds.

| Refusal | Meaning | Action |
| --- | --- | --- |
| `agent_unreachable` | The agent did not answer | Check `make services-status` on the host and the URL |
| `agent_version_incompatible` | Agent and coordinator wire contracts differ | Upgrade whichever is older (SOP-13) |

Registration trusts the URL you give it and sends the fleet management token
to it. Only register hosts you control. The coordinator should have
`TENSORSTEAD_AGENT_CA_BUNDLE` set to the private CA (the Ansible role sets it)
so a host with a public certificate cannot pass as an agent.

## Inspect nodes (read-only)

```sh
stead node list
stead node show spark-01
stead node reachability spark-01   # asks the agent now
stead node resources spark-01      # accelerator, memory and storage, read now
```

Resource readings are for your judgement; Tensorstead does not choose placement
from them. (The only memory check is at start: see `insufficient_memory` in
SOP-15.) On a
DGX Spark, CPU and GPU share one memory pool, so every deployment on the node
draws from the same memory.

## Reserve a node for work Tensorstead cannot see

```sh
stead node reserve spark-02 --reason "manual benchmark outside Tensorstead"
stead node unreserve spark-02
```

While reserved, deployment create, modify and start that target the node are
refused with `node_reserved`. Nothing on the node changes and nothing already
running there is stopped. Reserve and unreserve are CLI/API operations. An AI
agent that meets `node_reserved` should ask the operator, not unreserve.

## Give a node its own management token

By default every agent accepts the same fleet-wide management token, so one
compromised agent exposes a credential for every node. To narrow that:

1. Install a new random value as `TENSORSTEAD_MGMT_TOKEN` in that agent's own
   environment file (out of band, through your normal secure method) and
   restart the agent.
2. Tell the coordinator to present that value to this node:

   ```sh
   stead node rotate-management-token spark-01   # reads the value from a prompt or stdin
   ```

To return to the fleet token: `stead node clear-management-token spark-01`
(after restoring the fleet value on the agent). Registration itself still uses
the fleet token.

## Deregister a node

```sh
stead node deregister spark-01
```

Refused while any deployment names the node. Remove those deployments first
(SOP-14). Deregistering removes the record only. It does not uninstall the
agent or delete files on the host.
