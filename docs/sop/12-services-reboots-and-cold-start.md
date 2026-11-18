---
title: Tensorstead services, node reboots and cold start
sop_id: SOP-12
applies_to: Tensorstead 1.13
keywords: [services-status, services-start, services-restart, services-stop, services-boot-enable, services-boot-disable, make reboot, cold start, power cycle, boot restoration, restore-on-boot, TENSORSTEAD_ALLOW_BOOT_RESTORATION, tensorstead_services_enabled_at_boot, wedged node, power button]
---

# SOP-12: Tensorstead services, node reboots and cold start

## The three control-plane services

`tensorstead-coordinator`, `tensorstead-mcp` and `tensorstead-agent` are
systemd units. Restarting them does **not** stop inference: deployments run in
their own containers under their own units.

All `make services-*` targets use the operator profile in
`.tensorstead/deploy.mk`, report per host and unit, and skip units a host does
not have.

```sh
make services-status          # read-only: installed, enabled, running
make services-start           # start now; boot arrangement unchanged
make services-restart         # restart in place; safe on a busy estate
make services-stop            # stop now; boot arrangement unchanged
make services-boot-enable     # start at boot; starts nothing now
make services-boot-disable    # do not start at boot; stops nothing now
```

Use these instead of `make deploy` when you only need services up again.
`make deploy` reinstalls everything and refuses if this checkout's wheel is
missing.

## What comes back after a reboot

By default, **nothing** comes back on its own:

- The services are not enabled at boot (`tensorstead_services_enabled_at_boot`
  defaults to false, and `make deploy` enforces the setting).
- Deployments are not restored at boot. Restoration needs **both**
  `tensorstead_allow_boot_restoration: true` for the node (set only through
  Ansible with Vault and sudo passwords) **and** `--restore-on-boot` on the
  deployment. The node switch is out of reach of the management token, so an AI
  agent can request restoration but never grant it.

Enable boot restoration for a deployment only once it is known good. A
deployment that wedges the GPU driver and restarts at every boot can leave a
node unrecoverable without a reimage.

## Reboot one node (host-changing; get approval first)

```sh
make reboot TENSORSTEAD_REBOOT_LIMIT=spark-beta.internal
```

- One host at a time. The limit is required. `-e tensorstead_reboot_all=true`
  is the deliberate override; avoid it when a deployment spans both nodes.
- Before rebooting, the playbook reports the node's boot-restoration setting
  and any deployment units enabled at boot. Read them.
- A multi-node deployment loses its group when one member reboots. Stop it
  first, or plan the outage.

## Cold start after a reboot or power cycle

```sh
make services-status            # did the units come back?
make services-start             # if any did not
make readiness                  # coordinator reaches every agent
stead deployment list           # declared versus observed
stead deployment start <name>   # for each deployment that should be serving
```

Deployments declared `running` whose containers are gone show `not_running`.
`deployment start` checks the node first: it restarts a declared-running
deployment whose container is gone, and refuses with `agent_unreachable` if the
node cannot be observed.

## When a node is wedged

`make reboot` works over SSH, so it only reaches a node healthy enough to
answer. If a node pings but SSH, Docker and GPU paths do not respond, nothing in
Tensorstead can reach it. Use the physical power button, then follow the cold
start procedure. Before starting deployments again, check whether one of them
caused the hang. Do not let it restart at boot.

## Coordinator outage

Running deployments keep serving when the coordinator is down. Management is
unavailable until it is back. Operations that were in flight are marked
`outcome_unknown` on restart (SOP-11).
