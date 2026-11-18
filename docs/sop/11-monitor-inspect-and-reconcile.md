---
title: Monitor, inspect and reconcile Tensorstead deployments
sop_id: SOP-11
applies_to: Tensorstead 1.13
keywords: [deployment list, deployment show, deployment status, deployment runtime, logs, drift, divergence, reconcile, operation list, operation show, outcome_unknown, health check, JSON output, monitoring, dashboard, admission budgets, busy]
---

# SOP-11: Monitor, inspect and reconcile Tensorstead deployments

## Routine health check (read-only)

```sh
stead status                    # coordinator reachable, versions
stead node list
stead deployment list           # * marks declared/observed drift
stead deployment show <name>    # declared and observed blocks, plus divergences
```

How to read `deployment list`:

- `DESIRED` is the declared state, `HEALTH` the observed state.
- `RUNNING/CURRENT`: `3/3` is current; `2/3` means revision 3 is declared but 2
  still runs (restart needed); `—/3` means nothing runs.
- A node that cannot be reached shows `unreachable`, never a stale value.
  `unknown` and `unreachable` are answers, not errors.

## Why is a deployment not serving?

```sh
stead deployment status <name>          # observed state only; look for inference_ready
stead deployment runtime <name> --tail 500
```

`deployment runtime` reads from the node, at that moment, the runtime's
argument list, restart count and recent output. Nothing it shows is stored.
Missing libraries, out-of-memory and bad flags show up there by name.

## Find and fix drift

Drift means observed differs from declared: a container stopped by hand, an
unexpected container in the managed namespace, a node running the wrong
revision, or nodes left changed by a `partial_failure`. Tensorstead reports
drift and does nothing about it.

To converge one deployment to its declared state (host-changing):

```sh
stead deployment reconcile <name>
```

Reconcile is the only operation that acts on drift, and only when you run it.
It reports what it changed and what it could not.

## Read operation history

```sh
stead operation list --deployment <name>
stead operation show <operation id or unique prefix>
```

Every management action is recorded with inputs and outcome. Failures carry a
structured reason naming what failed and on which node. Operations that were
running when the coordinator restarted show `failed` with `outcome_unknown`:
the coordinator does not know whether they finished. Inspect the deployment and
reconcile if needed. The record does not say which person or agent acted;
everyone shares the management token.

## Monitoring and dashboards

Tensorstead does no background polling and sends no telemetry. To build a
dashboard, poll from your side with JSON output:

```sh
stead --json deployment list
stead --json deployment status <name>
```

Use `--json` in scripts. The human table format may change. ID prefixes shown in
tables can be pasted into `show` and `delete` commands when they are unique.

## Admission limits on the management API

Coordinator, agent and MCP refuse work they cannot bound rather than queueing it:

| Limit | Default | Override | Refusal |
| --- | --- | --- | --- |
| Request body | 16 MiB | `TENSORSTEAD_MAX_REQUEST_BYTES` | 413 `request_too_large` |
| JSON nesting depth | 64 | — | 400 `invalid_request` |
| Concurrent background operations (acquire, build, pull) | 8 | `TENSORSTEAD_MAX_BACKGROUND_OPERATIONS` | 429 `busy`; retry when one finishes |
| Concurrent connections per server | 64 | `TENSORSTEAD_MAX_CONNECTIONS` | 503 |

There is no per-client rate limiting. Put a rate-limiting proxy in front if the
management API is ever reachable by more than its operators. The OpenAPI schema
is at `GET /v1/openapi.json` with the management token.
