# Connecting the CLI to a coordinator

Read this before running any `stead` command against an appliance.

## The short version

```sh
stead login
```

It asks for the coordinator URL, the CA bundle, and (when your coordinator
uses one) the path to your token file, **verifies they work**, and writes
`~/.config/tensorstead/config.yml`. After that every command works in any shell,
with nothing exported. For a local coordinator running without a management
token, leave the token-file answer blank.

To see what the CLI is using and where each value came from:

```sh
stead status
```

That output names the source of every setting — environment variable, config
file, or built-in default — and prints the config file path *even when the file
does not exist*. You should never have to hunt for where a value came from.

## Where settings come from

Resolved in order; the first source that supplies a value wins:

| Order | Source | Use |
| --- | --- | --- |
| 1 | `--api`, `--ca-bundle` | one command, overriding everything below |
| 2 | `TENSORSTEAD_API`, `TENSORSTEAD_CA_BUNDLE`, `TENSORSTEAD_MGMT_TOKEN` | CI, scripts, temporary overrides |
| 3 | `~/.config/tensorstead/config.yml` | the normal case |
| 4 | built-in default (`http://127.0.0.1:8080`) | a colocated coordinator |

Environment variables still work, so nothing scripted breaks. They are simply
no longer the only mechanism.

`TENSORSTEAD_CONFIG` relocates the config file; `XDG_CONFIG_HOME` is honoured.

A one-off override, without touching your config:

```sh
stead --api https://other-host:8080 status
```

`status` reports that as coming *from a command-line flag*, not from your
config. The source it names is always the one you actually used.

## Answering "what am I running?"

```sh
stead --version    # client version; contacts no coordinator
stead status       # target, both versions, contract version, reachability
```

`--version` needs nothing configured, which is the point: it answers the first
question before you have set anything up. `status` additionally reports the
coordinator's version, so a deploy that reported success but shipped an older
release is visible rather than assumed.

## What the config file holds

```yaml
ca_bundle: /home/you/.local/share/tensorstead/ansible-tls/ca.pem
coordinator: https://spark-alpha.internal:8080
token_file: /home/you/.local/share/tensorstead/management-token
```

`token_file` is absent when the coordinator does not require a token (normally
only a local, loopback-only coordinator). It names a **file** holding the token
rather than the token itself, and the config file is written mode `0600`.

You can show someone your configuration without redacting it (secrets stay out
of ordinary configuration).

Address the coordinator by the **DNS name in its certificate**. Hostname
verification compares against the certificate's subject alternative name, so an
IP literal will not match — that is the check working.

There is deliberately **no option to skip verification**. A client that
accepted any certificate would leave your management token as exposed as the
plain HTTP that TLS replaced.

## When it does not work

Run `stead status` first. It answers most of these by naming the source.

### `not a Tensorstead coordinator: ... (answered by llama.cpp)`

Something else is listening. The built-in default is port 8080, which is also
llama.cpp's default — one of this product's own supported runtimes. Run
`stead login`.

### `could not reach coordinator at ...`

Nothing answered: the coordinator is down, the port is closed, the DNS name
does not resolve, or you are off the network.

### `certificate verify failed: unable to get local issuer certificate`

The CA bundle is unset or wrong. The coordinator's certificate is privately
signed, so the public trust store cannot validate it.

### `certificate is not valid for ...`

You addressed the coordinator by IP, or by a name not in its certificate.

### `HTTP 401` / `authorization_refused`

The token is wrong or stale. `stead status` names the file it read.

### Commands work but a list is empty

An empty list is a real answer. Check `stead status` names the coordinator
you meant.

## Running the MCP server locally

The stdio MCP server is another HTTP client of the coordinator and needs the
same trust anchor. Against a TLS coordinator without `TENSORSTEAD_CA_BUNDLE` it
fails verification and every tool returns `coordinator_unreachable` — which
surfaces as an *empty* list rather than an error, so it reads as "the
coordinator knows of nothing" when it means "this client could not verify the
coordinator". See [hosted MCP](mcp.md).

## Related

- [First deployment](first-deployment.md) — the end-to-end path for a new estate
- [Operations](operations.md) — day-two lifecycle, reconcile, retention, TLS
- [Hosted MCP](mcp.md) — the network MCP endpoint for agents
- [Release smoke tests](smoke-tests.md) — read-only verification of a live estate
