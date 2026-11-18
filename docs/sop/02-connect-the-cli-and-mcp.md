---
title: Connect the stead CLI and MCP clients to a coordinator
sop_id: SOP-02
applies_to: Tensorstead 1.13
keywords: [stead login, stead status, config.yml, CA bundle, token file, TENSORSTEAD_API, TENSORSTEAD_CA_BUNDLE, TENSORSTEAD_MGMT_TOKEN, MCP, certificate verify failed, coordinator_unreachable]
---

# SOP-02: Connect the stead CLI and MCP clients to a coordinator

## Purpose of this procedure

Configure the `stead` CLI, a local stdio MCP server, or a remote MCP client to
talk to a Tensorstead coordinator over verified TLS.

## What you need from the installation

- Coordinator URL using the **DNS name in its certificate**, for example
  `https://spark-alpha.internal:8080`. An IP address fails hostname
  verification by design.
- CA bundle path. With managed TLS this is
  `~/.local/share/tensorstead/ansible-tls/ca.pem` on the Ansible controller.
- A file containing the management token, readable only by you (mode 0600).

## Procedure: configure the stead CLI

```sh
stead login     # asks for URL, CA bundle and token file, verifies, then saves
stead status    # shows target, client and coordinator versions, contract
                # version, reachability, and where each setting came from
```

`stead login` writes `~/.config/tensorstead/config.yml` (mode 0600). It stores
the **path** of the token file, never the token:

```yaml
ca_bundle: /home/you/.local/share/tensorstead/ansible-tls/ca.pem
coordinator: https://spark-alpha.internal:8080
token_file: /home/you/.local/share/tensorstead/management-token
```

For a local evaluation coordinator (`stead coordinator serve
--insecure-dev-mode`, loopback only), accept the loopback URL and leave the
token-file answer blank.

## How the CLI chooses settings

The first source that supplies a value wins:

1. Flags `--api` and `--ca-bundle` (one command only).
2. Environment: `TENSORSTEAD_API`, `TENSORSTEAD_CA_BUNDLE`,
   `TENSORSTEAD_MGMT_TOKEN`.
3. `~/.config/tensorstead/config.yml` (`TENSORSTEAD_CONFIG` moves it;
   `XDG_CONFIG_HOME` is honoured).
4. Built-in default `http://127.0.0.1:8080`.

There is no option to skip certificate verification.

## Procedure: run a local stdio MCP server

```sh
export TENSORSTEAD_API=https://spark-alpha.internal:8080
export TENSORSTEAD_CA_BUNDLE=/path/to/ca.pem
export TENSORSTEAD_MGMT_TOKEN="$(< ~/.local/share/tensorstead/management-token)"
tensorstead-mcp
```

If `TENSORSTEAD_CA_BUNDLE` is missing, every tool returns
`coordinator_unreachable` and list tools come back **empty**. An empty list from
MCP can therefore mean "could not verify the coordinator", not "nothing exists".

## Procedure: connect a remote MCP client to the hosted endpoint

- URL: `https://<coordinator DNS name>:8090/mcp`.
- The client must trust the Tensorstead private CA.
- Send `Authorization: Bearer <management token>` on every request, from the
  client's secret store.
- The client must reach TCP 8090, usually over the private network or VPN.

The hosted endpoint manages models and deployments. It is not an inference
endpoint and has no shell or SSH tool.

## Troubleshooting connection errors

| Message | Cause | Fix |
| --- | --- | --- |
| `not a Tensorstead coordinator: ... (answered by llama.cpp)` | Something else is on the port. 8080 is also llama.cpp's default. | Run `stead login` with the right URL |
| `could not reach coordinator at ...` | Coordinator down, port closed, DNS failure, or off the network | Check `make services-status` and DNS |
| `certificate verify failed: unable to get local issuer certificate` | CA bundle unset or wrong | Set the managed CA bundle |
| `certificate is not valid for ...` | Addressed by IP or a name not in the certificate | Use the DNS name in the certificate |
| `HTTP 401` / `authorization_refused` | Token wrong or stale | `stead status` names the token file it read |
| Empty list | A real empty answer, or the wrong coordinator | Check `stead status` names the coordinator you meant |

## Checking versions

`stead --version` prints the client version without contacting anything.
`stead status` adds the coordinator's version and the wire contract version.
Use it after every upgrade to confirm the intended release landed.
