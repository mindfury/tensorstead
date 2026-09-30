# Hosted MCP endpoint

Ansible installs a separate, enabled `tensorstead-mcp` service beside every
coordinator. It exposes the same management tools as local stdio MCP at:

```text
https://<coordinator DNS name>:8090/mcp
```

For example, with a coordinator at `spark-alpha.internal` the URL is
`https://spark-alpha.internal:8090/mcp`.

An MCP client must trust the Tensorstead private CA and send
`Authorization: Bearer <management token>` with each request. Keep the token
in the client’s secret store; never put it in Git, chat, or a shared config.
The client must be able to reach the Spark DNS name and TCP port 8090 (usually
through the same private network or VPN). This endpoint manages models and
deployments; it is not an inference endpoint and it has no shell or SSH tool.

An agent should inspect nodes, runtimes, models, and deployments; acquire an
immutable model revision; poll the returned operation; create a deployment with
an unused endpoint; start it; then verify declared and observed state.

## Local stdio MCP against a TLS coordinator

The hosted endpoint above is one way to reach the tools; running the MCP server
locally over stdio is the other. That server is an ordinary HTTP client of the
coordinator, so once the coordinator serves TLS it needs the managed
private CA just as any other client does:

```sh
export TENSORSTEAD_API=https://spark-alpha.internal:8080
export TENSORSTEAD_CA_BUNDLE=/path/to/ca.pem
export TENSORSTEAD_MGMT_TOKEN=...   # from a protected local file, not a literal
tensorstead-mcp
```

Without `TENSORSTEAD_CA_BUNDLE` the client fails verification and every tool
returns `coordinator_unreachable`. Worth recognising, because a list tool then
comes back empty, which reads as "the coordinator knows of no nodes" rather
than "this client could not verify the coordinator". The failure is in the
client's trust configuration, not in the appliance.


## Repairing a runtime image from an agent

An agent that determines a runtime image is broken can fix it through MCP
rather than asking someone to SSH:

| Tool | Purpose |
| --- | --- |
| `buildspec_set` | Record a build spec — base image plus ordered steps. Executes nothing |
| `buildspec_list` | List specs, including whether each base is pinned |
| `buildspec_delete` | Delete a spec; refused while an image it produced is referenced |
| `image_build` | Build a recorded spec on a node |
| `image_import` | Import a prebuilt archive by name from the node's managed image store |

These exist because the alternative was SSH. When NVIDIA's vLLM image shipped
an `xgrammar` too old for its own tool-calling path, no product operation could
repair it — so the repair happened outside the product, unrecorded, which is
what Tensorstead exists to prevent.

Constraints an agent should expect:

- A build spec is **data**: ordered steps, no host command, no interactive
  build, and no tool that runs something on a host directly.
- `image_import` takes `archive_name`, a file name inside the node's managed
  store. No MCP tool accepts a path.
- An unpinned base is reported as unreproducible and never refused.
- A build returns `image_id`, not `digest`: a locally produced image has no
  registry digest.
