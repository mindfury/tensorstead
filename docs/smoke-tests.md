# Release smoke tests

`make test-smoke` is the repeatable, **read-only** acceptance check for an
already-running Tensorstead deployment. It verifies all three supported control
surfaces without creating or changing a deployment:

1. The coordinator API lists the expected nodes, confirms the model has an
   immutable resolved revision rather than `main`, and reports the named
   deployment as running.
2. The MCP server performs the equivalent read-only `node_list` and
   `deployment_list` calls through that same API, both in process and through
   a real local stdio MCP client/server handshake.
3. The **hosted Streamable HTTP MCP endpoint** is checked as a network
   service, which is a stronger claim than the stdio check: it is reached over
   TLS from another host, so its trust and authentication boundaries are all
   that stand between a network caller and full management authority. Four
   properties are asserted, and the negative ones carry most of the weight:
   - TLS verifies against the managed private CA;
   - the endpoint **fails closed against the public trust store**, proving the
     private CA is genuinely required rather than incidentally satisfied;
   - a missing token and a wrong token are both refused;
   - an authenticated `initialize` succeeds and `tools/list` returns exactly
     the supported management operation set. The expected tool names are
     derived from `service/registry.py`, so this is a live parity check
     rather than a hard-coded count — a tool added to the catalogue but absent
     from the deployed endpoint fails here.
4. The runtime rejects an unauthenticated request, advertises the expected
   model with its API key, and returns a short deterministic chat response.

## One-time local profile

Copy the template into the local, Git-ignored directory and set the endpoint
names and **paths** to protected token files:

```sh
mkdir -p .tensorstead
cp tests/smoke/profile.mk.example .tensorstead/smoke.mk
```

Do not put token values in `smoke.mk`. It names files such as the existing
owner-only inference-key file. Store the coordinator management token in a
separate owner-only local file and set `TENSORSTEAD_SMOKE_MGMT_TOKEN_FILE` to that
path.

`TENSORSTEAD_SMOKE_MCP_URL` and `TENSORSTEAD_SMOKE_MCP_CA_FILE` name the hosted MCP
endpoint and the managed private CA bundle that signs it. The CA bundle is a
public certificate rather than a secret, but it is named by path so the trust
anchor stays with the operator's installation instead of entering the
repository. Both are required: an unconfigured hosted endpoint would otherwise
be a silently skipped check, and a security boundary nobody exercises is one
nobody is holding.

Then run:

```sh
make test-smoke
```

The target deliberately fails before contacting anything when the profile has
not been configured. It has no lifecycle operation and will not acquire a
model, pull an image, create a deployment, stop a service, or alter any
host. It does make one ordinary inference request to the existing runtime.

## Separate hardware lifecycle tests

`make test-hardware` remains a different, explicit tier. It may create and
remove uniquely named `tensorstead-test-*` resources and is for planned lifecycle
validation only. See [the hardware safety rules](../tests/hardware/README.md).
