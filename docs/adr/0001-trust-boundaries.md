# ADR 0001: What approvals, registration and the audit trail do and do not guarantee

- **Status:** accepted, 2026-10-17
- **Context:** an independent Project CodeGuard review asked for three design
  positions to be written down explicitly rather than implied by wording. None
  is an exploitable finding within Tensorstead's stated model: a single-tenant
  management plane in which every holder of the management token is a fully
  trusted administrator.

This ADR records each position, the risk it accepts, and what would have to
change if the threat model changes.

## 1. Code-execution approvals are operator attestations, not an independent human authority

**What they are.** A code approval is a separate, immutable record that binds
one code-loading or isolation-relaxing option to one exact model revision and
one image digest. The tuple and digest checks are real consistency controls.
A deployment cannot quietly gain `trust_remote_code` by being edited, and an
approval stops matching when its model or image changes.

**What they are not.** They are not a second human's sign-off. The same
management identity, whether a person at the CLI or an agent over MCP, can
create an approval and then deploy with it. `approved_by` is free text supplied
by the caller. It is recorded as stated and is not authenticated.

**Decision.** We describe approvals as *operator attestations*: a deliberate,
separately recorded statement that someone reviewed this tuple. We do not call
them enforced human-in-the-loop authorization. The MCP tool text and the
README say so.

**If genuine human-only sign-off is required:** approvals must be issued by an
identity the deploying caller does not hold. That means one of:

- a separately authenticated approver credential that the MCP server is never
  given;
- signed, human-issued grants verified by the coordinator.

`approved_by` would then come from the authenticated principal, not the
request body.

## 2. Node registration trusts the destination the administrator names

**What happens.** Registration validates the agent URL's syntax and requires
HTTPS. It does not check the URL against an inventory. The registration probe
authenticates with the default fleet management token, unless the node has its
own token reference. With public-CA trust, a host the administrator was misled
into registering, and which holds a valid public certificate, could receive
that token.

**Decision.** Registration remains a fully trusted administrative act, like
editing the Ansible inventory. The deployment uses a private CA, which narrows
this risk to hosts that CA has issued certificates for. We accept the residual
risk and record it here rather than treating it as privilege escalation.

**Mitigations available today:**

- Set `TENSORSTEAD_AGENT_CA_BUNDLE` on the coordinator to the private CA bundle.
  When it is unset, agent connections fall back to the system's public CAs,
  which is what makes a publicly certified host a viable destination at all.
  The Ansible roles already set it (`tensorstead_agent_ca_bundle`).
- Rotate each node onto its own token
  (`POST /v1/nodes/{id}/rotate-management-token`). Later calls then present
  only that node's token. The registration probe itself still carries the
  fleet token, which is the remaining gap.

**If the model changes** (for example, registration delegated to less-trusted
callers):

- Add an allowlist of approved destinations.
- Make per-node credentials mandatory.
- Require the agent to prove its identity, for example with a pinned
  certificate or an enrollment challenge, before any reusable secret is sent.

## 3. The audit trail records what happened, not who did it

**What exists.**

- Every operation is recorded with its inputs and outcome, so agents and humans
  leave the same trail.
- Deployments are immutable revisions.
- Approvals are immutable once written.

**What does not.**

- *No per-person identity.* Authentication is a shared bearer token, so the
  record cannot say which human or which agent acted. Caller-written
  `approved_by` is a claim, not an identity.
- *No revocation history.* Revoking an approval deletes its row, so there is no
  durable record that it existed and was withdrawn.
- *Retention is not guaranteed.* Records live in the coordinator's SQLite
  database. The administrator can rewrite them, and they are retained only as
  long as that database is.

**Decision.** For a single-operator estate this is sufficient, and the
documentation must not claim more. "Recorded like anything a human does" means
the same operation records, not attributed accountability.

**If attribution or compliance is required:**

- Per-principal credentials, with the authenticated principal stamped on every
  record.
- Revocation recorded as an append-only event, not a delete.
- Export to append-only storage the coordinator cannot rewrite.

Whatever is added, credential values and raw originating prompts must never be
logged.
