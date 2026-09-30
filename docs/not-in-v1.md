# Not in v1

Four capabilities that a reader will look for, not find, and reasonably wonder
about. Each was considered and left out on purpose. This page exists so their
absence reads as a decision rather than an oversight — and so that whoever
implements one later knows what the original reasoning was and can disagree with
it deliberately.

None of these is a placeholder for something half-built. In each case the
product does something simpler and says so.

---

## Rollback

**What is missing:** there is no `deployment rollback` that returns a deployment
to an earlier revision.

**What exists instead:** every revision is retained forever and every one is
exportable. Going back to revision 2 is a modification that happens to restore
revision 2's values — and it becomes revision 5, not a rewind.

**Why:** a rollback command reads as "undo", and undo is a promise this system
cannot keep. The revision records a *definition*, not the world. If revision 2
referenced a model you have since deleted, or an image tag that now resolves to
different bytes, "rolling back" would produce something that is not revision 2
while telling you it was. Making the operator perform a modification keeps the
forward-only revision history honest, and keeps the failure visible if the old
definition is no longer reachable.

**If you add it later:** it should be a convenience that constructs a
modification from a retained revision, and it must still insert revision *n+1*.
A rollback that mutates history would break the guarantee that makes exports
trustworthy.

---

## Operation cancellation

**What is missing:** a running operation cannot be cancelled. There is no
`operation cancel`.

**What exists instead:** operations report progress and reach a terminal state.
A long acquisition runs to completion or fails.

**Why:** cancellation is only meaningful if the thing being cancelled can be
stopped *cleanly*, and most of what takes a long time here happens on the other
side of an agent call — a multi-gigabyte download inside `huggingface_hub`, an
image pull inside the container engine. A cancel that abandons the coordinator's
record while the host keeps working is worse than no cancel: you would have a
system that believes nothing is happening while something is.

Honest consequence: an acquisition of the wrong 140GB model will finish. You can
then delete it.

**If you add it later:** it needs cooperative cancellation *through* the agent
contract, with the agent confirming it stopped and cleaned up staging. The
coordinator marking an operation cancelled on its own would be a lie.

---

## Runtime-config redaction

**What is missing:** values in `runtime_config` are not scanned, masked, or
redacted anywhere — not in exports, not in the API, not in the CLI.

**What exists instead:** runtime configuration is recorded and displayed exactly
as given. Secret material for a model source goes through the credential
mechanism, which stores a reference and never a value.

**Why:** this is the honest version of a guarantee that is often faked.
Redaction requires guessing which values are sensitive, and a guess that is
right most of the time produces exactly the wrong belief — an operator who
thinks the export is safe because it *usually* masks things. The product instead
draws a clear line: **runtime config is not a secret-bearing surface**. What we
do guarantee is structural — a secret supplied through the credential mechanism
has no field in any export or record to leak from, so the structural guarantee
holds without a redaction step existing at all.

**The limitation, stated plainly:** if you put an API key in `--config`, it will
appear in the export. Do not do that.

**If you add it later:** resist pattern-matching. A declared per-runtime schema
marking specific fields as secret-bearing would be defensible; a regex for
things that look like tokens would not.

---

## Automatic recovery of interrupted operations

**What is missing:** operations that were in flight when the coordinator stopped
are not probed, resumed, or retried on restart.

**What exists instead:** on startup, any non-terminal operation is resolved to
`failed` with `outcome_unknown`. The record says exactly what is true — the
coordinator does not know whether it completed.

**Why:** the coordinator cannot distinguish "the agent never received it" from
"the agent completed it and the reply was lost". Guessing produces one of two
bad outcomes: a retry that duplicates a completed side effect, or a record
claiming failure for something that succeeded. `outcome_unknown` is less
satisfying and more accurate.

Note that this affects the *record*, not the world: a deployment that was
running before the restart is still running afterwards, because the boot unit
the agent installed is self-sufficient and holds no reference to the
coordinator.

**What to do:** inspect the deployment. Declared and observed are both
available, and the difference tells you what actually happened. Reconcile if you
want to converge.

**If you add it later:** it requires idempotency keys on the agent contract so a
replay is provably safe. Without those, recovery is guessing with extra steps.

---

## Also deliberately absent

Excluded by design rather than deferred, these are not roadmap items — they are
the things a management tool accretes when nobody is holding the line on scope:
RBAC, SSO, high availability, replication, consensus, message brokers,
scheduling, autoscaling, and policy engines.

Two are worth calling out because their absence is load-bearing:

- **No scheduler.** The product never chooses a node for you. Placement is
  something the operator states, because on a two-appliance estate the operator
  knows things the scheduler never will.
- **No capacity gating.** Resource readings are reported for your judgement and
  **never block an operation**. A tool that refuses to deploy because it thinks
  memory is short is a tool that is wrong at the worst moment.
