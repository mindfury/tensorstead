# Day-two operations

[first-deployment.md](./first-deployment.md) gets one model serving. This covers
everything after that: changing a deployment, running its lifecycle, dealing
with a host that drifted, and knowing what gets kept when things are removed.

One idea runs through all of it. **The product reports what it finds and changes
only what you asked it to change.** Nothing here reconciles, restarts, retries,
or cleans up on its own. That is occasionally less convenient and always more
predictable — you can leave this system alone for a month and it will not have
done anything you did not ask for.

## Reading command output

The default CLI output is an operator view: short labelled details for one
thing, and compact tables for lists. It is meant to answer the immediate
questions — what is running, where, and whether it is healthy — without making
you read nested data structures.

```bash
stead deployment list
stead node list
stead model list
stead operation list
```

In the deployment table, **RUNNING/CURRENT** is the revision actually running
followed by the newest declared revision. For example, `3/3` is fully current;
`2/3` means revision 3 is declared but revision 2 is still running, normally
because the change requires an explicit restart; and `—/3` means revision 3 is
defined but no revision is running.

For a script, ask Tensorstead for real JSON instead of trying to parse the
human-oriented display:

```bash
stead deployment list --json | python3 -m json.tool
# Equivalent global form, useful in scripts that select several commands:
stead --json deployment list | python3 -m json.tool
```

`--json` is the automation interface. The normal display can evolve to become
clearer for people; scripts should always use JSON.

Opaque IDs in the tables are shortened for readability. You can paste a shown
prefix into `stead model show`, `stead model delete`, or
`stead operation show`; Tensorstead resolves it only when it identifies one
record. It refuses an ambiguous prefix instead of guessing. Models may also be
named by their unique source-model name, such as `nvidia/Qwen3.6-27B-NVFP4`.

## Declared versus observed

Every inspection returns two labelled blocks, never merged:

- **Declared** — what you asked for. The coordinator's authoritative record.
- **Observed** — what the host reports *right now*, fetched at request time.

```bash
stead deployment show llama-70b     # both blocks, plus divergences
stead deployment status llama-70b   # observed only
```

Observed state is never written down as if it were declared state. There is no
cached "last known good" that could be served to you as current — if a node
cannot be reached you get `unreachable`, emphatically, rather than a stale
value that looks fine.

Declared state is always returned in full, even when every node is unreachable.
Losing sight of a host does not lose the record of what you asked for.

## Divergence and reconcile

When observed differs from declared, the difference is **reported and nothing is
done about it**. A container someone stopped by hand, an unexpected instance in
the managed namespace, a node running the wrong revision — all shown, none
touched.

Converging is an explicit act:

```bash
stead deployment reconcile llama-70b
```

Reconcile is the *only* operation that mutates in response to divergence, and
only because you ran it. It reports what it changed and what it could not.

Why this way: an automatic reconciler is indistinguishable from an automatic
reconciler with a bug, and on a shared appliance the blast radius of "it fixed
itself" is somebody else's running workload.

## Lifecycle

```bash
stead deployment start   llama-70b
stead deployment stop    llama-70b
stead deployment restart llama-70b
stead deployment remove  llama-70b
```

**Repeating an operation that is already satisfied is not an error.** Starting a
running deployment exits `0` with `already_in_state` and creates no second
instance. Scripts can be re-run without guarding every call.

**Stop means stop.** It removes the boot-restoration unit as well as the
container, so a reboot does not quietly bring it back. Conversely a deployment
whose desired state is `running` **survives a coordinator outage and a reboot**
— the unit the agent installed is self-sufficient and holds no reference to the
coordinator. You can stop the coordinator entirely and inference keeps serving.

### Across several nodes

A multi-node operation attempts **every** node and then judges the whole:

- all succeeded → the new desired state is recorded;
- some succeeded → `partial_failure`. An overall failure, per-node outcomes in
  the reason, and **desired state is not recorded as changed**. The nodes that
  did change show up as divergence on the next inspection rather than being
  rolled back — an automatic rollback is a second unrequested mutation;
- none succeeded → the ordinary single-node failure shape.

## Modifying a deployment

```bash
stead deployment modify llama-70b --config max_model_len=16384
stead deployment revisions llama-70b
```

Every accepted change **inserts a new numbered revision**. No revision is ever
updated in place and none is ever deleted, so the full history of a deployment
is inspectable forever. The deployment's id is stable across all of them.

A modification **never restarts anything**. If the change needs a restart to
take effect, the response says `restart_required: true` and stops there. You
choose when.

```bash
stead deployment export llama-70b --revision 2 -o rev2.yaml
```

Any revision can be exported, not only the current one.

## What is retained

The retention rules exist because the expensive things and the cheap things get
confused under pressure. Model weights take hours to reacquire; a container
takes seconds to recreate.

| Operation | Removed | Retained |
|---|---|---|
| `deployment stop` | container, boot unit | everything else |
| `deployment remove` | container, boot unit, deployment record | **model artifacts, images** |
| `model delete` | the model and its replicas | — |
| `image delete` | the image on that node | — |

Removing a deployment causes **zero** deletions of acquired models or images.
Deleting those is a separate, explicit act:

```bash
stead model list
stead model delete <model-id>
stead image list
stead image delete <node-id> <digest>
```

Both are **refused while any retained deployment revision still references
them**, and the refusal names the referring deployments. That includes old
revisions — which is the point, since a revision you can no longer recreate is
not much of a record.

Removal against an unreachable node **fails rather than orphaning**. The
alternative is a coordinator that has forgotten about a container still running
on a host you cannot see.

## Credentials

```bash
stead credential set huggingface personal --default   # prompts
stead credential list                                  # names and default only
stead credential delete huggingface org
```

A source may hold several named credentials with exactly one default. An
acquisition that names none uses the default.

Deleting the default **leaves the source with none and says so** — it is never
silently reassigned. Promote a successor deliberately:

```bash
stead credential delete huggingface personal --promote org
```

No command returns a secret value, because no such path exists.

## Inference credentials

The credentials above are for *acquiring* a model from an upstream source. An
inference credential is a different thing: the key a client must present to
the runtime's own endpoint. **It is optional.** Without one, a deployment serves
unauthenticated, and `deployment show` reports `endpoint_authenticated: false`
so that is never a surprise.

When you do use one, the product passes it to the runtime as environment
(`VLLM_API_KEY` for vLLM, `LLAMA_API_KEY` for llama.cpp) so it never appears in
the host-visible process command line, in an export, or in a log. SGLang and
ExLlama have no mechanism this product can use safely, so binding a key to one
of those deployments refuses the start rather than serving unauthenticated
behind your back.

A node may also carry a default key, if the installation enabled one
(`tensorstead_inference_api_key_enabled`). It applies to any deployment with no
binding of its own, **only where the runtime can enforce it**. A runtime that
cannot simply starts unauthenticated, so a node-wide default never stops a
runtime from running.

```bash
stead inferencekey set prod-key --from-env VLLM_API_KEY   # or --from-file, or prompt
stead inferencekey list                                  # names only; no read path
stead inferencekey bind llama-70b --name prod-key
stead inferencekey delete prod-key                        # refused while a deployment binds it
stead inferencekey bind llama-70b                         # omit --name to clear the binding
```

Binding changes a definition, **never host state**: it is recorded on a new
revision and the deployment reports `restart_required`, but nothing is
restarted as an implicit consequence. The runtime keeps serving with the old
key until you explicitly `deployment restart`. The secret is read from a
prompt, `--from-file`, or `--from-env` — never an argv flag — and no command
returns a stored value, because no read path exists.

## History

```bash
stead operation list --deployment llama-70b
stead operation show <operation-id>
```

Every management operation leaves a record with its outcome, and failures carry
a structured reason identifying what failed and on which node.

Operations interrupted by a coordinator restart are resolved to `failed` with
`outcome_unknown`. They are **not** probed or replayed — see
[not-in-v1.md](./not-in-v1.md).

## Idle behaviour

With no request outstanding, the product does nothing at all: no polling, no
telemetry, no background reconciliation. Reachability, resource readings, and
observed state are fetched **when you ask** and at no other time.

If you want a dashboard, poll it from your side. That is a client's business,
and keeping it there is what makes "idle means idle" true here.

## Management transport security

The coordinator serves its management API over TLS using the certificate the
managed private CA already issues to every coordinator and agent host. No new
certificate authority is introduced.

A remote client needs two things:

```sh
export TENSORSTEAD_API=https://spark-alpha.internal:8080
export TENSORSTEAD_CA_BUNDLE=/path/to/ca.pem
```

`TENSORSTEAD_CA_BUNDLE` names the managed CA bundle. It is a public certificate,
not secret material, but it is named by path so the trust anchor stays with the
installation rather than entering the repository.

Address the coordinator by the **DNS name in the certificate**. Hostname
verification compares against the certificate's subject alternative name, so an
IP literal will not match — that is the check working, not a defect.

There is deliberately **no option to skip verification**. A client that accepted
any certificate would leave the management token exactly as exposed as the plain
HTTP this replaces, which would make the setting worse than useless: it would
look like security while providing none.

### Why this matters

Every management operation carries the management token, and all authenticated
clients are equally trusted — there is no per-user permission model. A captured
token is therefore full control of the estate, not a partial disclosure. Before
this change the API was served over plain HTTP on a listener reachable from
other hosts, so the token crossed the network in cleartext on every call.

### Upgrading a running installation

`coordinator serve` still starts without TLS, and warns when it is asked to
listen on an address other than loopback without it. That is deliberate: a new
requirement should not strand an installation mid-upgrade. The warning names
the exposure so an untidy state cannot be mistaken for a deliberate one.

After deploying, update the local smoke profile's `TENSORSTEAD_SMOKE_API` to the
`https://` form. `make readiness` additionally checks that the management port
no longer answers plain HTTP, so a half-applied change is caught rather than
assumed.

## Deploying a release

`make deploy` and `make readiness` need three local paths: your inventory, your
non-secret settings, and your encrypted Vault file. Name them once in a local,
Git-ignored profile rather than passing them on every invocation:

```sh
mkdir -p .tensorstead
cp ansible/deploy.mk.example .tensorstead/deploy.mk
$EDITOR .tensorstead/deploy.mk
```

Then both targets take no arguments:

```sh
make release-local     # gate + a traceable wheel
make deploy            # prompts for Vault and sudo passwords
make readiness         # read-only verification
```

The profile holds **paths only**. The Vault file it names stays encrypted, and
its password and your sudo password are prompted for at deploy time — never
stored in the profile, the repository, or your shell history.

### The artifact comes from the checkout you run in

`make deploy` derives the wheel from the version in *this* checkout's
`pyproject.toml`. Run it from a checkout that does not contain the release you
intend, and it will deploy the older wheel and report success, because that is
exactly what it was asked to do. A clean deploy is not by itself evidence that
the intended release landed — confirm with `stead status`, which reports the
coordinator's version.

## Starting, restarting, and rebooting

### The three control-plane services

`tensorstead-coordinator`, `tensorstead-mcp` and `tensorstead-agent` are systemd units.
`make deploy` enables and starts all three, so a deploy is also a start — but it
is an expensive one: it copies a wheel, reinstalls the venv, regenerates config
and TLS, and refuses outright when the artifact for this checkout's version is
missing. When a machine has come back and you only need to know whether its
services came back with it, use these instead:

```sh
make services-status     # read-only: installed, enabled, running
make services-start      # start now; changes no boot arrangement
make services-restart    # restart in place
make services-stop       # stop now; changes no boot arrangement
```

Starting and *arranging to start at boot* are separate decisions here, and the
targets keep them separate. These four change what is running now and nothing
else; the two below change the boot arrangement and start nothing:

```sh
make services-boot-disable    # will not come back after a reboot
make services-boot-enable     # will
```

They take the same operator profile as `deploy` and `readiness`, and report per
host and per unit. A coordinator-only service does not exist on an agent-only
host, and its absence there is not a failure — the targets skip what a host does
not have rather than failing on it.

**None of this touches inference.** Deployments run as their own Docker
containers under their own units, so restarting the agent leaves a serving model
serving. That is the property that lets a release deploy underneath a live TP=2
group, and it means `services-restart` is safe against a busy estate.

### Rebooting a node

```sh
make reboot TENSORSTEAD_REBOOT_LIMIT=spark-beta.internal
```

One host at a time, and the limit is required rather than defaulted — a
deployment that spans both Sparks as one group makes an unlimited reboot an
outage rather than a rolling restart. Both the target and the playbook refuse
it; `-e tensorstead_reboot_all=true` is the deliberate override.

Before the node goes down, the playbook reports two facts about what will come
back: the node's `TENSORSTEAD_ALLOW_BOOT_RESTORATION` setting, and any deployment
units enabled to start at boot. That is there because what made that failure
unrecoverable was not the driver deadlock — it was that boot restoration
brought the same deployment back on every reboot, so each recovery attempt
re-entered the same state and the node was ultimately reimaged. A reboot is
only an escape hatch if you know what is waiting on the other side of it.

**This is the ordinary reboot, not the emergency one.** Ansible reboots over
SSH, so it reaches only a node healthy enough not to need rebooting. The failure
that makes an operator *want* to force one here is the opposite case: a node
pingable with a usable text console while `sshd`, Docker, the coordinator and
every GPU-touching path have stopped answering. Nothing in this repository
reaches that node. The recovery is the physical power button, and this estate
has done it twice — once for the driver deadlock described above, once when a
hand-run build at `BUILD_JOBS=16` starved `sshd`.

### Cold start, in order

After a reboot or a power cycle:

```sh
make services-status                       # did the units come back?
make services-start                        # if any did not
make readiness                             # coordinator can reach every agent
stead deployment list                   # what is declared versus observed
```

**Nothing comes back on its own.** This estate is deliberately configured so a
reboot brings up neither the control plane nor any model:

- the three services are **not enabled at boot**
  (`tensorstead_services_enabled_at_boot` defaults to false, and `make deploy`
  converges to it, so a deploy will not quietly re-enable them);
- deployments are **not restored at boot** (`TENSORSTEAD_ALLOW_BOOT_RESTORATION`
  defaults to false, and granting it needs the Vault and sudo passwords while
  using it needs only a deployment flag — so an agent can exercise the
  capability and never grant itself the capability).

Worth keeping the two straight, because they guard different things. The wedge
was *deployment* boot restoration: a deployment deadlocked the driver and came
back on every reboot until the node was reimaged. The control plane starting at
boot could not have done that — the product does nothing when idle, so a
coordinator or agent that starts on its own cannot start a model.

The reason to disable the control plane too is narrower: a control plane that is
not running cannot be *told* to start anything either, by a stray automation or
by an operator agent acting on a stale plan. This estate has had a watcher stop
a healthy deployment five seconds after it started. "Nothing comes up until a
human says so" is a posture, and these are its two switches.
