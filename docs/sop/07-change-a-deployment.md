---
title: Change, restart, rename or revert a Tensorstead deployment
sop_id: SOP-07
applies_to: Tensorstead 1.13
keywords: [deployment modify, --config, --replace-config, --expect-revision, restart_required, deployment restart, revisions, export, from-export, rename, revert, rollback, image change, restore-on-boot]
---

# SOP-07: Change, restart, rename or revert a Tensorstead deployment

## Purpose of this procedure

Change a deployment's runtime config, image, nodes or endpoint; apply the
change with a restart; inspect and export revisions; go back to an earlier
definition; and rename a deployment. To change **which model** a deployment
serves, use SOP-08 instead.

## Step 1: Record the current definition

```sh
stead deployment show <name>
stead deployment revisions <name>
stead deployment export <name> -o <name>-before.yaml
```

## Step 2: Modify the definition (no host change)

```sh
stead deployment modify <name> --expect-revision <current N> \
  --config max_model_len=16384
```

- `--config` **merges** by key: only named keys change.
- `--replace-config` replaces the whole runtime config and drops every key you
  do not name. It cannot be combined with `--config`.
- The merge is per top-level key. A nested value such as `extra_args` is
  replaced whole, so repeat any entries that should stay.
- `--image`, `--node` and `--endpoint` change those fields.
- `--expect-revision N` refuses the change if someone moved the deployment past
  revision N.
- Each accepted change inserts revision N+1. The response says whether a
  restart is required. **Nothing restarts.**
- Changing `--model` on a running deployment is refused with
  `model_change_refused`. See SOP-08.

## Step 3: Check the new revision before applying it

```sh
stead deployment show <name>     # declared block now shows the new revision
stead deployment list            # RUNNING/CURRENT shows e.g. 2/3
```

Confirm every setting you expect is present. This is a free window: no host has
changed yet.

## Step 4: Apply the change with a restart (host-changing)

```sh
stead deployment restart <name>
stead deployment show <name>     # wait for inference_ready: true
```

Restart is stop then start, so the endpoint is down meanwhile. If another
running deployment serves the same `served_model_name`, the restart is refused
(`served_name_conflict`) before anything is stopped.

## Revert to an earlier revision

There is no rollback command. Going back is a forward modification that
restores the old values and becomes a new revision:

```sh
stead deployment export <name> --revision 2 -o rev2.yaml
# read rev2.yaml, then apply its values:
stead deployment modify <name> --expect-revision <current N> \
  --image <rev2 image> --replace-config '<rev2 runtime_config as JSON>'
stead deployment restart <name>
```

If revision 2 referenced a model or image that has since been deleted, the
revert fails visibly instead of producing something that only looks like
revision 2.

## Recreate a deployment from an export

```sh
stead deployment create --from-export rev2.yaml
```

Creates a new deployment from the exported definition without touching hosts.
Change the name or endpoint first if the original still exists.

## Rename a deployment

```sh
stead deployment rename <old-name> <new-name>
```

The id, revisions and serving state are unchanged and nothing restarts. A name
another deployment holds is refused.

## Change an image to repair a broken runtime

1. Build or pull the fixed image on every node the deployment uses (SOP-05).
2. `stead deployment modify <name> --image <new reference>`.
3. If the deployment uses `trust_remote_code` or `allow_io_uring`, create a new
   code approval for the new image digest (SOP-10).
4. `stead deployment restart <name>` and confirm the outcome you wanted
   (for example that tool calls are parsed), not only that the old error went
   away.
