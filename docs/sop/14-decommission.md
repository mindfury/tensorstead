---
title: Decommission Tensorstead deployments, models, images and nodes
sop_id: SOP-14
applies_to: Tensorstead 1.13
keywords: [deployment stop, deployment remove, model delete, image delete, node deregister, retention, still_referenced, image_in_use, free disk space, cleanup]
---

# SOP-14: Decommission Tensorstead deployments, models, images and nodes

## What each removal deletes

| Command | Removes | Keeps |
| --- | --- | --- |
| `deployment stop` | container, boot unit | everything else |
| `deployment remove` | container, boot unit, deployment record and revisions | model artifacts, images |
| `model delete` | the model and all its replicas | — |
| `image delete` | the image on that node | — |
| `node deregister` | the node record | everything on the host |

Model weights can take hours to download again; containers take seconds to
recreate. That is why removing a deployment never deletes its model or image.

## Step 1: Stop the deployment (host-changing)

```sh
stead deployment stop <name>
```

This removes the container and its systemd unit, so a reboot does not bring it
back. Stopping a stopped deployment is not an error.

## Step 2: Remove the deployment record

```sh
stead deployment export <name> -o <name>-final.yaml    # keep a copy
stead deployment remove <name>
```

Removal against an unreachable node fails rather than leaving a container the
coordinator no longer knows about. Bring the node back first.

## Step 3: Delete the model (optional, frees disk)

```sh
stead model list
stead model delete <model>
```

Refused (`still_referenced`) while any remaining deployment, at any revision,
names it. The refusal lists them.

## Step 4: Delete images (optional, frees disk)

```sh
stead image list
stead image delete <node-id> <digest>
stead image reconcile        # clear records of images already gone from nodes
```

Refused while any remaining deployment revision references the image.

## Step 5: Deregister a node (optional)

```sh
stead node deregister <node>
```

Refused while any deployment names the node. This removes only the record. To
retire the host, also stop and disable its agent
(`make services-stop`, `make services-boot-disable`) and remove it from the
Ansible inventory.
