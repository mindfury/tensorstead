---
title: Upgrade the Tensorstead software on the estate
sop_id: SOP-13
applies_to: Tensorstead 1.13
keywords: [upgrade, release, make release-local, make deploy, deploy-release, verify release, provenance, rollback build, make builds, TENSORSTEAD_DEPLOY_ARTIFACT, wheelhouse, dependency upgrade, uv.lock, smoke test, make test-smoke, release candidate, agent_version_incompatible]
---

# SOP-13: Upgrade the Tensorstead software on the estate

## Purpose of this procedure

Install a new Tensorstead release on the coordinator and agents, verify it, and
roll back to an earlier build if needed. This upgrades Tensorstead itself, not
a model or runtime image. Running deployments keep serving throughout.

## Option A: deploy a build from this checkout

1. From a clean, reviewed checkout of the intended version:

   ```sh
   make release-local    # all checks, then a tracked wheel
   ```

   Record the release version, build number, SHA-256, Git revision and
   `source_tree_dirty` (should be `false`) from the manifest in `dist/`.
2. Get the operator's approval, then (host-changing):

   ```sh
   make deploy           # prompts for Vault and sudo passwords
   make readiness        # read-only
   ```

3. Confirm the version landed: `stead status`. `make deploy` installs the wheel
   for the version in **this checkout's** `pyproject.toml`, so a deploy from an
   old checkout succeeds while installing the old release.

## Option B: deploy a published release

```sh
make deploy-release VERSION=<x.y.z>
```

It downloads the release and refuses unless every asset has signed build
provenance from this repository's release workflow at the matching tag and
matches the attested `SHA256SUMS`. It needs `gh` to be authenticated.

## Verify the upgrade

```sh
stead status          # coordinator version and contract version
make readiness
make test-smoke       # read-only acceptance check (one-time setup below)
```

`make test-smoke` checks the coordinator API, local and hosted MCP (including
that the hosted endpoint rejects the public trust store, a missing token and a
wrong token), and makes one inference request to an existing deployment. It
creates and changes nothing. One-time setup: copy
`tests/smoke/profile.mk.example` to `.tensorstead/smoke.mk` and set endpoint
names and **paths** to token files, including `TENSORSTEAD_SMOKE_MCP_URL` and
`TENSORSTEAD_SMOKE_MCP_CA_FILE`.

If the coordinator and agents report `agent_version_incompatible`, one side was
not upgraded. Re-run `make deploy` for the hosts that were missed.

## Roll back to an earlier Tensorstead build

```sh
make builds           # archived builds and their Git revisions
make deploy TENSORSTEAD_DEPLOY_ARTIFACT=dist/builds/<n>/tensorstead-<version>-py3-none-any.whl
```

If the earlier build used a different `uv.lock`, also set
`TENSORSTEAD_DEPLOY_WHEELHOUSE` to that build's wheelhouse.

## Upgrade a dependency

Appliances install exactly what `uv.lock` pins, offline and hash-checked.

1. `uv lock --upgrade-package <name>` (or merge a Dependabot PR). Review the
   lock diff.
2. `make check` must pass.
3. `make build` rebuilds the wheelhouse. It refuses any pin without a hash or
   without a `linux-aarch64-cp312` wheel.
4. Deploy as in Option A.

## Release-candidate acceptance

A release candidate is accepted only after both gates are recorded:

- Controller gate: `make release-local` passes from a clean commit.
- Live gate (operator approval required): `make deploy`, `make readiness`,
  `make test-smoke`, confirm any new model's recorded revision is a commit and
  not `main`, and optionally `make test-hardware` on test-owned resources only.
