# Release-candidate checklist

This checklist separates controller-only evidence from live-appliance evidence.
It prevents a green local test suite from being mistaken for a deployment
approval.

> `make deploy` and `make readiness` read `.tensorstead/deploy.mk` and take no
> arguments. If you have not created it yet, the refusal names the command that
> does. See [operations](operations.md#deploying-a-release).

## Controller-only gate

Run this from a clean, reviewed worktree:

```sh
make release-local
```

It runs linting, type checking, every non-hardware test, Ansible syntax/lint
validation, and creates a tracked wheel. Record its release version, build
number, SHA-256, Git revision, and `source_tree_dirty` value from the generated
manifest. A release candidate should have `source_tree_dirty: false`.

## Live-appliance gate — explicit operator approval required

These steps may contact or change appliances and are intentionally not part of
`make release-local`:

1. Deploy the selected wheel with the reviewed Ansible profile. The `make`
   targets below explicitly override any stale wheel path in local settings.
2. Run Ansible readiness validation.
3. Configure the ignored smoke profile and run `make test-smoke`.
4. For a new model acquisition, verify the recorded revision is immutable, not
   `main`.
5. Run the separately guarded hardware lifecycle suite only on approved,
   test-owned resources.

Do not mark a release candidate accepted until both gates have recorded their
results. See [build tracking](builds.md) and [release smoke tests](smoke-tests.md).

Historical live-test evidence belongs in a dated, credential-free validation
record kept with the release evidence.

### Deploy and verify a reviewed build

After `make release-local` succeeds from a clean commit, deploy. Both targets
read `.tensorstead/deploy.mk` for your inventory, settings, and Vault paths, and
prompt for the Vault and sudo passwords; neither value is stored anywhere.

```sh
make deploy

make readiness
```

`deploy` is host-changing: it installs the selected wheel and reconciles the
Tensorstead services. `readiness` is read-only and verifies prerequisites,
services, TLS, and coordinator-to-agent reachability.
