# Server-side gate (tier 2)

`pre-receive` for a self-hosted bare repository (for example
`git.internal:/srv/git/tensorstead.git`). Not needed when hosting on GitHub.

## Why this exists when tier 1 already does

The hooks one directory up gate *this* clone and can be waved away with
`--no-verify`. That is a gate on a cooperative developer. It is not a gate on an
automated agent having a bad night, and pushes are increasingly delegated to
agents. `pre-receive` runs on the server: `--no-verify` is a client-side flag and
does nothing to it.

## What it checks, and what it deliberately does not

**Does:** `ruff format --check` and `ruff check` on the Python files a push
changes, **ref by ref**: each pushed ref is diffed against its own old value and
checked against its own tree, so a push of several refs cannot have one ref's
violation judged against another ref's tree (CodeGuard CG-13). File lists are
NUL-delimited end to end, so names with spaces or newlines are checked whole.

**Does not:** tests, mypy, coverage. Those need the uv-managed venv, which would
mean installing *and then tracking* this project's dependencies on a box whose
job is serving git — minutes on first run,
and a second installation free to drift from the first. The marginal catch over
the local `pre-push` hook is small for a single-developer repository. Tests stay
on the developer machine.

Ruff alone is one static binary, no venv, and catches the class that actually
broke `main`: a file that was never formatted.

## Properties worth knowing

- **Path-gated.** A push carrying no Python exits before doing any work, so a
  documentation-only push is never delayed.
- **Changed files only.** A push is never refused for a pre-existing problem in
  a file it did not touch.
- **Full tree for context.** The whole pushed tree is materialised even though
  only changed files are gated, because ruff's isort rules need the real layout
  to tell first-party from third-party. Checking files in isolation produced
  false `I001` failures on any push touching tests but not `src/` — found by
  testing against real commits, not reasoned about.
- **Fails closed.** Missing or version-skewed ruff, a diff or archive that
  cannot be produced, a ruff crash, or a ref that exceeds its time budget
  (`TENSORSTEAD_HOOK_REF_SECONDS`, default 120) refuses the push rather than
  waving it through. A gate that silently stops gating is this repository's
  defining failure mode.
- **The version comes from the pushed ref.** The expected ruff version is read
  from that ref's own `uv.lock`, so a dependency bump never needs the hook
  edited (a hard-coded pin had already drifted to 0.16.2 against a 0.17.0 lock).
- **Cost:** ~0.2s on a Python-heavy push, ~0s otherwise.

## Install

Requires bash and ruff at `/usr/local/bin/ruff` (override with
`TENSORSTEAD_HOOK_RUFF`) at the version in the pushed `uv.lock`; a different
version refuses the push. After a ruff bump lands, re-run
`make hooks-install-server` to install the new version.

```sh
make hooks-install-server      # installs pinned ruff, copies the hook
make hooks-status-server       # what is actually on the box
```

## If you are ever locked out

`pre-receive` cannot be bypassed from the client. Remove it on the server:

```sh
ssh git.internal 'rm /srv/git/tensorstead.git/hooks/pre-receive'
```
