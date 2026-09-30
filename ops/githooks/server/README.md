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
changes.

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
- **Fails closed.** Missing or version-skewed ruff refuses the push rather than
  waving it through. A gate that silently stops gating is this repository's
  defining failure mode.
- **Cost:** ~0.2s on a Python-heavy push, ~0s otherwise.

## Install

Requires ruff pinned to the version in `uv.lock` (currently **0.16.2**); a
different version on the server would disagree with the developer machine.

```sh
make hooks-install-server      # installs pinned ruff, copies the hook
make hooks-status-server       # what is actually on the box
```

## If you are ever locked out

`pre-receive` cannot be bypassed from the client. Remove it on the server:

```sh
ssh git.internal 'rm /srv/git/tensorstead.git/hooks/pre-receive'
```
