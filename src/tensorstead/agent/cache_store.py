"""Per-deployment durable cache — the agent owns the host side.

A runtime that compiles on first use — vLLM does, via Triton and
``torch.compile`` — throws that work away every time its container is replaced.
Tensorstead replaces the container on **every** restart and every revision change
(`lifecycle._start_on_nodes` creates a new one after stop removed the old), so
the compile cost is paid again and again rather than once. Observed on the
appliance on 2026-08-11: four Triton kernels compiling on live traffic, minutes
after a restart, on a deployment already started several times that day.

The fix is somewhere durable — but **not** an operator-named volume. The design
forbids `volumes` and `binds` in `host_config` because a bind can point anywhere
and a deployment definition that mounts an arbitrary host path is no longer a
true account of what is running. That rule stands. What changes is who chooses
the location: the adapter states *what must be true* ("durable storage visible
at this in-container path"), and this module decides where it actually lives.

That makes the cache exactly as accountable as the model store: a managed
location under a known root, reported by `node resources`, and removable.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import Any

# The fifth managed storage root, beside models, images, state, and
# credentials. Overridable for installs that put managed storage elsewhere.
DEFAULT_CACHE_ROOT = "/var/lib/tensorstead/cache"


def cache_root() -> Path:
    """The managed cache root for this node."""
    return Path(os.environ.get("TENSORSTEAD_CACHE_PATH", DEFAULT_CACHE_ROOT))


def cache_dir_for(deployment_id: str, *, root: Path | None = None) -> Path:
    """The host directory backing one deployment's durable cache.

    Keyed by deployment rather than shared node-wide. A shared cache would let
    two deployments reuse each other's compiled kernels — these caches are
    content-addressed internally, so it would be safe, and it would use less
    disk.

    Per-deployment wins for v1 on accountability: ``node resources`` reports
    managed storage by purpose, and with one shared directory nobody could say
    whose bytes those are or reclaim them with the deployment they belong to.
    Recorded rather than assumed — the shared variant is the thing to revisit
    under disk pressure.

    Raises ``ValueError`` if the result would not be a direct child of the
    cache root. The route boundary already rejects a non-ULID
    ``deployment_id`` before this is ever called; this is
    the second, independent check -- ``Path`` join silently discards the left
    side for an absolute right side, so an unvalidated caller could otherwise
    make this function return an arbitrary host path, which ``provision``
    below then creates, chmods, and possibly chowns.
    """
    resolved_root = (root or cache_root()).resolve()
    candidate = (resolved_root / deployment_id).resolve()
    if candidate.parent != resolved_root:
        raise ValueError(f"deployment_id {deployment_id!r} does not resolve beneath the cache root")
    return candidate


def provision(
    deployment_id: str, *, root: Path | None = None, owner_uid: int | None = None
) -> str | None:
    """Create the deployment's cache directory. Returns the host path, or None.

    Returns ``None`` when the directory cannot be created, and the caller then
    starts the container without a cache. That is the right failure: a runtime
    with no durable cache recompiles, which is slow, whereas refusing to start
    a deployment because a *performance* directory was unavailable would turn
    an optimisation into an outage.

    **Permissions are the part worth care, and not for the obvious reason.**
    The agent runs as root and the runtime container may not, so the directory
    has to be writable by whoever the image runs as. The tempting answer is
    0o777. It is wrong: a compile cache holds *executable* artefacts — Triton
    cubins, compiled shared objects — which the runtime loads and runs. A
    world-writable directory under the product's own storage root is therefore
    a path from "any local account on this host" to "code executing inside the
    inference container", which is a considerably worse outcome than a slow
    start.

    So the directory is owned by the user the image declares and kept private
    to it. When that user cannot be determined, it stays root-owned and 0o700:
    a container running as someone else then recompiles every start, which is
    slow, visible in the runtime log tail, and safe.
    """
    try:
        path = cache_dir_for(deployment_id, root=root)
        path.mkdir(parents=True, exist_ok=True)
        # Private to its owner. Nothing else on the host has any business
        # reading, and certainly not writing, what the runtime will execute.
        os.chmod(path, 0o700)
        if owner_uid is not None:
            os.chown(path, owner_uid, -1)
    except (OSError, ValueError):
        # A containment failure (ValueError) is handled exactly like a
        # permissions failure (OSError): degrade to no cache. The route
        # boundary is what actually refuses a bad deployment id;
        # this is defense in depth, and reaching it at
        # all means something upstream already went wrong -- not a reason to
        # also fail the deployment start over a performance directory.
        return None
    return str(path)


def discard(deployment_id: str, *, root: Path | None = None) -> bool:
    """Delete a deployment's cache. Returns whether anything was there.

    Called when the deployment is removed. Deliberately unlike model artifacts
    and images, which the design *retains*: those are expensive to re-acquire and
    may be shared, while a compile cache is derived from them and rebuilds
    itself. Keeping it after its deployment is gone would leak disk that
    nothing will ever claim.

    It is also the only supported way to force a cold start, which matters
    because a stale compile cache is a real failure class — one where warm and
    cold starts could diverge and the cache is the last thing anyone suspects.
    """
    try:
        path = cache_dir_for(deployment_id, root=root)
    except ValueError:
        return False
    if not path.exists():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def inspect(deployment_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Report what is actually *in* a deployment's cache.

    Provisioning a directory and a runtime using it are different facts, and
    the feature shipped able to state only the first. That gap became the blocker
    it was always going to be: on 2026-08-11 a restart reused persisted
    ``torch.compile`` artefacts and cut initialisation from 143.6s to 48.2s,
    while Triton kernels still JIT-compiled on the request path — and nobody
    could tell whether ``TRITON_CACHE_DIR`` was being honoured or whether the
    warning simply does not mean what it appears to.

    Distinguishing those needs the directory listing, not more argument. So:
    per-subtree entry count and byte total, measured now, stored nowhere.

    Sizes are computed by walking, which is bounded by what a compile cache
    holds (thousands of small files at most) and is why this is not on the
    observation path — it answers a question an operator asked, not one the
    product asks on every poll.
    """
    try:
        path = cache_dir_for(deployment_id, root=root)
    except ValueError:
        return {"path": None, "present": False, "subtrees": []}
    if not path.is_dir():
        return {"path": str(path), "present": False, "subtrees": []}

    subtrees = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        files = 0
        total = 0
        for entry in child.rglob("*"):
            if entry.is_file():
                files += 1
                with contextlib.suppress(OSError):
                    total += entry.stat().st_size
        subtrees.append({"name": child.name, "files": files, "bytes": total})

    return {"path": str(path), "present": True, "subtrees": subtrees}
