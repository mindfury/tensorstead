"""Which local image a reference means on this node, and what to call it.

A digest-pinned reference (``name@sha256:X``) names a manifest. An image copied
from a peer arrives without that name -- ``docker save``/``load`` carries none --
so the daemon holds it only by id, and asking for ``name@sha256:X`` misses it
(measured on the Sparks, 2026-10-16). On Docker's containerd image store that id
*is* the manifest digest ``X``, so the image is exactly the one pinned; on the
classic store the id is the config digest, never equal, and nothing matches.

Every place the agent hands a reference to the daemon has to use the same answer.
Resolving to "present" and then creating the container from the unresolvable
name is worse than pulling: the create fails with "no such image".
"""

from __future__ import annotations

from typing import Any

from tensorstead.domain.models import pinned_digest


def resolve_local(engine: Any, reference: str) -> tuple[str, str] | None:
    """``(reference to give the daemon, image id)`` for a held image, else ``None``."""
    lookup = getattr(engine, "image_digest", None)
    if lookup is None:
        return None
    found = lookup(reference)
    if found:
        return reference, str(found)
    pinned = pinned_digest(reference)
    if pinned is not None and lookup(pinned) == pinned:
        return pinned, pinned
    return None
