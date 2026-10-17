"""Decide whether an image archive may be loaded, before the daemon sees it.

``docker load`` takes every image and every tag an archive carries, and only
then can anyone ask what it took. The engine used to return the first loaded
image's id and the callers compared just that, so an archive carrying the
expected image *plus* another image, or extra tags -- including a tag that
already named something else on this node -- passed verification and left the
rest behind (CodeGuard CG-04). A compromised source agent answering a
legitimate distribution is enough to send one.

So the archive's own declaration is read first, from ``index.json`` (and
``manifest.json`` where present), without loading anything: exactly one image,
an identity matching the one expected, and no name outside those the caller
allowed. Only then is it loaded.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from typing import Any

# A declaration document, never image content: anything this large is not an
# index or manifest list any daemon writes.
_MAX_DECLARATION_BYTES = 4 * 1024 * 1024


class ArchiveRefused(ValueError):
    """The archive declares something the caller did not authorize."""


@dataclass(frozen=True)
class ArchiveImage:
    """One image an archive declares: what it may be identified as, and its names."""

    identities: frozenset[str]
    names: frozenset[str]


def familiar(name: str) -> str:
    """``docker.io/library/alpine:3`` and ``alpine:3`` are the same name."""
    for prefix in ("docker.io/library/", "docker.io/", "index.docker.io/library/"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def declared_images(archive_path: str) -> list[ArchiveImage]:
    """The images ``archive_path`` declares, read without loading it."""
    with tarfile.open(archive_path, "r:*") as tar:
        members = {m.name.lstrip("./"): m for m in tar.getmembers()}

        def _read(name: str) -> Any:
            member = members.get(name)
            if member is None or not member.isfile():
                return None
            if member.size > _MAX_DECLARATION_BYTES:
                raise ArchiveRefused(f"{name} is {member.size} bytes; not a declaration")
            handle = tar.extractfile(member)
            return None if handle is None else json.loads(handle.read().decode("utf-8"))

        def _blob(digest: str) -> Any:
            algorithm, _, hexdigest = digest.partition(":")
            return _read(f"blobs/{algorithm}/{hexdigest}")

        index = _read("index.json")
        if not isinstance(index, dict):
            raise ArchiveRefused("archive has no index.json; not an image archive this loads")
        repo_tags: list[str] = []
        legacy = _read("manifest.json")
        if isinstance(legacy, list):
            for entry in legacy:
                repo_tags += [t for t in (entry or {}).get("RepoTags") or [] if t]

        images: list[ArchiveImage] = []
        for descriptor in index.get("manifests") or []:
            digest = str(descriptor.get("digest") or "")
            identities = {digest} if digest else set()
            manifest = _blob(digest) if digest else None
            if isinstance(manifest, dict):
                config = (manifest.get("config") or {}).get("digest")
                if config:
                    identities.add(str(config))
            annotations = descriptor.get("annotations") or {}
            names = {
                familiar(str(value))
                for key in ("io.containerd.image.name", "org.opencontainers.image.ref.name")
                if (value := annotations.get(key)) and ("/" in value or ":" in value)
            }
            images.append(ArchiveImage(frozenset(identities), frozenset(names)))

        # manifest.json's RepoTags belong to the one image when there is one;
        # with several, they are attributed to all, which only makes the
        # refusal below stricter.
        if repo_tags:
            extra = frozenset(familiar(t) for t in repo_tags)
            images = [ArchiveImage(i.identities, i.names | extra) for i in images]
        return images


def authorize(
    images: list[ArchiveImage],
    *,
    expected_id: str | None,
    allowed_names: set[str],
) -> ArchiveImage:
    """The single image to load, or ``ArchiveRefused`` naming what is wrong."""
    if len(images) != 1:
        raise ArchiveRefused(f"archive declares {len(images)} images; exactly one is accepted")
    (image,) = images
    if expected_id is not None and expected_id not in image.identities:
        raise ArchiveRefused(
            f"archive declares {sorted(image.identities)}, not the expected {expected_id!r}"
        )
    allowed = {familiar(n) for n in allowed_names}
    unexpected = sorted(n for n in image.names if n not in allowed)
    if unexpected:
        raise ArchiveRefused(f"archive carries names it was not authorized to set: {unexpected}")
    return image
