"""Azure ML registry model source (``source_id: "azureml"``).

Some models are published only to the Azure ML public registry -- Microsoft's
MAI-Code-1.1-Flash-On-Device is the case that forced this. Its public endpoint
answers without an Azure account: model metadata, a digest of the version's
file set, and a short-lived read/list SAS link to the version's blob container.

Identity:

- ``model_id`` is ``<registry>/<model name>``, e.g.
  ``azureml/MAI-Code-1.1-Flash-On-Device``.
- A **version is required**. Registry versions are immutable, and an implicit
  "latest" would be the moving reference the rest of the product refuses.
- The resolved revision is ``<version>@<manifest digest>`` -- the registry's
  ``blobManifestDigest`` (SHA-512) in hex. A version whose files changed under
  the same number resolves to a different revision, and an acquire pinned to
  the old one is refused rather than silently fetching new bytes.

Transfer: every selected blob is streamed to ``<name>.part``, its MD5 checked
against the ``Content-MD5`` the container listing publishes, then renamed. The
SAS signature is a credential for the duration of the download: it is never
logged, written to disk, or included in an error message.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from tensorstead.ports.model_source import AcquireProgress

SOURCE_ID = "azureml"
_API = os.environ.get("TENSORSTEAD_AZUREML_API", "https://eastus.api.azureml.ms")
_METADATA_PATH = "/modelregistry/v1.0/registry/models/nonazureaccount"
_CHUNK = 8 * 1024 * 1024
_TIMEOUT = 60.0
_MAX_LISTING_BYTES = 4 * 1024 * 1024


@dataclass
class AzureMLAcquisition:
    resolved_revision: str
    revision_pinned: bool
    size_bytes: int
    content_digest: str | None = None


@dataclass(frozen=True)
class _Blob:
    name: str
    size: int
    md5_b64: str | None


def _split_model_id(model_id: str) -> tuple[str, str]:
    registry, _, name = model_id.partition("/")
    if not registry or not name or "/" in name:
        raise ValueError(
            f"azureml model ids are '<registry>/<model name>', e.g. "
            f"'azureml/MAI-Code-1.1-Flash-On-Device'; got {model_id!r}"
        )
    return registry, name


def _split_revision(revision: str | None) -> tuple[str, str | None]:
    if not revision:
        raise ValueError(
            "an azureml model needs an explicit version (registry versions are immutable; "
            "there is no implicit latest)"
        )
    version, _, digest = revision.partition("@")
    if not version.isdigit():
        raise ValueError(
            f"azureml version must be a number, optionally '@<digest>'; got {revision!r}"
        )
    return version, (digest.lower() or None)


def _redact(text: str) -> str:
    """Strip any SAS query from text that might reach an error message."""
    return text.split("?", 1)[0]


class AzureMLRegistrySource:
    """A ``ModelSource`` for the Azure ML public model registry."""

    supports_revision_pinning = True
    requires_credential = False

    def __init__(self, http: Any | None = None) -> None:
        # ``http`` is an ``httpx.Client``-like object; injectable so tests run
        # without the network.
        self._http = http

    def source_id(self) -> str:
        return SOURCE_ID

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=_TIMEOUT, follow_redirects=False)
        return self._http

    def _metadata(self, model_id: str, version: str) -> dict[str, Any]:
        registry, name = _split_model_id(model_id)
        asset = f"azureml://registries/{registry}/models/{name}/versions/{version}"
        response = self._client().get(f"{_API}{_METADATA_PATH}", params={"assetId": asset})
        if response.status_code == 404:
            raise LookupError(f"{model_id} version {version} is not in the Azure ML registry")
        if response.status_code >= 400:
            raise RuntimeError(
                f"Azure ML registry answered {response.status_code} for {model_id} v{version}"
            )
        body: dict[str, Any] = response.json()
        if body.get("modelEntity", {}).get("isArchived"):
            raise LookupError(f"{model_id} version {version} is archived in the registry")
        return body

    @staticmethod
    def _manifest_hex(body: dict[str, Any]) -> str:
        digest = body.get("blobManifestDigest")
        if not digest:
            raise RuntimeError(
                "the registry published no manifest digest; refusing an unpinnable version"
            )
        return base64.b64decode(digest).hex()

    def resolve(self, model_id: str, revision: str | None) -> str:
        version, pinned = _split_revision(revision)
        current = self._manifest_hex(self._metadata(model_id, version))
        if pinned is not None and pinned != current:
            raise RuntimeError(
                f"{model_id} version {version} now has manifest {current[:16]}..., not the pinned "
                f"{pinned[:16]}...: its files changed under the same version; refusing"
            )
        return f"{version}@{current}"

    def _list(self, sas: str) -> list[_Blob]:
        base, _, query = sas.partition("?")
        response = self._client().get(f"{base}?{query}&restype=container&comp=list")
        if response.status_code >= 400:
            raise RuntimeError(f"listing {_redact(sas)} failed: HTTP {response.status_code}")
        content = response.content
        # Untrusted XML, parsed without a new dependency: a container listing
        # never declares a DOCTYPE or entities, so any that do are refused
        # outright (entity-expansion and external-entity attacks both need
        # them), and the size is bounded. ElementTree does not fetch external
        # entities, and expat >= 2.4 also limits expansion.
        if len(content) > _MAX_LISTING_BYTES:
            raise RuntimeError(f"container listing is {len(content)} bytes; refusing")
        if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
            raise RuntimeError("container listing declares a DOCTYPE or entity; refusing")
        root = ET.fromstring(content)  # noqa: S314 - guarded above
        blobs = []
        for blob in root.iter("Blob"):
            name = blob.findtext("Name") or ""
            size = int(blob.findtext("Properties/Content-Length") or 0)
            blobs.append(
                _Blob(name=name, size=size, md5_b64=blob.findtext("Properties/Content-MD5"))
            )
        return blobs

    def acquire(
        self,
        model_id: str,
        revision: str,
        destination_dir: str,
        credential: str | None,  # noqa: ARG002 - public registry; port parity
        progress: AcquireProgress | None = None,
        file_selector: tuple[str, ...] = (),
    ) -> AzureMLAcquisition:
        version, pinned = _split_revision(revision)
        body = self._metadata(model_id, version)
        current = self._manifest_hex(body)
        if pinned is not None and pinned != current:
            raise RuntimeError(
                f"{model_id} version {version} changed since it was resolved; refusing"
            )
        sas = body.get("blobSasUri")
        if not sas:
            raise RuntimeError(f"the registry offered no download link for {model_id} v{version}")
        blobs = [
            b
            for b in self._list(sas)
            if not file_selector or any(fnmatch.fnmatch(b.name, pat) for pat in file_selector)
        ]
        if not blobs:
            raise LookupError(
                f"{model_id} v{version} has no files matching {list(file_selector) or 'any'}"
            )
        total = sum(b.size for b in blobs) or 1
        done = 0
        dest = Path(destination_dir)
        base, _, query = sas.partition("?")
        for blob in blobs:
            rel = PurePosixPath(blob.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"refusing blob name outside the model tree: {blob.name!r}")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            md5 = hashlib.md5()  # noqa: S324 - integrity check against Azure's Content-MD5
            with self._client().stream("GET", f"{base}/{quote(blob.name)}?{query}") as response:
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"downloading {blob.name} failed: HTTP {response.status_code}"
                    )
                with part.open("wb") as handle:
                    for chunk in response.iter_bytes(_CHUNK):
                        handle.write(chunk)
                        md5.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(min(done / total, 0.99), f"downloading {blob.name}")
            if blob.md5_b64 is not None and base64.b64encode(md5.digest()).decode() != blob.md5_b64:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{blob.name} does not match its published MD5; refusing it")
            if part.stat().st_size != blob.size:
                part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{blob.name} is {part.stat().st_size} bytes, expected {blob.size}"
                )
            part.rename(target)
        return AzureMLAcquisition(
            resolved_revision=f"{version}@{current}",
            revision_pinned=True,
            size_bytes=sum(b.size for b in blobs),
        )

    def verify(self, destination_dir: Path) -> None:
        """The same structural check as Hugging Face trees: GGUF or config.json."""
        from tensorstead.adapters.sources.huggingface import HuggingFaceSource

        HuggingFaceSource().verify(destination_dir)
