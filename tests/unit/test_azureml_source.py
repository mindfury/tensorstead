"""The Azure ML registry source: pinned identity, verified transfer, no leaked SAS.

MAI-Code-1.1-Flash-On-Device is published only to the Azure ML public
registry; Tensorstead fetched only from Hugging Face, so the alternative was a
hand download outside the product.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tensorstead.adapters.sources.azureml import AzureMLRegistrySource

pytestmark = pytest.mark.unit

_MODEL = "azureml/MAI-Code-1.1-Flash-On-Device"
_SIG = "SECRET-SIGNATURE-VALUE"
_SAS = f"https://acct.blob.core.windows.net/container?sv=2025&sr=c&sp=rl&sig={_SIG}"
_MANIFEST = hashlib.sha512(b"manifest").digest()
_FILES = {
    "model.gguf": b"GGUF" + b"\x00" * 64,
    "dflash.gguf": b"GGUF" + b"\x01" * 16,
    "llama-config.json": b"{}",
}


def _md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()  # noqa: S324


def _listing(
    files: dict[str, bytes], md5s: dict[str, str] | None = None, prefix: bytes = b""
) -> bytes:
    blobs = "".join(
        f"<Blob><Name>{n}</Name><Properties><Content-Length>{len(d)}</Content-Length>"
        f"<Content-MD5>{(md5s or {}).get(n, _md5(d))}</Content-MD5></Properties></Blob>"
        for n, d in files.items()
    )
    return (
        prefix
        + b'<?xml version="1.0"?><EnumerationResults><Blobs>'
        + blobs.encode()
        + b"</Blobs></EnumerationResults>"
    )


def _source(
    files: dict[str, bytes] = _FILES,
    *,
    listing: bytes | None = None,
    status: int = 200,
    archived: bool = False,
) -> tuple[AzureMLRegistrySource, list[str]]:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if "modelregistry" in url:
            if status != 200:
                return httpx.Response(status)
            return httpx.Response(
                200,
                json={
                    "blobSasUri": _SAS,
                    "blobManifestDigest": base64.b64encode(_MANIFEST).decode(),
                    "modelEntity": {"isArchived": archived},
                },
            )
        if "comp=list" in url:
            return httpx.Response(200, content=listing if listing is not None else _listing(files))
        name = request.url.path.rsplit("/", 1)[-1]
        if name in files:
            return httpx.Response(200, content=files[name])
        return httpx.Response(403, text=f"denied {url}")

    return AzureMLRegistrySource(http=httpx.Client(transport=httpx.MockTransport(handler))), seen


def test_resolve_pins_the_version_to_its_manifest_digest() -> None:
    src, seen = _source()
    assert src.resolve(_MODEL, "2") == f"2@{_MANIFEST.hex()}"
    assert "versions%2F2" in seen[0] or "versions/2" in seen[0]


def test_a_pin_whose_files_changed_is_refused() -> None:
    src, _ = _source()
    with pytest.raises(RuntimeError, match="files changed"):
        src.resolve(_MODEL, "2@" + "0" * 128)


@pytest.mark.parametrize(
    ("model", "rev", "match"),
    [
        (_MODEL, None, "explicit version"),
        (_MODEL, "latest", "must be a number"),
        ("MAI-Code", "2", "registry>/<model"),
    ],
)
def test_identity_must_be_explicit(model: str, rev: str | None, match: str) -> None:
    src, _ = _source()
    with pytest.raises(ValueError, match=match):
        src.resolve(model, rev)


def test_unknown_and_archived_versions_are_refused() -> None:
    with pytest.raises(LookupError, match="not in the Azure ML registry"):
        _source(status=404)[0].resolve(_MODEL, "9")
    with pytest.raises(LookupError, match="archived"):
        _source(archived=True)[0].resolve(_MODEL, "2")


def test_acquire_downloads_the_selected_files_verified(tmp_path: Path) -> None:
    src, _ = _source()
    rev = src.resolve(_MODEL, "2")
    out = src.acquire(_MODEL, rev, str(tmp_path), None, file_selector=("*.gguf",))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dflash.gguf", "model.gguf"]
    assert (tmp_path / "model.gguf").read_bytes() == _FILES["model.gguf"]
    assert out.resolved_revision == rev and out.revision_pinned
    assert out.size_bytes == len(_FILES["model.gguf"]) + len(_FILES["dflash.gguf"])
    src.verify(tmp_path)


def test_a_file_that_fails_its_md5_is_refused_and_not_left_behind(tmp_path: Path) -> None:
    src, _ = _source(listing=_listing(_FILES, md5s={"model.gguf": _md5(b"other")}))
    with pytest.raises(RuntimeError, match="published MD5"):
        src.acquire(_MODEL, "2", str(tmp_path), None)
    assert not any(p.name.startswith("model.gguf") for p in tmp_path.iterdir())


def test_hostile_listings_are_refused(tmp_path: Path) -> None:
    bomb = b'<!DOCTYPE l [<!ENTITY a "aaaa">]>'
    with pytest.raises(RuntimeError, match="DOCTYPE"):
        _source(listing=_listing(_FILES, prefix=bomb))[0].acquire(_MODEL, "2", str(tmp_path), None)
    with pytest.raises(RuntimeError, match="outside the model tree"):
        _source({"../escape.gguf": b"x"})[0].acquire(_MODEL, "2", str(tmp_path), None)


def test_the_sas_signature_never_reaches_an_error(tmp_path: Path) -> None:
    src, _ = _source(listing=_listing({"missing.gguf": b"x"}))
    with pytest.raises(RuntimeError) as caught:
        src.acquire(_MODEL, "2", str(tmp_path), None)
    assert _SIG not in str(caught.value)


def test_the_agent_routes_acquires_by_source_id(tmp_path: Path) -> None:
    from tensorstead.adapters.sources.huggingface import HuggingFaceSource
    from tensorstead.agent.app import build_agent_app
    from tests.fakes.hf_hub import FakeHuggingFaceHub

    app = build_agent_app(
        management_token="t",
        model_source=HuggingFaceSource(hub=FakeHuggingFaceHub()),
        store_dir=tmp_path / "m",
        marker_dir=tmp_path / "s",
    )
    assert set(app.state.model_sources) == {"huggingface", "azureml"}
    src, _ = _source()
    app.state.model_sources["azureml"] = src
    client = TestClient(app)
    ok = client.post(
        "/agent/v1/models:acquire",
        json={
            "source_id": "azureml",
            "source_model_id": _MODEL,
            "revision": "2",
            "file_selector": ["*.gguf"],
        },
        headers={"Authorization": "Bearer t"},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["resolved_revision"] == f"2@{_MANIFEST.hex()}"
    unknown = client.post(
        "/agent/v1/models:acquire",
        json={"source_id": "modelscope", "source_model_id": "x/y"},
        headers={"Authorization": "Bearer t"},
    )
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "unknown_model_source"
