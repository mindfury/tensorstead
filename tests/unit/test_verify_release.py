"""A published release is installed only after its provenance verifies.

CodeGuard CG-17: the recorded wheel checksum was unauthenticated -- anyone able
to replace the wheel could replace its sidecar -- and installation verified
nothing. ``scripts/verify_release.py`` gates ``make deploy-release``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[2] / "scripts" / "verify_release.py"
)
assert _SPEC and _SPEC.loader
vr: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vr)

_V = "9.9.9"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(out: Path, *, wheelhouse_ok: bool = True) -> None:
    out.mkdir(parents=True)
    wheel_bytes = b"wheel"
    (out / f"tensorstead-{_V}-py3-none-any.whl").write_bytes(wheel_bytes)
    (out / f"tensorstead-{_V}.tar.gz").write_bytes(b"sdist")
    dep = b"dependency wheel"
    manifest = {
        "target": "linux-aarch64-cp312",
        "files": {"dep-1-py3-none-any.whl": "sha256:" + _sha(dep if wheelhouse_ok else b"other")},
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in (
            (f"tensorstead-{_V}-wheelhouse/wheelhouse.json", json.dumps(manifest).encode()),
            (f"tensorstead-{_V}-wheelhouse/wheels/dep-1-py3-none-any.whl", dep),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (out / f"tensorstead-{_V}-wheelhouse.tar.gz").write_bytes(buffer.getvalue())
    sums = "".join(
        f"{_sha(p.read_bytes())}  {p.name}\n" for p in sorted(out.iterdir()) if p.is_file()
    )
    (out / "SHA256SUMS").write_text(sums)


def _gh(attested: set[str] | None = None) -> tuple[Any, list[list[str]]]:
    calls: list[list[str]] = []

    def run(args: list[str], **_kw: Any) -> Any:
        calls.append(args)
        ok = attested is None or Path(args[3]).name in attested
        return subprocess.CompletedProcess(args, 0 if ok else 1, "", "" if ok else "no attestation")

    return run, calls


def test_an_attested_untampered_release_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path / "r")
    run, calls = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    paths = vr.verify(_V, tmp_path / "r", download=False)
    assert paths["wheelhouse"].is_dir()
    verify_calls = [c for c in calls if c[:3] == ["gh", "attestation", "verify"]]
    assert len(verify_calls) == 4  # SHA256SUMS + 3 assets
    for call in verify_calls:
        assert call[call.index("--repo") + 1] == "mindfury/tensorstead"
        assert call[call.index("--signer-workflow") + 1].endswith("/.github/workflows/release.yml")
        assert call[call.index("--source-ref") + 1] == f"refs/tags/v{_V}"


def test_a_missing_attestation_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _release(tmp_path / "r")
    run, _ = _gh(attested={"SHA256SUMS", f"tensorstead-{_V}.tar.gz"})  # the wheel lacks one
    monkeypatch.setattr(vr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="no valid provenance"):
        vr.verify(_V, tmp_path / "r", download=False)


def test_a_tampered_asset_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _release(tmp_path / "r")
    (tmp_path / "r" / f"tensorstead-{_V}-py3-none-any.whl").write_bytes(b"swapped")
    run, _ = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="does not match SHA256SUMS"):
        vr.verify(_V, tmp_path / "r", download=False)


def test_an_unlisted_asset_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _release(tmp_path / "r")
    (tmp_path / "r" / "extra.whl").write_bytes(b"smuggled")
    run, _ = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="not covered by SHA256SUMS"):
        vr.verify(_V, tmp_path / "r", download=False)


def test_a_wheelhouse_mismatch_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _release(tmp_path / "r", wheelhouse_ok=False)
    run, _ = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="does not match its manifest"):
        vr.verify(_V, tmp_path / "r", download=False)


def test_a_release_without_checksums_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path / "r")
    (tmp_path / "r" / "SHA256SUMS").unlink()
    run, _ = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    with pytest.raises(SystemExit, match="no SHA256SUMS"):
        vr.verify(_V, tmp_path / "r", download=False)


def test_offline_verification_uses_the_downloaded_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _release(tmp_path / "r")
    run, calls = _gh()
    monkeypatch.setattr(vr.subprocess, "run", run)
    vr.verify(_V, tmp_path / "r", download=False, bundle_dir=tmp_path / "bundles")
    call = next(c for c in calls if c[:3] == ["gh", "attestation", "verify"])
    assert call[call.index("--bundle") + 1].startswith(str(tmp_path / "bundles" / "sha256:"))
