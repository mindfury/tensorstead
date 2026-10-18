#!/usr/bin/env python3
"""Fetch a published release and verify its provenance before anything installs it.

A wheel checksum was recorded, but nothing authenticated it: whoever could
replace a wheel could replace its unsigned checksum sidecar too, and the
installer verified neither (CodeGuard CG-17). Releases now carry signed build
provenance (``actions/attest-build-provenance`` in ``release.yml``), and this is
the gate in front of ``make deploy-release``:

1. every release asset must carry an attestation signed by **this repository's**
   ``.github/workflows/release.yml``, built from the matching ``refs/tags/vX``
   (``gh attestation verify``); a missing attestation, another repository, or
   another workflow fails;
2. every file must match ``SHA256SUMS`` -- itself an attested asset;
3. the wheelhouse is unpacked and each wheel re-checked against its manifest.

Any failure exits non-zero before a single byte is deployed. ``--bundle-dir``
verifies offline against attestation bundles fetched earlier with
``gh attestation download``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "mindfury/tensorstead"
WORKFLOW = f"{REPO}/.github/workflows/release.yml"


class VerificationFailed(SystemExit):
    """A release did not prove where it came from; nothing may be installed from it."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _attested(path: Path, version: str, bundle_dir: Path | None) -> None:
    args = [
        "gh",
        "attestation",
        "verify",
        str(path),
        "--repo",
        REPO,
        "--signer-workflow",
        WORKFLOW,
        "--source-ref",
        f"refs/tags/v{version}",
    ]
    if bundle_dir is not None:
        args += ["--bundle", str(bundle_dir / f"sha256:{_sha256(path)}.jsonl")]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise VerificationFailed(
            f"{path.name}: no valid provenance from {WORKFLOW} at v{version}; refusing it\n"
            f"{result.stderr.strip()[-2000:]}"
        )


def verify(
    version: str, out: Path, *, bundle_dir: Path | None = None, download: bool = True
) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    if download:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                f"v{version}",
                "--repo",
                REPO,
                "--dir",
                str(out),
                "--clobber",
            ],
            check=True,
        )
    sums = out / "SHA256SUMS"
    if not sums.is_file():
        raise VerificationFailed(f"v{version} has no SHA256SUMS; refusing an unverifiable release")

    listed: dict[str, str] = {}
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition("  ")
        listed[name.strip()] = digest.strip()
    assets = sorted(p for p in out.iterdir() if p.is_file() and p.name in listed)
    unlisted = sorted(
        p.name
        for p in out.iterdir()
        if p.is_file() and p.name not in listed and p.name != "SHA256SUMS"
    )
    if unlisted:
        raise VerificationFailed(f"assets not covered by SHA256SUMS: {unlisted}")
    if len(assets) != len(listed):
        raise VerificationFailed(f"SHA256SUMS names {len(listed)} files, {len(assets)} present")

    for path in [sums, *assets]:
        _attested(path, version, bundle_dir)
    for path in assets:
        if _sha256(path) != listed[path.name]:
            raise VerificationFailed(f"{path.name} does not match SHA256SUMS; refusing it")

    wheel = out / f"tensorstead-{version}-py3-none-any.whl"
    archive = out / f"tensorstead-{version}-wheelhouse.tar.gz"
    if not wheel.is_file() or not archive.is_file():
        raise VerificationFailed("the release lacks its wheel or its wheelhouse")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out, filter="data")
    wheelhouse = out / f"tensorstead-{version}-wheelhouse"
    manifest = json.loads((wheelhouse / "wheelhouse.json").read_text())
    for name, expected in manifest["files"].items():
        if f"sha256:{_sha256(wheelhouse / 'wheels' / name)}" != expected:
            raise VerificationFailed(f"wheelhouse file {name} does not match its manifest")
    return {"wheel": wheel, "wheelhouse": wheelhouse}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or ROOT / "dist" / f"release-{args.version}"
    paths = verify(args.version, out, bundle_dir=args.bundle_dir)
    print(f"verified v{args.version}: {paths['wheel']} and {paths['wheelhouse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
