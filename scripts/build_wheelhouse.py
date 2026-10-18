#!/usr/bin/env python3
"""Build the exact, hash-verified dependency set appliances install, offline.

CI installs the reviewed lock (``uv sync --locked``), but appliance installs
used to let pip resolve the wheel's lower-bounded dependencies fresh against an
index on every deploy, so a release, an index change or a compromised package
could change what ran independently of the reviewed build (CodeGuard CG-14).

This writes ``dist/tensorstead-<version>-wheelhouse/``:

- ``requirements.txt`` -- every dependency the appliance extras need, pinned to
  ``uv.lock`` with its hashes, filtered to the *target* platform. Markers are
  evaluated here against the declared target, not this machine, so a
  Linux-only dependency is never dropped because the build ran on macOS.
- ``wheels/`` -- exactly those pins as binary wheels for the target, downloaded
  with ``--no-deps`` (nothing is resolved) and ``--require-hashes``, then each
  file's SHA-256 re-checked against the lock.
- ``wheelhouse.json`` -- the target and every file's SHA-256.

Ansible installs from it with ``--no-index --require-hashes --only-binary=:all:``:
offline, verified, no source builds, and the same set every time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXTRAS = ("coordinator", "agent", "cli", "mcp")

# The DGX Spark appliances: Linux, aarch64, CPython 3.12 (the
# ``tensorstead_python_executable`` the Ansible role creates the venv with).
TARGETS: dict[str, dict[str, object]] = {
    "linux-aarch64-cp312": {
        "markers": {
            "os_name": "posix",
            "sys_platform": "linux",
            "platform_system": "Linux",
            "platform_machine": "aarch64",
            "platform_python_implementation": "CPython",
            "implementation_name": "cpython",
            "python_version": "3.12",
            "python_full_version": "3.12.3",
            "platform_release": "",
            "platform_version": "",
            "implementation_version": "3.12.3",
        },
        "platforms": [
            "manylinux_2_39_aarch64",
            "manylinux_2_28_aarch64",
            "manylinux_2_17_aarch64",
            "manylinux2014_aarch64",
            "linux_aarch64",
        ],
        "python_version": "3.12",
        "abis": ["cp312", "abi3", "none"],
    },
}


def _version() -> str:
    return str(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"])


def _exported() -> str:
    """The locked appliance dependency set, with every artifact's hash."""
    args = [
        "uv",
        "--color",
        "never",
        "export",
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--format",
        "requirements-txt",
    ]
    for extra in EXTRAS:
        args += ["--extra", extra]
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def _blocks(exported: str) -> list[tuple[Requirement, list[str]]]:
    """(requirement, its hashes) for each entry, continuation lines joined.

    Colour codes are stripped first: some environments force colour even into a
    pipe, and a terminal setting must never change what is parsed.
    """
    plain = re.sub(r"\x1b\[[0-9;]*m", "", exported)
    joined = re.sub(r"\\\n\s*", " ", plain)
    entries: list[tuple[Requirement, list[str]]] = []
    for line in joined.splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        hashes = re.findall(r"--hash=(sha256:[0-9a-f]{64})", line)
        spec = re.sub(r"\s*--hash=\S+", "", line).strip()
        entries.append((Requirement(spec), hashes))
    return entries


def build(target_name: str, out: Path) -> dict[str, object]:
    target = TARGETS[target_name]
    environment = {k: str(v) for k, v in dict(target["markers"]).items()}  # type: ignore[call-overload]
    wanted: list[tuple[Requirement, list[str]]] = []
    for requirement, hashes in _blocks(_exported()):
        if requirement.marker is not None and not Marker(str(requirement.marker)).evaluate(
            environment
        ):
            continue
        if not hashes:
            raise SystemExit(
                f"{requirement.name}: no hash in the lock; refusing an unverifiable pin"
            )
        wanted.append((requirement, hashes))

    if out.exists():
        shutil.rmtree(out)
    (out / "wheels").mkdir(parents=True)
    lines = []
    for requirement, hashes in wanted:
        pin = f"{requirement.name}{requirement.specifier}"
        lines.append(" ".join([pin, *(f"--hash={h}" for h in hashes)]))
    requirements = out / "requirements.txt"
    requirements.write_text("\n".join(lines) + "\n")

    download = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--quiet",
        "--no-deps",
        "--only-binary=:all:",
        "--require-hashes",
        "--python-version",
        str(target["python_version"]),
        "--implementation",
        "cp",
        "-r",
        str(requirements),
        "-d",
        str(out / "wheels"),
    ]
    for platform in target["platforms"]:  # type: ignore[attr-defined]
        download += ["--platform", platform]
    for abi in target["abis"]:  # type: ignore[attr-defined]
        download += ["--abi", abi]
    subprocess.run(download, check=True)

    allowed = {h.split(":", 1)[1] for _, hashes in wanted for h in hashes}
    files: dict[str, str] = {}
    for wheel in sorted((out / "wheels").iterdir()):
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        if digest not in allowed:
            raise SystemExit(f"{wheel.name}: sha256 {digest} is not in the lock; refusing it")
        files[wheel.name] = f"sha256:{digest}"
    if len(files) != len(wanted):
        raise SystemExit(f"downloaded {len(files)} wheels for {len(wanted)} pins")

    manifest = {"target": target_name, "requirements": len(wanted), "files": files}
    (out / "wheelhouse.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="linux-aarch64-cp312", choices=sorted(TARGETS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or DIST / f"tensorstead-{_version()}-wheelhouse"
    manifest = build(args.target, out)
    print(f"wheelhouse: {out} ({manifest['requirements']} pins, target {manifest['target']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
