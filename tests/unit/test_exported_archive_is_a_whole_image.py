"""An export that writes no content is not a successful export (findings/078).

``docker save`` returned cleanly and wrote **16,896 bytes** for a 31 GB image
on this estate's own head node. The image ran there; it simply had no
exportable content, having been produced by the classic builder -- which is
what docker-py's ``images.build`` drives, the SDK having no BuildKit support --
under Docker 29's containerd image store. Measured on the same host:

    classic-builder image, 31 GB on disk  ->  docker save = 16,896 bytes
    the same recipe built with BuildKit   ->  docker save = 9,814,834,688
    a pulled base image                   ->  docker save = 10,533,214,208

That 16 KB archive transfers, loads without error, and reports the expected
image id on the far side, because the id comes from the manifest and the
manifest is the part that is present. The destination-side materializability
check catches it, but blames the wrong node. This catches it at the source.

The check is deliberately self-referential -- the archive must carry every
digest its own index and manifests name -- so it needs no agreement with the
daemon and cannot drift from one.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from tensorstead.agent.container_engine.docker_py import _archive_omits_its_own_blobs

pytestmark = pytest.mark.unit


def _digest_of(payload: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_archive(path: Path, *, blobs: dict[str, bytes], index: dict) -> None:
    with tarfile.open(path, "w") as tar:

        def _add(name: str, payload: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

        _add("index.json", json.dumps(index).encode())
        for digest, payload in blobs.items():
            algorithm, _, hexdigest = digest.partition(":")
            _add(f"blobs/{algorithm}/{hexdigest}", payload)


def _whole_image(path: Path) -> tuple[str, str]:
    """A self-contained archive: config and both layers present."""
    layers = [b"layer-one", b"layer-two"]
    config = b'{"architecture":"arm64","os":"linux"}'
    manifest = {
        "config": {"digest": _digest_of(config)},
        "layers": [{"digest": _digest_of(each)} for each in layers],
    }
    manifest_bytes = json.dumps(manifest).encode()
    blobs = {_digest_of(config): config, _digest_of(manifest_bytes): manifest_bytes}
    for each in layers:
        blobs[_digest_of(each)] = each
    _write_archive(
        path,
        blobs=blobs,
        index={"manifests": [{"digest": _digest_of(manifest_bytes)}]},
    )
    return _digest_of(config), _digest_of(layers[0])


def test_a_self_contained_archive_is_accepted(tmp_path: Path) -> None:
    archive = tmp_path / "whole.tar"
    _whole_image(archive)

    assert _archive_omits_its_own_blobs(str(archive)) is None


def test_the_real_failure_a_manifest_with_no_content_behind_it(tmp_path: Path) -> None:
    """The 16 KB archive, reduced to its essentials.

    Index and manifest present and internally consistent; config and layers
    simply absent. Every identity check passes on this.
    """
    archive = tmp_path / "hollow.tar"
    config = b'{"architecture":"arm64","os":"linux"}'
    manifest = {
        "config": {"digest": _digest_of(config)},
        "layers": [{"digest": _digest_of(b"layer-one")}],
    }
    manifest_bytes = json.dumps(manifest).encode()
    _write_archive(
        archive,
        blobs={_digest_of(manifest_bytes): manifest_bytes},
        index={"manifests": [{"digest": _digest_of(manifest_bytes)}]},
    )

    omission = _archive_omits_its_own_blobs(str(archive))

    assert omission is not None
    assert "config" in omission
    assert "layer" in omission


def test_a_missing_layer_alone_is_enough_to_refuse(tmp_path: Path) -> None:
    """Partial content is not a lesser problem than none."""
    archive = tmp_path / "partial.tar"
    config = b'{"architecture":"arm64","os":"linux"}'
    present, missing_layer = b"layer-one", b"layer-two"
    manifest = {
        "config": {"digest": _digest_of(config)},
        "layers": [{"digest": _digest_of(present)}, {"digest": _digest_of(missing_layer)}],
    }
    manifest_bytes = json.dumps(manifest).encode()
    _write_archive(
        archive,
        blobs={
            _digest_of(manifest_bytes): manifest_bytes,
            _digest_of(config): config,
            _digest_of(present): present,
        },
        index={"manifests": [{"digest": _digest_of(manifest_bytes)}]},
    )

    omission = _archive_omits_its_own_blobs(str(archive))

    assert omission is not None
    assert _digest_of(missing_layer) in omission


def test_a_manifest_the_index_names_but_does_not_carry_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "no-manifest.tar"
    _write_archive(archive, blobs={}, index={"manifests": [{"digest": _digest_of(b"absent")}]})

    omission = _archive_omits_its_own_blobs(str(archive))

    assert omission is not None
    assert "manifest" in omission


def test_an_unrecognised_archive_shape_is_not_judged(tmp_path: Path) -> None:
    """Unrecognised is not the same as broken.

    Older archive layouts are not produced by the daemons this runs against.
    Refusing what it cannot parse would turn this check into a new way for a
    good export to fail.
    """
    archive = tmp_path / "legacy.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("manifest.json")
        payload = b"[]"
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    assert _archive_omits_its_own_blobs(str(archive)) is None


def test_something_that_is_not_an_archive_is_its_own_answer(tmp_path: Path) -> None:
    not_an_archive = tmp_path / "junk.tar"
    not_an_archive.write_bytes(b"this is not a tar file")

    omission = _archive_omits_its_own_blobs(str(not_an_archive))

    assert omission is not None
    assert "could not be read" in omission


# Multi-platform images (#61). A node pulls only its own platform, and ``docker
# save`` writes the whole index regardless, so NGC's multi-arch images exported
# from an arm64 node name an amd64 manifest the archive cannot carry. The layout
# below is the one measured on nvcr.io/nvidia/tensorrt-llm/release@sha256:124da3f1.

_ARM64 = ("linux", "arm64")
_AMD64 = ("linux", "amd64")
_ATTESTATION = {"vnd.docker.reference.type": "attestation-manifest"}


def _image_blobs(
    layers: list[bytes], *, drop_layer: bool = False
) -> tuple[bytes, dict[str, bytes]]:
    config = b'{"architecture":"arm64","os":"linux"}'
    manifest = json.dumps(
        {
            "config": {"digest": _digest_of(config)},
            "layers": [{"digest": _digest_of(each)} for each in layers],
        }
    ).encode()
    blobs = {_digest_of(manifest): manifest, _digest_of(config): config}
    for each in layers[:-1] if drop_layer else layers:
        blobs[_digest_of(each)] = each
    return manifest, blobs


def _multi_platform(
    path: Path,
    *,
    drop_arm64_layer: bool = False,
    entries: list[dict] | None = None,
    extra_blobs: dict[str, bytes] | None = None,
) -> None:
    """An index holding arm64 in full and naming amd64 and attestations."""
    arm64, blobs = _image_blobs([b"arm-one", b"arm-two"], drop_layer=drop_arm64_layer)
    amd64 = b'{"absent": "amd64 manifest"}'
    attestation = json.dumps({"layers": [{"digest": _digest_of(b"in-toto statement")}]}).encode()
    blobs[_digest_of(attestation)] = attestation  # present, its layer is not
    if entries is None:
        entries = [
            {"digest": _digest_of(amd64), "platform": {"os": "linux", "architecture": "amd64"}},
            {"digest": _digest_of(b"absent attestation"), "annotations": _ATTESTATION},
            {"digest": _digest_of(arm64), "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": _digest_of(attestation), "annotations": _ATTESTATION},
        ]
    index = json.dumps({"manifests": entries}).encode()
    blobs[_digest_of(index)] = index
    blobs.update(extra_blobs or {})
    _write_archive(path, blobs=blobs, index={"manifests": [{"digest": _digest_of(index)}]})


def test_a_partly_pulled_multi_platform_image_is_whole_for_its_own_platform(
    tmp_path: Path,
) -> None:
    """The #61 archive: amd64 and one attestation absent, arm64 whole."""
    archive = tmp_path / "ngc.tar"
    _multi_platform(archive)

    assert _archive_omits_its_own_blobs(str(archive), platform=_ARM64) is None


def test_the_same_archive_is_refused_on_a_node_of_the_absent_platform(tmp_path: Path) -> None:
    archive = tmp_path / "ngc.tar"
    _multi_platform(archive)

    omission = _archive_omits_its_own_blobs(str(archive), platform=_AMD64)

    assert omission is not None
    assert "manifest" in omission


def test_a_platform_manifest_with_a_missing_layer_is_refused(tmp_path: Path) -> None:
    """The gap the old check left: it confirmed a platform manifest existed and
    never looked at the content behind it ("checked on the next pass", which
    did not exist). A multi-platform archive with no layers passed."""
    archive = tmp_path / "hollow-arm64.tar"
    _multi_platform(archive, drop_arm64_layer=True)

    omission = _archive_omits_its_own_blobs(str(archive), platform=_ARM64)

    assert omission is not None
    assert "layer" in omission


def test_an_index_with_nothing_for_this_platform_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "amd64-only.tar"
    amd64, blobs = _image_blobs([b"amd-one"])
    _multi_platform(
        archive,
        entries=[
            {"digest": _digest_of(amd64), "platform": {"os": "linux", "architecture": "amd64"}}
        ],
        extra_blobs=blobs,
    )

    omission = _archive_omits_its_own_blobs(str(archive), platform=_ARM64)

    assert omission is not None
    assert "manifest for linux/arm64" in omission


def test_an_absent_entry_that_declares_no_platform_is_refused(tmp_path: Path) -> None:
    """Only another platform or an attestation may be absent; an entry that
    says neither is content the archive should carry."""
    archive = tmp_path / "unknown-entry.tar"
    arm64, blobs = _image_blobs([b"arm-one"])
    _multi_platform(
        archive,
        entries=[
            {"digest": _digest_of(arm64), "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": _digest_of(b"who knows")},
        ],
        extra_blobs=blobs,
    )

    omission = _archive_omits_its_own_blobs(str(archive), platform=_ARM64)

    assert omission is not None
    assert _digest_of(b"who knows") in omission


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("aarch64", _ARM64), ("arm64", _ARM64), ("x86_64", _AMD64), ("AMD64", _AMD64)],
)
def test_the_host_platform_uses_oci_architecture_names(
    monkeypatch: pytest.MonkeyPatch, machine: str, expected: tuple[str, str]
) -> None:
    import platform as host

    from tensorstead.agent.container_engine.docker_py import _local_platform

    monkeypatch.setattr(host, "machine", lambda: machine)

    assert _local_platform() == expected
