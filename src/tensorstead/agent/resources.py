"""On-demand agent resource reads.

The agent reads accelerator, memory, and managed-storage figures on demand.
There is **no sampler and no history retained**: a reading is
taken when the coordinator asks and never otherwise. The figures are reported
without being acted on — nothing here refuses, defers, or reserves
work.

Two sources are read:

- **NVML** (``nvidia-ml-py``) for accelerator utilization and memory. On GB10
  the accelerator-memory figure is a view onto shared CPU/GPU memory, so
  ``memory_is_unified`` is reported rather than a discrete-VRAM model being
  assumed.
- **Filesystem** ``statvfs`` for the model store and image store — the
  locations the product manages. Two locations may resolve to
  the same filesystem, in which case they report the same figures — that is
  reported as observed rather than deduplicated.

Both are injected via ``app.state`` so tests supply the fakes.
When a source is unavailable (no GPU, no NVML, no managed path), the reading
degrades to ``unknown`` for that field rather than failing the whole request.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class NVMLSource(Protocol):
    """The accelerator-reading seam (fake in tests)."""

    def read(self) -> dict[str, Any]: ...


class FilesystemSource(Protocol):
    """The managed-storage-reading seam (fake in tests)."""

    def read(self, path: str) -> dict[str, Any]: ...


class HostMemorySource(Protocol):
    """The unified-memory reading seam (fake in tests)."""

    def read(self) -> dict[str, int]: ...


_UNREAD: dict[str, Any] = {
    "accelerator_utilization_pct": None,
    "accelerator_memory_used": None,
    "accelerator_memory_total": None,
}


class _RealNVMLSource:
    """NVML-backed accelerator reads. Degrades per field, never wholesale.

    Utilization and memory are read in **separate** attempts on purpose. They
    are not equally available: on GB10 the accelerator's memory is the host's
    unified memory, and NVML's discrete-VRAM query may be unsupported there
    while ``nvmlDeviceGetUtilizationRates`` answers perfectly well. Reading
    both under one ``try`` meant a single unsupported call discarded a figure
    that had already been obtained, and reported the node as having told us
    nothing when in fact it told us half.

    Where NVML cannot supply memory on a unified-memory platform, the host's
    own memory accounting is the correct source rather than a fallback: on
    that hardware system memory *is* accelerator memory, which is why NVIDIA's
    own dashboard shows this figure as "System Memory".
    """

    def read(self) -> dict[str, Any]:
        try:
            import pynvml  # type: ignore[import-untyped]
        except ImportError:
            return dict(_UNREAD)

        reading = dict(_UNREAD)
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return reading

        with contextlib.suppress(Exception):
            reading["accelerator_utilization_pct"] = float(
                pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            )

        with contextlib.suppress(Exception):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            reading["accelerator_memory_used"] = int(memory.used)
            reading["accelerator_memory_total"] = int(memory.total)

        return reading


class _HostMemorySource:
    """Host memory accounting, for platforms whose accelerator memory is unified.

    Reads ``/proc/meminfo`` because that is what the figure actually is on such
    a platform. ``MemTotal`` less ``MemAvailable`` is used for "used" rather
    than ``MemFree``: page cache is reclaimable, and counting it as consumed
    would misreport a healthy node as nearly full.
    """

    MEMINFO = Path("/proc/meminfo")

    def read(self) -> dict[str, int]:
        try:
            fields: dict[str, int] = {}
            for line in self.MEMINFO.read_text(encoding="utf-8").splitlines():
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    fields[key] = int(rest.split()[0]) * 1024
            total, available = fields.get("MemTotal"), fields.get("MemAvailable")
            if total is None or available is None:
                return {}
            return {"used": total - available, "total": total}
        except (OSError, ValueError, IndexError):
            return {}


class _RealFilesystemSource:
    """``statvfs``-backed managed-storage reads."""

    def read(self, path: str) -> dict[str, Any]:
        try:
            stat = os.statvfs(path)
            return {
                "capacity_bytes": stat.f_blocks * stat.f_frsize,
                "available_bytes": stat.f_bavail * stat.f_frsize,
            }
        except OSError:
            return {"capacity_bytes": 0, "available_bytes": 0}


def read_resources(
    *,
    nvml_source: NVMLSource | None = None,
    filesystem_source: FilesystemSource | None = None,
    host_memory_source: HostMemorySource | None = None,
    model_store_path: str = "/var/lib/tensorstead/models",
    image_store_path: str = "/var/lib/docker",
    cache_path: str | None = None,
    memory_is_unified: bool = False,
) -> dict[str, Any]:
    """Take an on-demand reading; no history retained.

    Returns the resource observation as a dict the agent route and the
    coordinator's ``NodeResourcesResponse`` both render.

    ``status`` is ``ok`` when at least one source answered and ``unknown`` when
    none did; ``unreachable`` is set by the caller when the node itself cannot
    be contacted. This is what the docstring has always claimed and what the
    code did not do — it returned ``ok`` unconditionally, so a node that told
    us nothing was indistinguishable from an idle one.

    ``unreadable_fields`` names what was missed. It is deliberately *not* part
    of the agent contract: reporting partial readings as a distinct status
    would widen ``NodeResourceObservationResponse.status``, which is a contract
    change requiring a version bump and a specification rather than a repair.
    It is carried here so callers and tests can see the gap without inferring
    it from nulls.
    """
    nvml = nvml_source or _RealNVMLSource()
    fs = filesystem_source or _RealFilesystemSource()

    gpu = dict(nvml.read())

    # On a unified-memory platform the accelerator's memory is the host's, so
    # the host is the authoritative source rather than a substitute for one.
    if memory_is_unified and gpu.get("accelerator_memory_total") is None:
        host = (host_memory_source or _HostMemorySource()).read()
        if host:
            gpu["accelerator_memory_used"] = host["used"]
            gpu["accelerator_memory_total"] = host["total"]

    model_store = fs.read(model_store_path)
    image_store = fs.read(image_store_path)

    storage = [
        {"purpose": "models", "path": model_store_path, **model_store},
        {"purpose": "images", "path": image_store_path, **image_store},
    ]
    # The compile cache is managed storage like the other two, so the disk a
    # runtime's compiled artefacts occupy is visible and reclaimable rather than
    # accumulating somewhere nobody looks. Omitted rather than
    # reported as zero when the caller does not supply a path: an absent
    # location and an empty one are different facts.
    if cache_path:
        storage.append({"purpose": "cache", "path": cache_path, **fs.read(cache_path)})

    accelerator_fields = (
        "accelerator_utilization_pct",
        "accelerator_memory_used",
        "accelerator_memory_total",
    )
    unreadable = [name for name in accelerator_fields if gpu.get(name) is None]
    if not any(entry.get("capacity_bytes") for entry in storage):
        unreadable.append("storage")

    # "unknown" only when nothing at all was readable; the contract's status
    # vocabulary is ok/unknown/unreachable and this repair does not widen it.
    status = "unknown" if len(unreadable) == len(accelerator_fields) + 1 else "ok"

    return {
        "status": status,
        "observed_at": datetime.now().astimezone().isoformat(),
        "accelerator_utilization_pct": gpu.get("accelerator_utilization_pct"),
        "accelerator_memory_used": gpu.get("accelerator_memory_used"),
        "accelerator_memory_total": gpu.get("accelerator_memory_total"),
        "memory_is_unified": memory_is_unified,
        "storage": storage,
        "unreadable_fields": unreadable,
    }


def default_managed_paths() -> tuple[str, str]:
    """Return the default managed-storage paths from the environment.

    Overridable so tests and non-standard installs point at the right
    locations without hard-coding them into the reading.
    """
    models = os.environ.get("TENSORSTEAD_MODEL_STORE", "/var/lib/tensorstead/models")
    images = os.environ.get("TENSORSTEAD_IMAGE_STORE", "/var/lib/docker")
    return models, images


def default_cache_path() -> str:
    """The managed compile-cache root.

    Reported as managed storage alongside models and images so the disk a
    runtime's compiled artefacts occupy is visible and reclaimable rather than
    accumulating somewhere nobody looks.
    """
    return os.environ.get("TENSORSTEAD_CACHE_PATH", "/var/lib/tensorstead/cache")
