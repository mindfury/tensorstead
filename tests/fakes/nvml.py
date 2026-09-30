"""Fake NVML and filesystem-capacity source.

Models on-demand accelerator and storage readings so
resource tests need no NVIDIA GPU and no real filesystem (tier 1). The real
source is ``nvidia-ml-py`` (NVML) plus a filesystem ``statvfs`` read; the fake
returns configurable readings on demand and keeps no history.

``memory_is_unified`` is reported (never a discrete-VRAM assumption) because on
GB10 the accelerator-memory figure is a view onto shared memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FakeStorageReading:
    purpose: str
    path: str
    capacity_bytes: int
    available_bytes: int


class FakeNVMLSource:
    """A stateful in-memory accelerator + storage source.

    ``read()`` returns the current readings on demand. Calling it records a read
    (for tests asserting no sampler/no history) but never writes.
    """

    def __init__(
        self,
        *,
        utilization_pct: float = 0.0,
        memory_used: int = 0,
        memory_total: int = 0,
        memory_is_unified: bool = True,
        storage: list[FakeStorageReading] | None = None,
    ) -> None:
        self.utilization_pct = utilization_pct
        self.memory_used = memory_used
        self.memory_total = memory_total
        self.memory_is_unified = memory_is_unified
        self.storage = storage or []
        self.read_count = 0

    def read(self) -> dict:
        """Return an on-demand reading. Records the read; never writes to a store."""
        self.read_count += 1
        return {
            "status": "ok",
            "observed_at": datetime.now().astimezone().isoformat(),
            "accelerator_utilization_pct": self.utilization_pct,
            "accelerator_memory_used": self.memory_used,
            "accelerator_memory_total": self.memory_total,
            "memory_is_unified": self.memory_is_unified,
            "storage": [
                {
                    "purpose": s.purpose,
                    "path": s.path,
                    "capacity_bytes": s.capacity_bytes,
                    "available_bytes": s.available_bytes,
                }
                for s in self.storage
            ],
        }
