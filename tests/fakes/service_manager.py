"""Fake service manager.

Models the agent-side service-manager seam — systemd on Linux,
launchd anticipated on macOS but not implemented. The real backend writes and
enables systemd units; the fake stands in so boot-restoration logic is testable
with no systemd (tier 1).

The key behaviour modelled is the boot-restoration contract: a unit
is installed and *enabled* when desired state is ``running``, and disabled and
*removed* when ``stopped``, so a reboot restores the running deployment and does
not start a stopped one. The fake never references the coordinator
and there is no watchdog of our own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeUnit:
    name: str
    installed: bool = False
    enabled: bool = False


class FakeServiceManager:
    """A stateful in-memory service manager implementing the narrow seam."""

    def __init__(self) -> None:
        self.units: dict[str, FakeUnit] = {}

    def install(self, name: str) -> None:
        """Install a unit file (self-sufficient; never references the coordinator)."""
        self.units.setdefault(name, FakeUnit(name=name)).installed = True

    def enable(self, name: str) -> None:
        """Enable for boot restoration."""
        unit = self.units.setdefault(name, FakeUnit(name=name))
        unit.enabled = True
        unit.installed = True

    def disable(self, name: str) -> None:
        """Disable so a reboot does not start it."""
        unit = self.units.get(name)
        if unit:
            unit.enabled = False

    def remove(self, name: str) -> None:
        """Remove the unit entirely."""
        self.units.pop(name, None)

    def is_enabled(self, name: str) -> bool:
        unit = self.units.get(name)
        return bool(unit and unit.enabled)

    def is_installed(self, name: str) -> bool:
        unit = self.units.get(name)
        return bool(unit and unit.installed)
