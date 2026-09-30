"""Service-manager seam.

A narrow boot-restoration surface. The agent delegates restart-on-boot to the
host's service manager (systemd on Linux, launchd anticipated elsewhere); the
platform-specific mechanism never leaks into the domain or the agent routes.

The contract models the boot-restoration behaviour:

- ``install`` writes a self-sufficient unit that restores the deployment on
  boot and never references the coordinator;
- ``enable`` marks it for boot restoration while desired state is ``running``;
- ``disable`` removes boot restoration so a reboot does not start a deployment
  desired ``stopped``;
- ``remove`` deletes the unit entirely.

There is no watchdog, supervisor, or restart loop of our own — the
service manager provides restart-on-boot; nothing here polls or restarts.
"""

from __future__ import annotations

from typing import Protocol


class ServiceManager(Protocol):
    """Boot-restoration surface delegated to the host's service manager."""

    def install(self, name: str) -> None:
        """Install a self-sufficient unit for ``name``.

        The unit must restore the deployment on boot with the coordinator
        stopped or uninstalled and never reference it.
        """

    def enable(self, name: str) -> None:
        """Enable boot restoration (desired state ``running``)."""

    def disable(self, name: str) -> None:
        """Disable boot restoration so a reboot does not start it."""

    def remove(self, name: str) -> None:
        """Remove the unit entirely."""

    def is_enabled(self, name: str) -> bool:
        """Whether ``name`` is enabled for boot restoration."""

    def is_installed(self, name: str) -> bool:
        """Whether a unit for ``name`` is installed."""
