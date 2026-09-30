"""Hardware test: reboot-restoration acceptance.

With the coordinator stopped, deployments desiring ``running`` are
serving after a reboot and those desiring ``stopped`` do not start.
This is a **hardware-tier** test: it requires a real
host with systemd and a container engine, and is behind the
``hardware`` opt-in flag.

The test:
1. Creates a deployment and starts it (desired state ``running``).
2. Creates a second deployment and leaves it stopped.
3. Stops the coordinator process.
4. Simulates a reboot by restarting the agent's host (manual step or
   release-gated).
5. Verifies the ``running`` deployment is serving and the ``stopped``
   deployment is not.

This test is skipped by default. To run it, pass ``--hardware`` to
pytest and ensure a real host is configured via ``TENSORSTEAD_TEST_NODE``.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not os.environ.get("TENSORSTEAD_HARDWARE_ENABLED"),
        reason="hardware tests require TENSORSTEAD_HARDWARE_ENABLED=1",
    ),
]


def test_reboot_restoration_running_and_stopped() -> None:
    """Reboot restores running deployments and not stopped ones.

    This test is a placeholder for the manual release-gated procedure
    described in the spec. The automated precondition is that the agent
    is running on a real host with systemd. The full procedure is
    documented in ``docs/first-deployment.md`` and the release checklist.
    """
    pytest.skip("reboot-restoration is a manual release-gated acceptance test")
