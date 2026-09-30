"""No-self-installation guardrail.

Asserts that no code path installs, upgrades, or removes the product's own
node-side software. Registration verifies and records; a node whose agent is
absent fails with ``agent_unreachable`` rather than triggering a bootstrap.

Verified structurally: the coordinator never installs/upgrades/removes an
agent — there is no package-management, service-install, or binary-drop code
path in the product. The agent is installed by external tooling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"

# Mechanisms that would install/upgrade/remove the product's own node-side
# software — none may appear outside the allowed install-time paths.
_SELF_INSTALL_MECHANISMS = (
    "pip install",
    "apt-get",
    "apt install",
    "dnf install",
    "yum install",
    "curl -o /usr/local/bin",
    "wget -O /usr/local/bin",
    "install -m",
    "systemctl enable tensorstead-agent",
    "systemctl install",
)


def _walk_py(root: Path) -> list[Path]:
    """Every Python file under ``root``, ``__init__.py`` included.

    Package initialisers are deliberately *not* skipped — see the note in
    ``test_no_data_path._walk_py``: the coordinator's whole route surface lives
    in ``coordinator/routes/__init__.py``.
    """
    return sorted(root.rglob("*.py"))


def test_no_self_installation_mechanism() -> None:
    """No code path installs, upgrades, or removes the agent.

    The node agent is a prerequisite installed by external tooling. The product
    verifies it is running at registration and records it — it never bootstraps.
    """
    for path in _walk_py(_SRC):
        text = path.read_text()
        for mechanism in _SELF_INSTALL_MECHANISMS:
            assert mechanism not in text, (
                f"{path.relative_to(_SRC)} contains {mechanism!r}, a "
                f"self-installation mechanism; registration verifies and records, "
                f"it never bootstraps the agent"
            )


def test_registration_verifies_never_bootstraps() -> None:
    """The node service verifies reachability, never installs the agent.

    A node whose agent is absent must fail with ``agent_unreachable``, not
    trigger a bootstrap. The service layer's ``register`` calls
    ``get_info`` to verify; it has no install/upgrade path.
    """
    nodes = _SRC / "service" / "nodes.py"
    text = nodes.read_text()
    # Registration performs a verification call to the agent.
    assert "get_info" in text
    # No install/upgrade *mechanism* in the node service. The word "bootstrap"
    # may appear in a docstring describing what the code does *not* do; what is
    # forbidden is the mechanism itself.
    banned = ("subprocess", "pip install", "import docker", "os.system", "systemctl install")
    for banned_mechanism in banned:
        assert banned_mechanism not in text, "nodes.py must not install/upgrade an agent"
