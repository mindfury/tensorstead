"""No-colocation-assumption guardrail.

Asserts that no code path assumes the coordinator and an agent share a host,
and none assumes they do not. Concretely:

- **no localhost defaults** for the agent endpoint when the coordinator decides
  where an agent lives — every coordinator→agent call is over the contract via
  the node's recorded ``agent_endpoint``;
- **no shared filesystem path** is used by the coordinator to reach a node's
  model store or container runtime (there is no in-process shortcut);
- every coordinator→agent call goes over the same contract either way.

This is verified structurally over the source tree: the coordinator code must
not default to ``localhost``/``127.0.0.1`` for an *agent* target, must not
reach into ``/var/lib/...`` or a Docker socket path directly, and must route all
agent work through the node-client port.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract


_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_COORDINATOR = _SRC / "coordinator"

# Agent endpoint must come from the node record, never a hardcoded loopback.
_FORBIDDEN_AGENT_DEFAULTS = ("127.0.0.1", "localhost", "::1")

# A shared-filesystem shortcut would reach a node's store or socket directly.
_FORBIDDEN_SHARED_PATHS = ("/var/lib/tensorstead", "/var/lib/docker", "/var/run/docker.sock")


def _walk_py(root: Path) -> list[Path]:
    """Every Python file under ``root``, ``__init__.py`` included.

    Package initialisers are deliberately *not* skipped — see the note in
    ``test_no_data_path._walk_py``: the coordinator's whole route surface lives
    in ``coordinator/routes/__init__.py``, which is precisely where a colocation
    assumption would be introduced.
    """
    return sorted(root.rglob("*.py"))


def test_no_localhost_default_for_agent_endpoint() -> None:
    """The coordinator never defaults an agent target to a loopback address.

    Every coordinator→agent call must go to the node's recorded
    ``agent_endpoint``. A hardcoded loopback
    would silently assume colocation.
    """
    for path in _walk_py(_COORDINATOR):
        text = path.read_text()
        for token in _FORBIDDEN_AGENT_DEFAULTS:
            assert token not in text, (
                f"{path.relative_to(_SRC)} hardcodes {token!r} as an agent target, "
                f"assuming colocation"
            )


def test_no_shared_filesystem_shortcut() -> None:
    """The coordinator never reaches a node's store or runtime via a local path.

    Model weights and container images live on the nodes; the
    coordinator owns no shared filesystem with them. Any direct local path into
    a node store or the Docker socket would assume colocation and bypass the
    contract.
    """
    for path in _walk_py(_COORDINATOR):
        text = path.read_text()
        for token in _FORBIDDEN_SHARED_PATHS:
            assert token not in text, (
                f"{path.relative_to(_SRC)} references {token!r}, a shared-filesystem "
                f"shortcut that assumes colocation"
            )


def test_all_agent_work_routes_through_node_client() -> None:
    """Coordinator→agent communication goes over the node-client contract.

    The node client is the only way the product acts on a node.
    This asserts the coordinator's agent-facing
    modules do not import an HTTP client directly to reach an agent.
    """
    for path in _walk_py(_COORDINATOR):
        if path.name == "node_http.py":
            continue  # the node client itself owns the transport
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "httpx":
                        raise AssertionError(
                            f"{path.relative_to(_SRC)} imports httpx directly; "
                            f"coordinator→agent calls must go through the node client "
                            f""
                        )
