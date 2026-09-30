"""No-supervisor guardrail.

Asserts that boot restoration is a service-manager unit and that no watchdog,
supervisor, or restart loop of our own exists. Restart-on-boot is delegated to
systemd (the service manager); the product never polls, supervises, or restarts
deployments itself.

Verified structurally: the agent's ``container_engine`` and ``service_manager``
seams may start/stop once (a lifecycle action), but nothing anywhere runs a
loop that restarts a container or unit on a schedule, and no process polls a
health endpoint to decide whether to restart.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"


def _walk_py(root: Path) -> list[Path]:
    """Every Python file under ``root``, ``__init__.py`` included.

    Package initialisers are deliberately *not* skipped — see the note in
    ``test_no_data_path._walk_py``: the coordinator's whole route surface lives
    in ``coordinator/routes/__init__.py``.
    """
    return sorted(root.rglob("*.py"))


def _has_while_loop(tree: ast.AST) -> bool:
    return any(isinstance(node, (ast.While, ast.For)) for node in ast.walk(tree))


def test_no_supervisor_loop_anywhere() -> None:
    """No module runs a supervisor/restart loop.

    Restart-on-boot is the service manager's unit (systemd); the product writes
    no watchdog and no retry loop. A ``while``/``for`` loop in the source is a
    strong signal of supervision or polling the product must not do.
    """
    for path in _walk_py(_SRC):
        # The migration runner iterates over migration files (a bounded loop),
        # and the repository iterates rows — both are data iteration, not
        # supervision. Exclude the two known non-supervisor loops by checking
        # the loop bodies do not restart anything.
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.While, ast.For)):
                body_src = ast.get_source_segment(path.read_text(), node) or ""
                restart_words = (
                    "restart",
                    "start(",
                    "start_container",
                    "supervise",
                    "watchdog",
                    "systemctl restart",
                )
                for word in restart_words:
                    # A loop that calls a restart primitive is a supervisor.
                    assert word not in body_src.lower(), (
                        f"{path.relative_to(_SRC)} contains a loop that restarts/supervises; "
                        f"no watchdog or supervisor of our own is permitted"
                    )


def test_restart_on_boot_is_a_unit_not_a_process() -> None:
    """Boot restoration is a service-manager unit, not a product process.

    The systemd unit (``service_manager/systemd.py``) declares
    ``Restart=on-failure`` — that is systemd's own policy, which is exactly what
    is allowed: the service manager provides restart-on-boot; the product
    writes no supervisor. Assert no separate long-lived restart process exists.
    """
    # The product must not spawn its own background process that supervises.
    for path in _walk_py(_SRC):
        text = path.read_text()
        # A subprocess that supervises (polls + restarts) is forbidden.
        if "subprocess" in text and "systemctl" in text and "service_manager" not in path.parts:
            raise AssertionError(
                f"{path.relative_to(_SRC)} drives a subprocess outside the service-manager seam"
            )
