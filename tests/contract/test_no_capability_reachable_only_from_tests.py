"""No service capability is reachable only from the test suite.

This exists because the same defect appeared four times in one working session,
and three of those were the same shape: **something built, tested, and called
by nothing.**

- `unexpected_instance` asked the engine for an attribute only the test fake
  had, so on real hardware it answered false every time. The guarantee
  reported nothing in production and the suite was green.
- The API, CLI, and MCP surfaces existed while the specs index said they
  did not, so the work was believed outstanding and would have been done twice.
- `ImageBuildService.build_and_distribute` — the produce-once-then-copy
  orchestration that a tensor-parallel deployment requires — was written,
  unit-tested, and reached by no route, CLI command, or MCP tool for the whole
  life of the feature.

`make deadcode` should have caught the third. It cannot: vulture scans `src`,
`tests`, and `scripts` together, so a method the tests call looks used. That is
precisely the hiding place — a capability with tests *looks* healthier than one
without, and the tests are what conceal it.

So this asks a different question: is anything in the service layer reachable
from the product itself? A method called only by tests is a capability no
operator can use, however well tested.

The allowlist is not a suppression list. Every entry names why it is there, and
the ones inherited from the earliest work are already tracked as tasks — this test makes
them visible on every run instead of only when someone reads a Makefile
comment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_SERVICE = _SRC / "service"

# Known and tracked, each with the reason it is not a bug in this test.
#
# These are *not* "fine" -- listing one here keeps it visible rather than
# resolved, and the list is meant to shrink. (The duplicates --
# ``ObservationService.node_resources`` and ``NodeService.get_by_name`` -- were
# removed and left this list; they are no longer exempt because they no longer
# exist. ``check_revision`` was removed 2026-08-12 when the optimistic
# check was finally wired through the PATCH route: this guardrail named that
# defect for weeks, and an external audit found it before we acted on our own
# record of it.)
_KNOWN_UNREACHED: dict[str, str] = {
    "export": "the export route calls `as_dict`; this entry point has no caller",
    "set_progress": "called only by `report_progress`, which itself has no caller",
}


def _public_service_methods() -> dict[str, str]:
    """Public methods defined on classes in the service layer, by defining file."""
    defined: dict[str, str] = {}
    for path in sorted(_SERVICE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    defined.setdefault(item.name, path.name)
    return defined


def test_no_service_capability_is_reachable_only_from_tests() -> None:
    """A capability the product cannot reach is a capability no operator has."""
    sources = {path: path.read_text() for path in _SRC.rglob("*.py")}
    unreached: list[str] = []

    for name, origin in sorted(_public_service_methods().items()):
        called = any(
            path.name != origin and re.search(rf"\.{re.escape(name)}\s*\(", text)
            for path, text in sources.items()
        )
        if not called and name not in _KNOWN_UNREACHED:
            unreached.append(f"{origin}:{name}")

    assert not unreached, (
        "service capabilities reachable from no route, CLI, or MCP surface — "
        f"only from tests: {unreached}. A capability with tests looks healthier "
        "than one without, and the tests are what hide it. Either wire it to a "
        "surface or add it to _KNOWN_UNREACHED with the reason."
    )


def test_the_allowlist_does_not_outlive_its_entries() -> None:
    """An allowlist entry for something now wired is a stale exemption.

    The failure mode this prevents is the one that produced the test above: a
    list nobody rereads, quietly excusing a problem that was fixed or a problem
    that grew.
    """
    sources = {path: path.read_text() for path in _SRC.rglob("*.py")}
    defined = _public_service_methods()
    stale: list[str] = []

    for name in _KNOWN_UNREACHED:
        if name not in defined:
            stale.append(f"{name} (no longer defined in the service layer)")
            continue
        origin = defined[name]
        called = any(
            path.name != origin and re.search(rf"\.{re.escape(name)}\s*\(", text)
            for path, text in sources.items()
        )
        if called:
            stale.append(f"{name} (now reachable; remove the exemption)")

    assert not stale, f"stale entries in _KNOWN_UNREACHED: {stale}"
