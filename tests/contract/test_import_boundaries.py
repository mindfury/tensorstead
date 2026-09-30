"""Import-boundary guardrail.

Asserts that the platform-neutral core domain imports nothing platform-specific,
and that ``systemd`` appears only under the agent's service-manager
implementation. This is verified as an *import-boundary test*, not a convention.

The platform-neutral core is ``src/tensorstead/domain/``. It must not import any
module that is OS- or vendor-specific: no ``platform``, ``docker``,
``systemd``, ``nvidia``, ``huggingface``. Only the agent's service-manager
implementation may mention ``systemd``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

# Modules that would couple the domain to a specific platform or vendor.
_PLATFORM_SPECIFIC_IMPORTS = {
    "platform",  # os.family etc. — domain must stay neutral
    "docker",
    "systemd",
    "subprocess",
    "nvidia",
    "huggingface",
}

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_DOMAIN = _SRC / "domain"


def _imports_in_tree(root: Path) -> set[str]:
    """Return the set of top-level modules imported anywhere under ``root``.

    ``__init__.py`` is included: a package initialiser is as capable of
    importing ``docker`` as any other module, and re-exports commonly live
    there.
    """
    names: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def test_domain_imports_nothing_platform_specific() -> None:
    """The core domain package imports nothing platform-specific."""
    imports = _imports_in_tree(_DOMAIN)
    offending = imports & _PLATFORM_SPECIFIC_IMPORTS
    assert not offending, (
        f"domain/ imports platform-specific modules: {sorted(offending)} "
        f"(the core must stay platform-neutral)"
    )


def test_systemd_appears_only_under_agent_service_manager() -> None:
    """The systemd *mechanism* appears only under the agent's service manager.

    This rule confines platform-specific mechanism — importing a systemd
    module or invoking ``systemctl`` — to ``agent/service_manager/``. Reporting
    the service-manager *name* in ``GET /agent/v1/info`` is not a coupling to
    systemd's mechanism, so it is permitted; what is forbidden is any code path
    that drives systemd (systemctl / systemd unit API) elsewhere.
    """
    for path in sorted(_SRC.rglob("*.py")):
        if "service_manager" in path.parts:
            continue  # the one place systemd mechanism is allowed
        text = path.read_text()
        # Importing a systemd module, or invoking systemctl, would couple a
        # non-service-manager module to the platform mechanism.
        assert "import systemd" not in text, (
            f"{path.relative_to(_SRC)} imports a systemd module outside the "
            f"agent service-manager implementation"
        )
        assert "systemctl" not in text, (
            f"{path.relative_to(_SRC)} invokes systemctl outside the "
            f"agent service-manager implementation"
        )
