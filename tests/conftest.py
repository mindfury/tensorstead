"""Pytest tier gating.

Enforces the four safety-tiered test levels:

- unit / contract / integration: always runnable (integration uses real local Docker,
  no GPU).
- hardware:          requires ``--hardware`` (real hardware, non-disruptive).
- hardware_disruptive: requires ``--hardware-disruptive`` (reboot, multi-node teardown)
  as a separate, additional opt-in, so it can never run by accident.

Verification stays a first-class concern; these gates keep stateful, shared
hardware hosts safe while letting the fast tiers run without ceremony.
"""

from collections.abc import Iterator

import pytest

# Markers registered in pyproject.toml [tool.pytest.ini_options] markers.
_HARDWARE = {"hardware", "hardware_disruptive"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="run non-disruptive tests against real hardware (opt-in)",
    )
    parser.addoption(
        "--hardware-disruptive",
        action="store_true",
        default=False,
        help="run disruptive tests against real hardware (opt-in, additional)",
    )


def _opt_in_flags(config: pytest.Config) -> tuple[bool, bool]:
    return bool(config.getoption("--hardware")), bool(config.getoption("--hardware-disruptive"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    hardware_on, hardware_disruptive_on = _opt_in_flags(config)
    skip_hardware = pytest.mark.skip(reason="hardware tests require --hardware (opt-in)")
    skip_disruptive = pytest.mark.skip(
        reason="hardware_disruptive tests require --hardware-disruptive (separate opt-in)"
    )

    for item in items:
        markers = {mark.name for mark in item.iter_markers()}
        has_disruptive = "hardware_disruptive" in markers
        has_hardware = "hardware" in markers

        if has_disruptive:
            if not hardware_disruptive_on:
                item.add_marker(skip_disruptive)
            continue
        if has_hardware and not hardware_on:
            item.add_marker(skip_hardware)


@pytest.fixture(autouse=True)
def _isolate_operator_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the suite away from the machine's real Tensorstead configuration.

    The CLI reads ``~/.config/tensorstead/config.yml`` when no environment
    variable overrides it. Without this fixture the suite passes or fails
    depending on whether the developer happens to have run ``stead login``,
    which is exactly the kind of hidden dependency that makes a green run
    meaningless. Pointing at a path inside a temporary directory keeps every
    test resolving from a known-absent file.
    """
    absent = tmp_path_factory.mktemp("no-operator-config") / "config.yml"
    monkeypatch.setenv("TENSORSTEAD_CONFIG", str(absent))


@pytest.fixture(autouse=True)
def _close_helper_connections() -> Iterator[None]:
    """Close SQLite connections the test helpers opened (test hygiene only).

    The coordinator itself opens **one** connection for the process lifetime,
    which is the right shape for a long-running service and is not what this
    is about. The helpers open one per constructed app and nothing closed them,
    so CPython eventually reported each as an unclosed resource -- attributed to
    whichever test was running when the collector fired, which was reliably an
    innocent one.
    """
    yield
    from tests.helpers import close_test_connections

    close_test_connections()
