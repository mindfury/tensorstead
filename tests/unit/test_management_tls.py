"""Management transport security.

The coordinator's management API carried the token in cleartext on a broadly
bound listener. These tests pin the repair, including the negative property
that matters most: no client may disable certificate verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tensorstead.cli import config
from tensorstead.cli.coordinator_cmds import _is_loopback, coordinator_app

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),  # the whole 127.0.0.0/8 block is loopback, not one address
        ("127.255.255.255", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("10.0.0.11", False),
        ("spark-alpha.internal", False),
    ],
)
def test_is_loopback(host: str, expected: bool) -> None:
    assert _is_loopback(host) is expected


def test_ca_bundle_is_used_as_the_verify_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENSORSTEAD_CA_BUNDLE", "/etc/tensorstead/ca.pem")
    assert config.tls_verify() == "/etc/tensorstead/ca.pem"


def test_verification_stays_on_when_no_bundle_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absence of a private CA must not mean absence of verification."""
    monkeypatch.delenv("TENSORSTEAD_CA_BUNDLE", raising=False)
    assert config.tls_verify() is True


def test_blank_bundle_is_treated_as_unset_not_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORSTEAD_CA_BUNDLE", "   ")
    assert config.tls_verify() is True


def test_no_client_path_can_disable_verification() -> None:
    """Checked structurally rather than by reading the call sites."""
    roots = [
        Path("src/tensorstead/cli"),
        Path("src/tensorstead/mcp"),
        Path("src/tensorstead/coordinator"),
    ]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "verify=False" in text or "verify = False" in text:
                offenders.append(str(path))
    assert offenders == [], f"certificate verification is disabled in: {offenders}"


def test_serve_refuses_a_certificate_without_its_key(tmp_path: Path) -> None:
    """Half a TLS configuration is a misconfiguration, not a default."""
    cert = tmp_path / "server.pem"
    cert.write_text("not-a-real-certificate\n")
    result = CliRunner().invoke(
        coordinator_app,
        ["serve", "--store", str(tmp_path / "x.db"), "--tls-cert", str(cert)],
    )
    assert result.exit_code != 0
    assert "must be supplied together" in result.output


def test_serve_refuses_listening_off_host_without_tls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse, don't warn and still serve.

    The prior behavior (warn on stderr, serve anyway, exit 0) meant a fresh
    operator's first ``serve`` with no environment configured got a fully
    open, cleartext management API and, at best, a line on stderr nothing
    was necessarily watching.
    """
    served: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        served.update(kwargs)

    monkeypatch.setattr("tensorstead.cli.coordinator_cmds.uvicorn.run", fake_run)
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test-token")
    result = CliRunner().invoke(
        coordinator_app,
        ["serve", "--host", "0.0.0.0", "--store", str(tmp_path / "y.db")],
    )

    assert result.exit_code != 0
    assert "without TLS" in result.output
    assert served == {}, "the server must not have started"


def test_serve_refuses_an_off_host_bind_with_no_token_even_with_tls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TLS protects the token in transit; it is not a substitute for having one.

    An off-loopback, TLS-enabled listener with no token configured still
    accepts every request from anything that can reach it.
    """
    served: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        served.update(kwargs)

    monkeypatch.setattr("tensorstead.cli.coordinator_cmds.uvicorn.run", fake_run)
    monkeypatch.delenv("TENSORSTEAD_MGMT_TOKEN", raising=False)
    cert, key = tmp_path / "c2.pem", tmp_path / "k2.pem"
    cert.write_text("c\n")
    key.write_text("k\n")

    result = CliRunner().invoke(
        coordinator_app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--store",
            str(tmp_path / "y2.db"),
            "--tls-cert",
            str(cert),
            "--tls-key",
            str(key),
        ],
    )

    assert result.exit_code != 0
    assert "TENSORSTEAD_MGMT_TOKEN" in result.output
    assert served == {}


def test_serve_refuses_loopback_with_no_token_and_no_dev_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deny by default: even the safe host still needs an explicit opt-in."""
    monkeypatch.delenv("TENSORSTEAD_MGMT_TOKEN", raising=False)

    result = CliRunner().invoke(
        coordinator_app,
        ["serve", "--store", str(tmp_path / "y3.db")],
    )

    assert result.exit_code != 0
    # Not the full flag text: Click's rich error rendering wraps and
    # per-token-colorizes long option names, splitting "--insecure-dev-mode"
    # across ANSI codes and a line break in the captured output.
    assert "TENSORSTEAD_MGMT_TOKEN is not set" in result.output
    assert "insecure" in result.output


def test_serve_allows_loopback_with_no_token_given_the_dev_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one, narrowly-scoped, conspicuous escape hatch."""
    served: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        served.update(kwargs)

    monkeypatch.setattr("tensorstead.cli.coordinator_cmds.uvicorn.run", fake_run)
    monkeypatch.delenv("TENSORSTEAD_MGMT_TOKEN", raising=False)

    result = CliRunner().invoke(
        coordinator_app,
        ["serve", "--store", str(tmp_path / "y4.db"), "--insecure-dev-mode"],
    )

    assert result.exit_code == 0, result.output
    assert served["host"] == "127.0.0.1"


def test_insecure_dev_mode_does_not_override_the_off_loopback_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag's own help text promises this; prove it rather than trust it."""
    served: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        served.update(kwargs)

    monkeypatch.setattr("tensorstead.cli.coordinator_cmds.uvicorn.run", fake_run)
    monkeypatch.delenv("TENSORSTEAD_MGMT_TOKEN", raising=False)

    result = CliRunner().invoke(
        coordinator_app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--store",
            str(tmp_path / "y5.db"),
            "--insecure-dev-mode",
        ],
    )

    assert result.exit_code != 0
    assert served == {}


def test_serve_passes_tls_material_to_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        served.update(kwargs)

    monkeypatch.setattr("tensorstead.cli.coordinator_cmds.uvicorn.run", fake_run)
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test-token")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("c\n")
    key.write_text("k\n")

    result = CliRunner().invoke(
        coordinator_app,
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--store",
            str(tmp_path / "z.db"),
            "--tls-cert",
            str(cert),
            "--tls-key",
            str(key),
        ],
    )

    assert result.exit_code == 0
    assert served["ssl_certfile"] == str(cert)
    assert served["ssl_keyfile"] == str(key)
    # No cleartext warning when TLS is configured.
    assert "cleartext" not in result.output
