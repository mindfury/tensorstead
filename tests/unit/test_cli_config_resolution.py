"""CLI configuration resolution and its provenance.

Three environment variables had to be exported in every shell for the CLI to
work, and when a value was wrong there was no way to discover where it came
from. This suite pins the config file, the precedence order, and — most
importantly — that every value can name its own source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.cli import config

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TENSORSTEAD_API",
        "TENSORSTEAD_CA_BUNDLE",
        "TENSORSTEAD_MGMT_TOKEN",
        "TENSORSTEAD_CONFIG",
        "XDG_CONFIG_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("TENSORSTEAD_CONFIG", str(path))
    return path


def test_a_url_from_the_config_file_is_not_path_normalised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: Path() collapses "//" and produced "https:/host:8080".

    The symptom was a connection refused with a URL that looked almost right,
    which is worse than an obvious error because it reads as a network problem.
    """
    _write_config(tmp_path, monkeypatch, "coordinator: https://spark.internal:8080\n")

    assert config.api_url() == "https://spark.internal:8080"
    assert "https:/s" not in config.api_url().replace("https://", "https:XX")


def test_the_config_file_supplies_settings_with_no_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(
        tmp_path,
        monkeypatch,
        "coordinator: https://spark.internal:8080\nca_bundle: /etc/tensorstead/ca.pem\n",
    )

    api = config.resolved_api_url()
    assert api.value == "https://spark.internal:8080"
    assert str(path) in api.source
    assert config.resolved_ca_bundle().value == "/etc/tensorstead/ca.pem"


def test_the_environment_overrides_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(tmp_path, monkeypatch, "coordinator: https://from-file:8080\n")
    monkeypatch.setenv("TENSORSTEAD_API", "https://from-env:8080")

    api = config.resolved_api_url()
    assert api.value == "https://from-env:8080"
    assert api.source == "env TENSORSTEAD_API"


def test_the_default_names_itself_as_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value that sent a user to llama.cpp must not look configured."""
    monkeypatch.setenv("TENSORSTEAD_CONFIG", "/nonexistent/config.yml")

    api = config.resolved_api_url()
    assert api.value == config.DEFAULT_API
    assert api.source == "built-in default"


def test_the_token_is_read_from_the_file_the_config_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("s3cr3t\n", encoding="utf-8")
    _write_config(tmp_path, monkeypatch, f"token_file: {token}\n")

    resolved = config.resolved_token()
    assert resolved.value == "s3cr3t"
    assert str(token) in resolved.source
    assert "s3cr3t" not in resolved.source, "the source must not leak the secret"


@pytest.mark.parametrize(
    ("body", "expected"),
    [("token_file: /nope/token\n", "unreadable"), ("", "not set")],
)
def test_an_unusable_token_source_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, expected: str
) -> None:
    _write_config(tmp_path, monkeypatch, body)

    resolved = config.resolved_token()
    assert resolved.value is None
    assert expected in resolved.source


def test_an_empty_token_file_is_distinguished_from_an_absent_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("\n", encoding="utf-8")
    _write_config(tmp_path, monkeypatch, f"token_file: {token}\n")

    assert "empty" in config.resolved_token().source


def test_a_malformed_config_file_does_not_break_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken file must degrade to the default, never make the tool unusable."""
    _write_config(tmp_path, monkeypatch, "this: [is: not: valid\n")

    assert config.api_url() == config.DEFAULT_API


def test_the_config_path_is_reported_even_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "There is no config file, and here is where one goes" is the answer."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/does-not-exist")

    path = config.config_path()
    assert path.name == "config.yml"
    assert "tensorstead" in str(path)
    assert not path.is_file()


def test_verification_is_never_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No configuration may produce verify=False."""
    _write_config(tmp_path, monkeypatch, "coordinator: https://spark.internal:8080\n")
    assert config.tls_verify() is True

    _write_config(tmp_path, monkeypatch, "ca_bundle: /etc/tensorstead/ca.pem\n")
    assert config.tls_verify() == "/etc/tensorstead/ca.pem"


def test_a_flag_is_reported_as_a_flag_not_as_an_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance must name the source the operator used, not the mechanism.

    The flag is applied by exporting it, so every command resolves through one
    path. Reporting "from env TENSORSTEAD_API" would be true of the plumbing and
    false to the person who typed `--api` — and the whole point of reporting a
    source is that it matches what the operator did.
    """
    config.FLAG_SOURCED.clear()
    monkeypatch.setenv("TENSORSTEAD_API", "https://from-flag:8080")
    config.note_flag_source("TENSORSTEAD_API")
    try:
        resolved = config.resolved_api_url()
        assert resolved.value == "https://from-flag:8080"
        assert resolved.source == "command-line flag"
    finally:
        config.FLAG_SOURCED.clear()


def test_an_environment_variable_not_set_by_a_flag_still_reports_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.FLAG_SOURCED.clear()
    monkeypatch.setenv("TENSORSTEAD_API", "https://from-env:8080")

    assert config.resolved_api_url().source == "env TENSORSTEAD_API"
