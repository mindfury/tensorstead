"""Unit test credential defaulting.

Three properties, each of which is a rule an operator would otherwise have to
hold in their head:

- **Exactly one default per source**. Setting a credential as default
  demotes whichever one held it; the invariant is maintained by the service, not
  asserted in a comment.
- **An acquisition naming none uses the default** — this is how
  "applied automatically" is actually satisfied. Naming is optional and
  additive.
- **Deleting the default reports the consequence** rather than
  silently reassigning it. A source left with no default says so; a promotion
  happens only because it was asked for.

The provider under test is the real local-file provider, not a stub, so
"the store holds a reference, never a value" is exercised end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider
from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.errors import NotFoundError
from tensorstead.service.credentials import CredentialService

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)


@pytest.fixture
def service(tmp_path: Path) -> CredentialService:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repository = SQLiteRepository(conn)
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    return CredentialService(repository, provider)


# ------------------------------------------ exactly one default per source
def test_first_credential_for_a_source_becomes_the_default(service: CredentialService) -> None:
    """A source's first credential is its default — never zero defaults."""
    service.set("huggingface", "personal", secret="hf_personal")

    credentials = service.list()
    assert [(c.source_id, c.name, c.is_default) for c in credentials] == [
        ("huggingface", "personal", True)
    ]


def test_exactly_one_default_per_source(service: CredentialService) -> None:
    """Promoting a second credential demotes the first."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org", default=True)

    defaults = [c.name for c in service.list() if c.is_default]
    assert defaults == ["org"], "exactly one credential per source may be default"


def test_additional_non_default_credential_leaves_the_default_alone(
    service: CredentialService,
) -> None:
    """Naming is additive — adding one does not move the default."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org")

    defaults = [c.name for c in service.list() if c.is_default]
    assert defaults == ["personal"]


def test_default_is_per_source_not_global(service: CredentialService) -> None:
    """Each source carries its own default."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("other-source", "main", secret="other_secret")

    defaults = {c.source_id: c.name for c in service.list() if c.is_default}
    assert defaults == {"huggingface": "personal", "other-source": "main"}


# ----------------------------- an acquisition naming none uses the default
def test_acquisition_naming_no_credential_uses_the_default(service: CredentialService) -> None:
    """An acquisition that names none resolves the source default."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org", default=True)

    assert service.resolve_for_acquisition("huggingface") == "hf_org"


def test_acquisition_may_name_a_specific_credential(service: CredentialService) -> None:
    """Naming one selects it over the default."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org", default=True)

    assert service.resolve_for_acquisition("huggingface", "personal") == "hf_personal"


def test_acquisition_with_no_credential_for_the_source_resolves_to_none(
    service: CredentialService,
) -> None:
    """No credential is not an error here — upstream decides."""
    assert service.resolve_for_acquisition("huggingface") is None


def test_naming_a_credential_that_does_not_exist_is_an_error(service: CredentialService) -> None:
    """A named credential that is absent is reported, never silently skipped."""
    service.set("huggingface", "personal", secret="hf_personal")

    with pytest.raises(NotFoundError) as exc_info:
        service.resolve_for_acquisition("huggingface", "nonexistent")
    assert "nonexistent" in str(exc_info.value)


# ---------------------------- deleting the default reports the consequence
def test_deleting_the_default_reports_the_consequence(service: CredentialService) -> None:
    """Deleting the default leaves the source with none, and says so."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org")

    result = service.delete("huggingface", "personal")

    assert result["was_default"] is True
    assert result["default_now"] is None
    assert "no default" in result["consequence"]
    # The surviving credential was NOT silently promoted.
    assert [c.name for c in service.list() if c.is_default] == []


def test_deleting_the_default_can_promote_a_named_successor(service: CredentialService) -> None:
    """A promotion happens only because it was asked for."""
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org")

    result = service.delete("huggingface", "personal", promote="org")

    assert result["was_default"] is True
    assert result["default_now"] == "org"
    assert [c.name for c in service.list() if c.is_default] == ["org"]


def test_deleting_a_non_default_leaves_the_default_alone(service: CredentialService) -> None:
    service.set("huggingface", "personal", secret="hf_personal")
    service.set("huggingface", "org", secret="hf_org")

    result = service.delete("huggingface", "org")

    assert result["was_default"] is False
    assert result["default_now"] == "personal"


def test_deleting_a_credential_that_does_not_exist_is_an_error(service: CredentialService) -> None:
    with pytest.raises(NotFoundError):
        service.delete("huggingface", "nonexistent")


def test_deleting_removes_the_stored_value_from_the_provider(
    service: CredentialService, tmp_path: Path
) -> None:
    """Deletion reaches the provider's store, not only the reference."""
    credential = service.set("huggingface", "personal", secret="hf_personal")
    stored = list((tmp_path / "credentials").glob("*"))
    assert stored, "the provider wrote the value to its protected store"

    service.delete("huggingface", credential.name)
    assert list((tmp_path / "credentials").glob("*")) == []


# ---------------------------------------------------- reference, never value
def test_the_store_holds_a_reference_never_a_value(service: CredentialService) -> None:
    """The database column is a pointer into the provider's store."""
    service.set("huggingface", "personal", secret="hf_personal_SECRET")

    credential = service.list()[0]
    assert credential.secret_ref != "hf_personal_SECRET"
    assert "hf_personal_SECRET" not in credential.secret_ref


def test_list_exposes_no_secret_value(service: CredentialService) -> None:
    """``credential list`` shows names and default status only."""
    service.set("huggingface", "personal", secret="hf_personal_SECRET")

    rendered = repr([c.__dict__ for c in service.list()])
    assert "hf_personal_SECRET" not in rendered


def test_service_offers_no_read_path_for_a_value(service: CredentialService) -> None:
    """There is no management-surface method that returns a secret.

    ``resolve_for_acquisition`` is the acquisition-time path and is
    never routed; the guardrail is that nothing named like a read accessor
    exists for a client to reach.
    """
    read_paths = {"get", "read", "show", "reveal", "value", "secret"}
    exposed = {name for name in dir(service) if not name.startswith("_")}
    assert exposed & read_paths == set(), f"credential read path exposed: {exposed & read_paths}"


# --------------------------------------------------------- reference sources
def test_set_from_env_stores_the_value_and_persists_a_reference(
    service: CredentialService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``from_env`` is a reference form; the value still lands in the store."""
    monkeypatch.setenv("HF_TOKEN_FOR_TEST", "hf_from_env")
    service.set("huggingface", "personal", from_env="HF_TOKEN_FOR_TEST")

    assert service.resolve_for_acquisition("huggingface") == "hf_from_env"
    assert "hf_from_env" not in service.list()[0].secret_ref


def test_set_from_file_stores_the_value_and_persists_a_reference(
    service: CredentialService, tmp_path: Path
) -> None:
    secret_file = tmp_path / "hf-org"
    secret_file.write_text("hf_from_file\n")
    service.set("huggingface", "org", from_file=str(secret_file))

    assert service.resolve_for_acquisition("huggingface", "org") == "hf_from_file"


def test_set_requires_exactly_one_of_the_three_forms(service: CredentialService) -> None:
    """secret | from_env | from_file — exactly one."""
    with pytest.raises(ValueError):
        service.set("huggingface", "personal")
    with pytest.raises(ValueError):
        service.set("huggingface", "personal", secret="a", from_env="B")


def test_set_from_missing_env_is_reported(service: CredentialService) -> None:
    with pytest.raises(ValueError) as exc_info:
        service.set("huggingface", "personal", from_env="DEFINITELY_UNSET_TEST_VAR")
    assert "DEFINITELY_UNSET_TEST_VAR" in str(exc_info.value)


def test_set_from_env_refuses_tensorstead_own_variables(
    service: CredentialService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``from_env`` refuses to bind Tensorstead's own operational variables.

    ``from_env`` exists so an operator can bind a third-party credential
    (a Hugging Face token, say) without typing the value. It must not be a
    way to read Tensorstead's own operational secrets -- an MCP caller asking
    to bind ``TENSORSTEAD_MGMT_TOKEN`` as an "inference credential" is exactly
    the exfiltration path a recorded finding described. Refused for every caller,
    not only MCP: there is no legitimate reason for it via the CLI either.
    """
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "fleet-secret")

    with pytest.raises(ValueError, match="operational variable"):
        service.set("huggingface", "personal", from_env="TENSORSTEAD_MGMT_TOKEN")

    assert service.list() == [], "no reference should have been persisted"


def test_set_from_file_refuses_a_symlink(service: CredentialService, tmp_path: Path) -> None:
    """``O_NOFOLLOW`` refuses atomically at open time, not stat-then-open."""
    real_secret = tmp_path / "real-secret"
    real_secret.write_text("root-owned-value\n")
    link = tmp_path / "link-to-secret"
    link.symlink_to(real_secret)

    with pytest.raises(ValueError, match="could not read"):
        service.set("huggingface", "personal", from_file=str(link))


def test_set_from_file_refuses_a_non_regular_file(
    service: CredentialService, tmp_path: Path
) -> None:
    """A FIFO (or device, or socket) is not a credential file."""
    fifo_path = tmp_path / "not-a-file"
    os.mkfifo(fifo_path)

    with pytest.raises(ValueError, match="not a regular file"):
        service.set("huggingface", "personal", from_file=str(fifo_path))


def test_set_from_file_refuses_a_file_over_the_size_cap(
    service: CredentialService, tmp_path: Path
) -> None:
    oversized = tmp_path / "oversized"
    oversized.write_bytes(b"x" * (65536 + 1))

    with pytest.raises(ValueError, match=r"exceeds the .*-byte limit"):
        service.set("huggingface", "personal", from_file=str(oversized))


def test_replacing_a_credential_deletes_the_superseded_value(
    service: CredentialService, tmp_path: Path
) -> None:
    """Re-setting the same name replaces the value; the old one does not linger."""
    service.set("huggingface", "personal", secret="hf_old")
    service.set("huggingface", "personal", secret="hf_new")

    assert service.resolve_for_acquisition("huggingface") == "hf_new"
    stored = [p.read_text() for p in (tmp_path / "credentials").glob("*")]
    assert "hf_old" not in stored


# --------------------------------------------------------------- provider
def test_provider_store_is_permission_restricted(tmp_path: Path) -> None:
    """The value's file and directory are owner-only."""
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    ref = provider.store("huggingface", "personal", "hf_secret")

    root = tmp_path / "credentials"
    assert root.stat().st_mode & 0o777 == 0o700
    stored = next(iter(root.glob("*")))
    assert stored.stat().st_mode & 0o777 == 0o600
    assert provider.resolve(ref) == "hf_secret"


def test_provider_resolve_of_an_unknown_reference_is_an_error(tmp_path: Path) -> None:
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    with pytest.raises(NotFoundError):
        provider.resolve("local-file:does-not-exist")
