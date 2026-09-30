"""``InferenceCredentialService.resolve_for_deployment`` must not fail open.

``None`` means exactly one thing: nothing is bound, so the node's own
provisioning applies -- the pre-existing migration path. A binding that
exists but cannot actually be resolved -- its record vanished, or the
provider could not produce a value -- is a different fact and must raise
rather than collapse into the same ``None`` a caller then treats as safe to
fall back from. Before this fix it did collapse: the record-missing and
falsy-value cases returned ``None`` directly, and ``LifecycleService`` caught
every other exception and also turned it into ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorstead.adapters.credentials.local_file import LocalFileCredentialProvider
from tensorstead.adapters.sqlite.connection import connect
from tensorstead.adapters.sqlite.migrations import migrate
from tensorstead.adapters.sqlite.repository import SQLiteRepository
from tensorstead.domain.errors import CredentialResolutionError
from tensorstead.domain.models import Deployment
from tensorstead.service.credentials import InferenceCredentialService

pytestmark = pytest.mark.unit

_MIGRATIONS = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "tensorstead"
    / "adapters"
    / "sqlite"
    / "migrations"
)

# deployment_inference_credentials.deployment_id is a real foreign key --
# unrelated but the same table, so every
# test below saves a minimal Deployment row first rather than binding
# against an id nothing references.
_DEPLOYMENT_ID = "01J00000000000000000000001"


@pytest.fixture
def service(tmp_path: Path) -> InferenceCredentialService:
    conn = connect(":memory:")
    migrate(conn, _MIGRATIONS)
    repository = SQLiteRepository(conn)
    repository.save_deployment(
        Deployment(id=_DEPLOYMENT_ID, name="test-dep", desired_state="stopped", current_revision=1)
    )
    provider = LocalFileCredentialProvider(root=tmp_path / "credentials")
    return InferenceCredentialService(repository, provider)


def test_nothing_bound_resolves_to_none(service: InferenceCredentialService) -> None:
    """The legitimate case: no binding means the node's own provisioning applies."""
    assert service.resolve_for_deployment(_DEPLOYMENT_ID) is None


def test_a_bound_resolvable_credential_returns_its_value(
    service: InferenceCredentialService,
) -> None:
    service.set("primary", value="the-real-secret")
    service.bind(_DEPLOYMENT_ID, "primary")

    assert service.resolve_for_deployment(_DEPLOYMENT_ID) == "the-real-secret"


def test_a_binding_whose_record_vanished_raises_rather_than_returning_none(
    service: InferenceCredentialService,
) -> None:
    """Corrupt state, reachable only by bypassing two independent guards.

    ``InferenceCredentialService.delete`` refuses while a deployment binds
    the name, and the schema's own foreign key backs that up -- deleting a
    still-bound ``inference_credentials`` row was tried first and the
    database itself refused it, which is a good sign for the schema and a
    problem for constructing this test. Foreign keys are turned off for
    this one delete to reach the state anyway: the database already
    documents that this schema has real ways to end up inconsistent
    under partial transaction failure, so defending
    ``resolve_for_deployment`` against a bound name with no record is not
    defending against the impossible.
    """
    service.set("primary", value="the-real-secret")
    service.bind(_DEPLOYMENT_ID, "primary")
    conn = service._repo._conn  # type: ignore[attr-defined]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        service._repo.delete_inference_credential("primary")
    finally:
        conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(CredentialResolutionError, match="no longer exists"):
        service.resolve_for_deployment(_DEPLOYMENT_ID)


def test_a_provider_failure_raises_rather_than_returning_none(
    service: InferenceCredentialService,
) -> None:
    """The provider itself failing -- e.g. the backing file became unreadable."""
    service.set("primary", value="the-real-secret")
    service.bind(_DEPLOYMENT_ID, "primary")

    def _broken_resolve(_ref: str) -> str:
        raise OSError("permission denied")

    service._provider.resolve = _broken_resolve

    with pytest.raises(CredentialResolutionError, match="could not be resolved"):
        service.resolve_for_deployment(_DEPLOYMENT_ID)


def test_a_resolved_empty_value_raises_rather_than_returning_none(
    service: InferenceCredentialService,
) -> None:
    """An empty string is not a valid credential and not the same as 'nothing bound'."""
    service.set("primary", value="the-real-secret")
    service.bind(_DEPLOYMENT_ID, "primary")

    service._provider.resolve = lambda _ref: ""

    with pytest.raises(CredentialResolutionError, match="empty value"):
        service.resolve_for_deployment(_DEPLOYMENT_ID)


class _RaisingInferenceCredentials:
    """A minimal stand-in whose resolve always raises, for the lifecycle test."""

    def resolve_for_deployment(self, deployment_id: str) -> str | None:
        raise CredentialResolutionError(
            f"deployment {deployment_id!r}'s bound credential is broken"
        )


def test_lifecycle_no_longer_swallows_a_resolution_failure() -> None:
    """The other half of the repair: the caller must not catch this and
    return None.

    Constructs the minimal object the real method needs (an
    ``_inference_credentials`` attribute) rather than a full
    ``LifecycleService``, since the property under test is entirely local to
    ``_resolve_inference_credential``'s own exception handling.
    """
    from tensorstead.service.lifecycle import LifecycleService

    stub = object.__new__(LifecycleService)
    stub._inference_credentials = _RaisingInferenceCredentials()  # type: ignore[attr-defined]

    with pytest.raises(CredentialResolutionError, match="broken"):
        LifecycleService._resolve_inference_credential(
            stub,
            _DEPLOYMENT_ID,
        )


def test_lifecycle_still_returns_none_with_no_credential_service_configured() -> None:
    """The estate-without-this-feature case must be unaffected."""
    from tensorstead.service.lifecycle import LifecycleService

    stub = object.__new__(LifecycleService)
    # No _inference_credentials attribute at all -- getattr(..., None) path.

    result = LifecycleService._resolve_inference_credential(
        stub,
        _DEPLOYMENT_ID,
    )

    assert result is None
