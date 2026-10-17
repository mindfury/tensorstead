"""The coordinator's catch-all handler surfaces the exception type.

Before the fix an untyped exception became ``internal_error`` with an empty
``detail`` -- an opaque terminal outcome that named nothing, which is the
recurring defect in this codebase (a record that stopped matching reality,
with nothing comparing them). The handler logs the message and traceback
under an ``error_id`` and puts the exception type and that id in ``detail``,
so any future ``internal_error`` is diagnosable from the log -- while the
exception's text never reaches the client, where it could carry request or
provider content (CodeGuard CG-12). These tests pin that contract without
coupling to a specific route's failure mode.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.coordinator import app as app_module
from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.unit

_AUTH = {"Authorization": "Bearer test"}


def _app_with_boom_route() -> Any:
    """A test coordinator with one extra route that raises an untyped error."""
    app, _ = build_test_coordinator()

    @app.get("/__boom")
    async def _boom() -> dict[str, str]:
        raise RuntimeError("the widget store is on fire")

    return app


def _client(app: Any) -> TestClient:
    # ``raise_server_exceptions=False`` lets the registered 500 handler produce
    # its JSONResponse instead of the TestClient re-raising the original error.
    return TestClient(app, raise_server_exceptions=False)


def test_catch_all_names_the_type_and_an_error_id_not_the_message() -> None:
    """The type and a log correlation id reach the client; the exception text does not."""
    client = _client(_app_with_boom_route())
    resp = client.get("/__boom", headers=_AUTH)
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    detail = body["detail"]
    assert detail["exception_type"] == "RuntimeError"
    assert len(detail["error_id"]) == 12
    assert "widget store is on fire" not in resp.text


def test_catch_all_logs_exception_under_the_error_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler logs the unexpected error with a traceback, keyed by the id it returns."""
    calls: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        app_module._logger,
        "exception",
        lambda msg, *a, **kw: calls.append((msg, a)),
    )
    client = _client(_app_with_boom_route())
    error_id = client.get("/__boom", headers=_AUTH).json()["detail"]["error_id"]
    assert calls, "the catch-all must log the unexpected error (D6)"
    assert calls[0][0].startswith("unexpected internal error")
    assert error_id in calls[0][1]
