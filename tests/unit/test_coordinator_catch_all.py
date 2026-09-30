"""The coordinator's catch-all handler surfaces the exception type.

Before the fix an untyped exception became ``internal_error`` with an empty
``detail`` -- an opaque terminal outcome that named nothing, which is the
recurring defect in this codebase (a record that stopped matching reality,
with nothing comparing them). The handler now logs a traceback and puts the
exception type and a bounded message in ``detail``, so any future
``internal_error`` is diagnosable. These tests pin that contract without
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


def test_catch_all_includes_exception_type_and_message() -> None:
    """An untyped exception surfaces its type and a bounded message, not an empty detail."""
    client = _client(_app_with_boom_route())
    resp = client.get("/__boom", headers=_AUTH)

    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    detail = body["detail"]
    assert detail["exception_type"] == "RuntimeError"
    assert "widget store is on fire" in detail["message"]


def test_catch_all_caps_message_length() -> None:
    """The message is bounded so an oversized exception cannot flood the response."""
    app, _ = build_test_coordinator()

    @app.get("/__longboom")
    async def _longboom() -> dict[str, str]:
        raise RuntimeError("x" * 5000)

    client = _client(app)
    body = client.get("/__longboom", headers=_AUTH).json()
    assert len(body["detail"]["message"]) <= 500


def test_catch_all_logs_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler logs the unexpected error with a traceback for diagnosis."""
    calls: list[str] = []
    monkeypatch.setattr(
        app_module._logger,
        "exception",
        lambda msg, *a, **kw: calls.append(msg),
    )
    client = _client(_app_with_boom_route())
    client.get("/__boom", headers=_AUTH)

    assert calls, "the catch-all must log the unexpected error (D6)"
    assert calls[0] == "unexpected internal error"
