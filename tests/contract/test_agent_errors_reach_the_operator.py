"""🚫 GUARDRAIL — every agent failure reaches the operator with its reason.

Three times now, an exception the agent raises has reached an operator as a bare
500 with its cause discarded:

- the design notes: ``ImageBuildError`` had no handler, so the first real hardware build
  failed as ``internal_error: unexpected internal error``, naming neither the
  reference nor the cause;
- the same fix added ``ImagePullError``;
- an earlier investigation: the first live multi-node build failed while distributing,
  and the operation recorded ``"detail": "Internal Server Error"`` — which does
  not distinguish a refused credential from a routing failure from an identifier
  mismatch, so no retry could be reasoned about.

Each time the fix covered the error that had just failed. This covers the class:
every exception type defined in the agent package must have somewhere to go.

Deliberately **not** a source scan. An earlier guardrail of that shape matched
text, found two of seven cases, and produced false positives — it was deleted.
This imports the package and inspects the app's registered handlers, so it is
answering a question about the running application rather than about its
spelling.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import tensorstead.agent
from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

# Errors handled inside their own route rather than by an app-level handler,
# each with the reason. An entry here is a decision, not an exemption from
# reporting: ``replicate`` catches ``ReplicationError`` and translates it into
# an ``HTTPException`` carrying ``{code, message, detail}`` itself, which is the
# same structured shape by a different route.
_HANDLED_IN_ROUTE: dict[str, str] = {
    "ReplicationError": "routes/replication.py catches it and raises a structured HTTPException",
    "ImageInUseError": "routes/images.py remove_image translates it to 409 image_in_use",
    "ImageNotPresentError": (
        "routes/images.py remove_image treats it as the already-absent case and returns "
        "200 removed=false"
    ),
}


def _agent_exception_classes() -> dict[str, type]:
    """Every exception type defined anywhere in the agent package."""
    found: dict[str, type] = {}
    for module in pkgutil.walk_packages(tensorstead.agent.__path__, "tensorstead.agent."):
        imported = importlib.import_module(module.name)
        for _, obj in vars(imported).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Exception)
                and obj.__module__.startswith("tensorstead.agent")
            ):
                found[obj.__name__] = obj
    return found


def _app(tmp_path: Path) -> Any:
    return build_agent_app(
        management_token="mgmt",
        replication_token="repl",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )


def test_every_agent_exception_has_somewhere_to_go(tmp_path: Path) -> None:
    """The guardrail: no agent error may fall through to a bare 500."""
    registered = {cls.__name__ for cls in _app(tmp_path).exception_handlers if inspect.isclass(cls)}

    unhandled = [
        name
        for name in _agent_exception_classes()
        if name not in registered and name not in _HANDLED_IN_ROUTE
    ]

    assert not unhandled, (
        f"agent exceptions with no handler: {sorted(unhandled)}. Each would reach "
        f"the operator as a bare 500 with its reason discarded, which has now "
        f"happened three times. Register a handler, or add it to "
        f"_HANDLED_IN_ROUTE with the route that translates it."
    )


def test_the_distribution_failure_is_reported_with_its_reason(tmp_path: Path) -> None:
    """The specific case from the live build.

    The destination could not fetch from the source, and the operator was told
    ``Internal Server Error``.
    """
    from fastapi.testclient import TestClient

    from tensorstead.agent.image_distribution import ImageDistributionError

    app = _app(tmp_path)

    class _Refusing:
        def pull_from_peer(self, **kwargs: Any) -> str:
            raise ImageDistributionError(
                "could not fetch 'local/dspark:0.1.1' from https://10.0.0.11:8443: "
                "Client error '401 Unauthorized'",
                reference=str(kwargs.get("reference", "")),
            )

    app.state.image_distribution = _Refusing()

    resp = TestClient(app).post(
        "/agent/v1/images:distribute",
        json={
            "reference": "local/dspark:0.1.1",
            "source_endpoint": "https://10.0.0.11:8443",
            "expected_image_id": "sha256:abc",
        },
        headers={"Authorization": "Bearer mgmt"},
    )

    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["code"] == "image_distribution_failed"
    assert "401 Unauthorized" in body["message"], (
        "the destination's actual reason was discarded; a retry cannot be reasoned about"
    )
    assert body["detail"]["reference"] == "local/dspark:0.1.1"
    assert body["message"] != "Internal Server Error"


def test_the_real_failure_message_carries_no_credential(tmp_path: Path) -> None:
    """A failure message travels to an operator and into operation history.

    The replication token goes in a header and must not appear in the text. This
    drives the **real** ``pull_from_peer`` against an endpoint that refuses the
    connection, so it tests the message this code actually builds -- including
    the upstream exception it interpolates, whose contents we do not control.

    An earlier draft raised an error containing the token on purpose and
    asserted it appeared; that proves a leak leaks, and nothing about the
    product.
    """
    from tensorstead.agent.image_distribution import (
        ImageDistributionError,
        ImageDistributionService,
    )

    secret = "s3cret-replication-token"
    service = ImageDistributionService(FakeContainerEngine(), str(tmp_path / "images"))

    with pytest.raises(ImageDistributionError) as caught:
        service.pull_from_peer(
            reference="local/x:1",
            # Reserved as "discard"; nothing listens, so the connection fails
            # without needing a server or a network round trip.
            source_endpoint="http://127.0.0.1:9",
            expected_image_id="sha256:abc",
            token=secret,
        )

    message = str(caught.value)
    assert secret not in message, (
        f"the replication credential reached the failure message: {message!r}"
    )
    assert "local/x:1" in message, "a failure that does not name the reference is not actionable"
