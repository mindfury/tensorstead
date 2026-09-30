"""Contract test ``GET /v1/runtimes``.

Each runtime entry must declare ``supports_distributed`` — the
capability the distributed-request tests check against for multi-node requests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import build_test_coordinator

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer test"}


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app)


def test_each_runtime_declares_supports_distributed(client: TestClient) -> None:
    """Every runtime entry declares ``supports_distributed``."""
    resp = client.get("/v1/runtimes", headers=_AUTH)
    assert resp.status_code == 200
    runtimes = resp.json()
    assert runtimes, "at least one runtime is expected"
    for runtime in runtimes:
        assert "supports_distributed" in runtime, (
            f"runtime {runtime['type']!r} must declare supports_distributed"
        )
        assert isinstance(runtime["supports_distributed"], bool)
        assert runtime["type"]


def test_vllm_is_distributed(client: TestClient) -> None:
    """vLLM is distributed-capable."""
    runtimes = client.get("/v1/runtimes", headers=_AUTH).json()
    vllm = next((r for r in runtimes if r["type"] == "vllm"), None)
    assert vllm is not None, "vllm runtime is expected"
    assert vllm["supports_distributed"] is True


def test_runtimes_requires_auth(client: TestClient) -> None:
    """Management token gates the runtimes surface."""
    assert client.get("/v1/runtimes").status_code == 401


def test_each_runtime_names_a_suggested_image_reference(client: TestClient) -> None:
    """Each runtime can name at least one usable image reference.

    The reference alone is not enough -- it goes stale on the runtime's release
    schedule, not ours -- so each suggestion carries a note saying what kind of
    suggestion it is. A suggestion with no note would be the absence-as-answer
    defect: a tag presented as authoritative when nobody vouches for it.
    """
    runtimes = client.get("/v1/runtimes", headers=_AUTH).json()
    assert runtimes, "at least one runtime is expected"
    for runtime in runtimes:
        suggestions = runtime.get("suggested_images", [])
        assert suggestions, (
            f"runtime {runtime['type']!r} names no suggested image."
            "If the adapter genuinely cannot name one, report an empty list and "
            "say so on the surface -- do not let absence read as 'no starting "
            "point is knowable'."
        )
        for img in suggestions:
            assert img["reference"], "a suggested image must carry a reference"
            assert img["note"], (
                f"suggestion {img['reference']!r} has no note; a tag without a "
                "note reads as a guarantee this product does not make; it is a "
                "suggestion, never a promise"
            )
