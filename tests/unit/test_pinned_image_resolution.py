"""A digest-pinned reference resolves to a local image held only by that digest.

An image copied from a peer arrives without its ``name@digest`` reference
(docker save/load carries none), so the agent used to miss it and pull from the
registry anyway. Measured on the Sparks, 2026-10-16.
"""

from __future__ import annotations

import pytest

from tensorstead.agent.routes.deployments import _resolve_image
from tensorstead.domain.models import pinned_digest

pytestmark = pytest.mark.unit

_DIGEST = "sha256:" + "cd" * 32
_PINNED = f"repo/vllm@{_DIGEST}"


class _Engine:
    def __init__(self, held: dict[str, str]) -> None:
        self.held = held
        self.pulls: list[str] = []

    def image_digest(self, reference: str) -> str | None:
        return self.held.get(reference)

    def pull_image(self, reference: str) -> str:
        self.pulls.append(reference)
        return _DIGEST


def test_held_by_bare_digest_resolves_without_a_pull() -> None:
    engine = _Engine({_DIGEST: _DIGEST})  # containerd store: id == manifest digest
    assert _resolve_image(engine, _PINNED) == _DIGEST
    assert engine.pulls == []


def test_a_different_image_under_that_id_is_not_taken() -> None:
    """Classic store: the id is the config digest, never the pinned manifest digest."""
    engine = _Engine({_DIGEST: "sha256:" + "ef" * 32})
    _resolve_image(engine, _PINNED)
    assert engine.pulls == [_PINNED]


def test_a_tag_reference_is_unaffected() -> None:
    engine = _Engine({})
    _resolve_image(engine, "repo/vllm:latest")
    assert engine.pulls == ["repo/vllm:latest"]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (_PINNED, _DIGEST),
        ("repo/vllm:latest", None),
        ("repo/vllm@sha256:short", None),
        ("localhost:5000/repo@" + _DIGEST, _DIGEST),
    ],
)
def test_pinned_digest(reference: str, expected: str | None) -> None:
    assert pinned_digest(reference) == expected
