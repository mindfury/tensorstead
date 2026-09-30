"""The fake hub and the real one must offer the same surface.

``HuggingFaceSource`` is written once against one contract, and which object
implements it — ``tests/fakes/hf_hub.FakeHuggingFaceHub`` under test, ``_RealHubAdapter``
in production — is supposed to be invisible to it.

That only holds while the two surfaces agree. When file selection was added
the source began passing ``allow_patterns``; the fake took
it because the test's own stub accepted ``**kwargs``, every unit test passed,
and the real adapter — which declares its parameters explicitly — raised
``TypeError: got an unexpected keyword argument 'allow_patterns'`` on the first
real acquire against a live node.

A fake with a *narrower* surface than the real thing hides precisely the defect
it exists to catch. This compares the two signatures directly, so the next
parameter added to one has to be added to the other.
"""

from __future__ import annotations

import inspect

import pytest

from tensorstead.adapters.sources.huggingface import _RealHubAdapter
from tests.fakes.hf_hub import FakeHuggingFaceHub

pytestmark = pytest.mark.contract


def _accepted(method: object) -> set[str]:
    """Parameter names a callable accepts, excluding ``self``."""
    return {
        name
        for name, p in inspect.signature(method).parameters.items()  # type: ignore[arg-type]
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


def test_the_fake_accepts_everything_the_real_hub_accepts() -> None:
    """The direction that matters: the source calls both the same way.

    A fake missing a parameter the real adapter has means production takes an
    argument the tests never exercised.
    """
    real = _accepted(_RealHubAdapter.snapshot_download)
    fake = _accepted(FakeHuggingFaceHub.snapshot_download)
    missing = real - fake
    assert not missing, (
        f"FakeHuggingFaceHub.snapshot_download does not accept {sorted(missing)}, which "
        f"_RealHubAdapter does. The source is written against one contract; a "
        f"narrower fake means production is called in a way nothing tested."
    )


def test_the_real_hub_accepts_everything_the_fake_does() -> None:
    """And the reverse, so a test cannot pass by exercising a fiction."""
    real = _accepted(_RealHubAdapter.snapshot_download)
    fake = _accepted(FakeHuggingFaceHub.snapshot_download)
    extra = fake - real
    assert not extra, (
        f"FakeHuggingFaceHub.snapshot_download accepts {sorted(extra)}, which "
        f"_RealHubAdapter does not. A test using it would be asserting behaviour "
        f"that cannot happen in production."
    )


def test_resolve_revision_agrees_too() -> None:
    assert _accepted(_RealHubAdapter.resolve_revision) == _accepted(
        FakeHuggingFaceHub.resolve_revision
    )


def test_file_selection_is_part_of_that_surface() -> None:
    """Pinned by name: this is the parameter whose omission caused the defect."""
    assert "allow_patterns" in _accepted(_RealHubAdapter.snapshot_download)
    assert "allow_patterns" in _accepted(FakeHuggingFaceHub.snapshot_download)
