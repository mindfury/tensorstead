"""Runtime adapters name a usable starting image per runtime.

This feature requires the product to *report* a usable image reference for each
supported runtime without requiring an existing deployment. The design's
out-of-scope note is explicit that reporting is all this is: selection stays
with the operator (no-scheduling rule).

The thing this test pins is the honesty of the report. A bare tag is not a
report -- it goes stale on the runtime's release schedule, not ours -- so each
suggestion carries a note saying what kind of suggestion it is. A reference
without a note is the absence-as-answer defect: a tag presented as
authoritative when nothing vouches for it. The contract test
in ``test_api_runtimes`` checks the same thing through the API; this one pins
it on the adapter, so the capability does not depend on a coordinator running.
"""

from __future__ import annotations

import pytest

from tensorstead.adapters.runtimes.exllama import ExLlamaAdapter
from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.sglang import SGLangAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "adapter", [VLLMAdapter(), LlamaCppAdapter(), SGLangAdapter(), ExLlamaAdapter()]
)
def test_each_adapter_names_a_noted_suggested_image(adapter: object) -> None:
    """Every shipped adapter names at least one image, each with a note."""
    suggested = getattr(adapter, "suggested_images", ())
    assert suggested, (
        f"{type(adapter).__name__} names no suggested image."
        "If it genuinely cannot, declare the empty tuple and the surface says "
        "so -- do not let absence read as 'no starting point is knowable'."
    )
    for img in suggested:
        assert img.reference, "a suggested image must carry a reference"
        assert ":" in img.reference, (
            f"{img.reference!r} has no tag; an image reference names a tag, not "
            "a bare repository, so it does not move under the operator"
        )
        assert img.note, (
            f"suggestion {img.reference!r} has no note; a tag without a note "
            "reads as a guarantee this product does not make"
        )


def test_vllm_names_the_ngc_image_the_estate_runs() -> None:
    """vLLM's suggestion is the real NGC image, not a placeholder.

    The estate runs the NGC vLLM image, so the
    suggestion is a reference known to have worked rather than a guess. Pinning
    the namespace keeps a refactor from silently swapping in a fabricated tag.
    """
    references = {img.reference for img in VLLMAdapter().suggested_images}
    assert any(r.startswith("nvcr.io/nvidia/vllm:") for r in references), (
        f"expected an nvcr.io/nvidia/vllm:* reference; got {references}"
    )
