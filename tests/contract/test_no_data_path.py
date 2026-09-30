"""No-data-path guardrail.

Asserts that no route, socket, or handler in the codebase carries inference
request or response payloads. The product records an endpoint; it never stands
on one. Inference clients reach the runtime's own endpoint directly, using the
runtime's own protocol.

Verified structurally: no route path accepts an ``/infer``, ``/generate``,
``/completions``, ``/chat``, or ``/v1/models``-style inference body, and no
handler references an inference payload field. The managed endpoint is a
recorded value only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"

# Route fragments that would expose an inference data path. None of these may
# appear as a product-owned route or handler.
_INFERENCE_ROUTE_FRAGMENTS = (
    "/infer",
    "/generate",
    "/completions",
    "/chat/completions",
    "/v1/chat",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/rerank",
)

# Inference payload field names that must never be handled by the product.
_INFERENCE_PAYLOAD_FIELDS = (
    "prompt",
    "messages",
    "max_tokens",
    "temperature",
    "stream",
    "input_ids",
)


def _walk_py(root: Path) -> list[Path]:
    """Every Python file under ``root``, ``__init__.py`` included.

    Package initialisers are deliberately *not* skipped. The coordinator's
    entire route surface lives in ``coordinator/routes/__init__.py``, so
    excluding initialisers blinds this guardrail to the single largest file it
    exists to police — a violation could be added there and stay green.
    """
    return sorted(root.rglob("*.py"))


def test_no_inference_route_fragment() -> None:
    """No product route carries an inference request/response payload."""
    for path in _walk_py(_SRC):
        text = path.read_text()
        for fragment in _INFERENCE_ROUTE_FRAGMENTS:
            # Matched as a whole path segment, not a substring. A bare `in`
            # check reported "/infer" inside "/v1/inference-credentials",
            # which is a management route about a credential and carries no
            # inference payload. A guardrail that fires on unrelated names
            # teaches people to route around it, which costs more than the
            # rule it protects.
            pattern = re.escape(fragment) + r"(?![a-zA-Z0-9_-])"
            assert not re.search(pattern, text), (
                f"{path.relative_to(_SRC)} references inference route {fragment!r}; "
                f"the product must not stand on an inference endpoint"
            )


def test_no_inference_payload_fields() -> None:
    """No handler accepts an inference payload field."""
    for path in _walk_py(_SRC):
        text = path.read_text()
        for field in _INFERENCE_PAYLOAD_FIELDS:
            # Allow the word to appear in comments/docstrings about the rule;
            # but reject a Pydantic model or FastAPI body declaring it.
            assert f'"{field}":' not in text and f"'{field}':" not in text, (
                f"{path.relative_to(_SRC)} declares an inference payload field {field!r}"
            )
