"""Reading the served model name a deployment declares."""

from __future__ import annotations

import pytest

from tensorstead.domain.served_name import declared_served_name

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"served_model_name": "chat"}, "chat"),
        ({"extra_args": {"served-model-name": "qwen3.8-27b"}}, "qwen3.8-27b"),
        ({"extra_args": {"--served-model-name": "a b"}}, "a"),
        ({"extra_args": {"alias": "llama"}}, "llama"),
        ({"served_model_name": "  "}, None),
        ({"extra_args": {"other": "x"}}, None),
        ({}, None),
        (None, None),
    ],
)
def test_declared_served_name(config: dict | None, expected: str | None) -> None:
    assert declared_served_name(config) == expected
