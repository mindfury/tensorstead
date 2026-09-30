"""Regression coverage for the normal operator-facing CLI projection."""

from __future__ import annotations

import pytest

from tensorstead.cli.commands import _render_human


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            [{"id": "node-identifier", "name": "spark-a", "agent_endpoint": "https://a"}],
            "NODES",
        ),
        (
            [
                {
                    "id": "model-identifier",
                    "source_id": "huggingface",
                    "source_model_id": "example/model",
                    "resolved_revision": "abcdef0123456789",
                    "revision_pinned": True,
                    "replicas": [],
                }
            ],
            "MODELS",
        ),
        ([{"type": "vllm", "versions": [], "supports_distributed": True}], "RUNTIMES"),
        (
            [{"source_id": "huggingface", "name": "default", "is_default": True, "set_at": "now"}],
            "CREDENTIAL REFERENCES",
        ),
        (
            [{"id": "operation-identifier", "kind": "start", "state": "succeeded"}],
            "OPERATIONS",
        ),
        (
            [
                {
                    "node_id": "node-identifier",
                    "reference": "example/image",
                    "digest": "sha256:abcdef",
                }
            ],
            "IMAGES",
        ),
    ],
)
def test_known_list_shapes_render_as_named_operator_tables(
    capsys: pytest.CaptureFixture[str], payload: list[dict], expected: str
) -> None:
    _render_human(payload)

    output = capsys.readouterr().out
    assert expected in output
    assert not output.startswith("[")


def test_resource_reading_is_labelled_for_an_operator(capsys: pytest.CaptureFixture[str]) -> None:
    _render_human(
        {
            "status": "ok",
            "observed_at": "2026-08-09T12:00:00",
            "accelerator_utilization_pct": 25.0,
            "accelerator_memory_used": 1,
            "accelerator_memory_total": 2,
            "memory_is_unified": True,
            "storage": [],
        }
    )

    output = capsys.readouterr().out
    assert "Status  ok" in output
    assert "accelerator use" in output
    assert "unified memory     yes" in output
