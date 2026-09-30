"""Operator-facing formatting for ``stead deployment list``."""

from __future__ import annotations

import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

from tensorstead.cli import commands as cli_commands
from tensorstead.cli.commands import _parse_runtime_config, _render_deployment_list, _resolve_record
from tensorstead.cli.main import app as cli_app


def test_deployment_list_renders_a_compact_operational_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _render_deployment_list(
        [
            {
                "declared": {
                    "name": "qwen36-27b",
                    "desired_state": "running",
                    "current_revision": 3,
                    "running_revision": 3,
                    "revision": {
                        "endpoint": "spark-alpha.internal:8000",
                        "model": {
                            "source_id": "huggingface",
                            "source_model_id": "nvidia/Qwen3.6-27B-NVFP4",
                        },
                    },
                },
                "observed": {
                    "status": "running",
                    "per_node": {"node-a": {"status": "running", "endpoint_reachable": True}},
                },
                "divergences": [],
            },
            {
                "declared": {
                    "name": "needs-attention",
                    "desired_state": "running",
                    "current_revision": 2,
                    "running_revision": None,
                    "revision": {
                        "endpoint": "spark-beta.internal:8000",
                        "model": {"source_id": "huggingface", "source_model_id": "example/model"},
                    },
                },
                "observed": {"status": "degraded", "per_node": {}},
                "divergences": [{"kind": "state_mismatch"}],
            },
        ]
    )

    output = capsys.readouterr().out
    assert "NAME" in output
    assert "RUNNING/CURRENT" in output
    assert "qwen36-27b" in output
    assert "running" in output
    assert "3/3" in output
    assert "spark-alpha.internal:8000" in output
    assert "huggingface:nvidia/Qwen3.6-27B-NVFP4" in output
    assert "needs-attention" in output
    assert "degraded" in output
    assert "—/2" in output
    assert "example/model *" in output


def test_deployment_list_handles_an_empty_response(capsys: pytest.CaptureFixture[str]) -> None:
    _render_deployment_list([])

    assert capsys.readouterr().out == "No deployments.\n"


def test_runtime_config_accepts_a_json_object() -> None:
    assert _parse_runtime_config(['{"tensor_parallel_size": 1, "max_model_len": 262144}']) == {
        "tensor_parallel_size": 1,
        "max_model_len": 262144,
    }


def test_runtime_config_rejects_an_unparseable_value(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit):
        _parse_runtime_config(["this-is-not-a-setting"])

    assert "use key=value or one JSON object" in capsys.readouterr().err


def test_identifier_prefix_resolves_when_unambiguous() -> None:
    result = _resolve_record(
        [{"id": "01KZJ5JJESQ48BX3JNR3ZR4QZM", "name": "first"}],
        "01KZJ5JJESQ4",
        kind="operation",
        display=lambda record: str(record["name"]),
    )

    assert result["name"] == "first"


def test_identifier_prefix_refuses_an_ambiguous_match(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit):
        _resolve_record(
            [
                {"id": "01KZJ5JJESQ48BX3JNR3ZR4QZM", "name": "first"},
                {"id": "01KZJ5JJESQ4ZZZZZZZZZZZZZZ", "name": "second"},
            ],
            "01KZJ5JJESQ4",
            kind="operation",
            display=lambda record: str(record["name"]),
        )

    assert "ambiguous" in capsys.readouterr().err


def test_root_json_flag_applies_to_a_read_command(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def get(self, _path: str, **_kwargs: object) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "declared": {"name": "demo"},
                        "observed": {"status": "running"},
                        "divergences": [],
                    }
                ],
            )

    monkeypatch.setattr(cli_commands, "_client", lambda **_kwargs: Client())

    result = CliRunner().invoke(cli_app, ["--json", "deployment", "list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["declared"]["name"] == "demo"
