"""The CLI must say when it reached something that is not a coordinator.

The default API address is `http://127.0.0.1:8080`, which is also llama.cpp's
default port — one of this product's own supported runtimes. So "the CLI talked
to an unrelated service" is the *most likely* misconfiguration on a workstation,
not an exotic one. Rendering that service's error body verbatim produced a
message about a system the operator never asked about, and never named the
address that was actually called.
"""

from __future__ import annotations

import httpx
import pytest
import typer

from tensorstead.cli.commands import _report_error

pytestmark = pytest.mark.unit

# Verbatim from llama.cpp, which is what produced the original report.
_LLAMA_CPP_404 = {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}


def _response(payload: object, status: int = 404, **headers: str) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        headers=headers,
        request=httpx.Request("GET", "http://127.0.0.1:8080/v1/deployments"),
    )


def test_a_foreign_error_body_is_not_rendered_as_a_tensorstead_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TENSORSTEAD_API", raising=False)

    with pytest.raises(typer.Exit):
        _report_error(_response(_LLAMA_CPP_404, server="llama.cpp"))

    err = capsys.readouterr().err
    assert "not a Tensorstead coordinator" in err
    assert "http://127.0.0.1:8080" in err, "the address actually called must be named"
    assert "llama.cpp" in err, "name what answered, so the cause is obvious"
    assert "TENSORSTEAD_API" in err, "say which setting fixes it"
    # The foreign service's own wording must not masquerade as our failure.
    assert "http_error: " not in err


def test_an_unset_api_variable_is_called_out_explicitly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TENSORSTEAD_API", raising=False)

    with pytest.raises(typer.Exit):
        _report_error(_response(_LLAMA_CPP_404))

    err = capsys.readouterr().err
    assert "is not set" in err
    assert "export TENSORSTEAD_API=" in err


def test_a_configured_api_variable_is_echoed_back_for_checking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TENSORSTEAD_API", "http://192.0.2.10:8000")

    with pytest.raises(typer.Exit):
        _report_error(_response(_LLAMA_CPP_404))

    err = capsys.readouterr().err
    assert "http://192.0.2.10:8000" in err
    assert "not an inference runtime" in err


def test_a_real_coordinator_failure_is_still_rendered_normally(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The structured shape must keep its existing, useful rendering."""
    with pytest.raises(typer.Exit):
        _report_error(
            _response(
                {"code": "endpoint_conflict", "message": "collides with qwen36-27b"}, status=409
            )
        )

    err = capsys.readouterr().err
    assert "endpoint_conflict: collides with qwen36-27b" in err
    assert "not a Tensorstead coordinator" not in err


def test_a_non_json_body_is_treated_as_a_foreign_service(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An HTML error page is a proxy or a web server, never a coordinator."""
    monkeypatch.delenv("TENSORSTEAD_API", raising=False)
    response = httpx.Response(
        502,
        text="<html><body>502 Bad Gateway</body></html>",
        request=httpx.Request("GET", "http://127.0.0.1:8080/v1/deployments"),
    )

    with pytest.raises(typer.Exit):
        _report_error(response)

    assert "not a Tensorstead coordinator" in capsys.readouterr().err
