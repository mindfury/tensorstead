"""The CLI's rendering functions.

`cli.commands` is the primary human surface, was ~51% covered, and carried 879
of the 1672 surviving mutants in the codebase — more than every other module
combined. The correlation with it also being the least documented was not a
coincidence: nobody exercised it, so nobody noticed what it did or did not say.

These cover the render paths, where a wrong output is a wrong answer to an
operator rather than a crash.
"""

from __future__ import annotations

import pytest

from tensorstead.cli import commands

pytestmark = pytest.mark.unit


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


# -------------------------------------------------------- runtime config split


def test_runtime_config_modelled_keys_render_under_runtime_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Modelled, adapter-validated keys render under a 'runtime config' header."""
    commands._render_runtime_config(
        {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.4, "tool_call_parser": "hermes"}
    )
    out = _out(capsys)
    assert "runtime config" in out
    assert "tensor_parallel_size = 1" in out
    assert "gpu_memory_utilization = 0.4" in out
    assert "tool_call_parser = hermes" in out
    # Nothing forwarded here.
    assert "forwarded" not in out


def test_runtime_config_split_labels_extra_args_as_forwarded_and_unvalidated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``extra_args`` is the forwarded, unvalidated half and must say so."""
    commands._render_runtime_config(
        {
            "tensor_parallel_size": 1,
            "extra_args": {"disable-uvicorn-access-log": "true"},
        }
    )
    out = _out(capsys)
    # Modelled half is labelled plainly; forwarded half says it was not checked.
    assert "runtime config\n    tensor_parallel_size = 1" in out
    assert "extra args (forwarded, not validated by this product)" in out
    assert "disable-uvicorn-access-log = true" in out


def test_runtime_config_split_labels_host_config_as_forwarded_and_unvalidated(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``host_config`` is forwarded container config and must say so."""
    commands._render_runtime_config(
        {
            "tensor_parallel_size": 2,
            "host_config": {"shm_size": 1073741824, "ipc_mode": "host"},
        }
    )
    out = _out(capsys)
    assert "host config (forwarded, not validated by this product)" in out
    assert "shm_size = 1073741824" in out
    assert "ipc_mode = host" in out


def test_runtime_config_empty_modelled_shows_none(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty modelled half says so, rather than a blank that reads as a frozen line."""
    commands._render_runtime_config({})
    out = _out(capsys)
    assert "runtime config" in out
    assert "(none)" in out
    assert "forwarded" not in out


# ----------------------------------------------------------- suggested images


def test_suggested_images_render_reference_and_note(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A suggested image prints its reference and the note that makes it honest."""
    commands._render_suggested_images(
        [
            {
                "type": "vllm",
                "suggested_images": [
                    {
                        "reference": "nvcr.io/nvidia/vllm:26.07-py3",
                        "note": "NGC vLLM; tag moves on NVIDIA's schedule.",
                    }
                ],
            }
        ]
    )
    out = _out(capsys)
    assert "SUGGESTED IMAGES (a starting point, not a guarantee)" in out
    assert "vllm  nvcr.io/nvidia/vllm:26.07-py3" in out
    assert "NGC vLLM; tag moves on NVIDIA's schedule." in out


def test_suggested_images_runtime_with_none_is_named_not_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A runtime that named nothing is reported, not silently dropped."""
    commands._render_suggested_images(
        [
            {"type": "vllm", "suggested_images": [{"reference": "x:1", "note": "n"}]},
            {"type": "llamacpp", "suggested_images": []},
        ]
    )
    out = _out(capsys)
    assert "llamacpp  (none named)" in out


def test_suggested_images_section_omitted_when_no_runtime_named_any(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No section header when every runtime named nothing -- absence is not a heading."""
    commands._render_suggested_images(
        [
            {"type": "vllm", "suggested_images": []},
            {"type": "llamacpp", "suggested_images": []},
        ]
    )
    out = _out(capsys)
    assert "SUGGESTED IMAGES" not in out


# ----------------------------------------------- unredacted log-tail warning


def test_runtime_warns_unredacted_when_a_node_has_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-empty log tail is preceded by the unredacted-output warning."""
    commands._render_runtime(
        {
            "per_node": {
                "01JNODE": {
                    "running": True,
                    "restart_count": 0,
                    "log_tail": "INFO: started\nERROR: oops",
                }
            }
        }
    )
    out = _out(capsys)
    assert "shown verbatim and unredacted" in out
    assert "INFO: started" in out


def test_runtime_warning_omitted_when_no_node_has_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No warning when every node's tail is empty or absent.

    A warning printed on every invocation is one nobody reads by the time it
    matters, so it appears only when there is output to warn about.
    """
    commands._render_runtime(
        {
            "per_node": {
                "01JNODE": {"running": True, "log_tail": None},
                "02JNODE": {"running": True, "log_tail": ""},
            }
        }
    )
    out = _out(capsys)
    assert "unredacted" not in out


def test_runtime_warning_omitted_when_no_nodes_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No nodes reported -> the 'no participating nodes' line, no warning."""
    commands._render_runtime({"per_node": {}})
    out = _out(capsys)
    assert "No participating nodes reported." in out
    assert "unredacted" not in out


# ------------------------------------------------------------------ health


def test_all_nodes_running_and_reachable_reports_running() -> None:
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": True},
            "b": {"status": "running", "endpoint_reachable": True},
        },
    }
    assert commands._deployment_health(observed) == "running"


def test_one_unreachable_node_is_not_reported_as_running() -> None:
    """A partial answer must never be rendered as whole-deployment health."""
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": True},
            "b": {"status": "running", "endpoint_reachable": False},
        },
    }
    assert commands._deployment_health(observed) == "degraded"


def test_one_stopped_node_is_not_reported_as_running() -> None:
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": True},
            "b": {"status": "not_running", "endpoint_reachable": True},
        },
    }
    assert commands._deployment_health(observed) == "degraded"


def test_listening_but_not_serving_is_not_reported_as_running() -> None:
    """The word an operator reads before deciding nothing is wrong.

    Every node running, every endpoint reachable, and no inference served —
    the state the 2026-08-10 incident summarised as "running".
    """
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": True, "inference_ready": False},
        },
    }
    assert commands._deployment_health(observed) == "not serving"


def test_unknown_readiness_still_reports_running() -> None:
    """``None`` is "we could not tell", which is not a fault to report.

    A runtime whose adapter declares no probe would otherwise render as
    permanently degraded, which teaches operators to ignore the field.
    """
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": True, "inference_ready": None},
        },
    }
    assert commands._deployment_health(observed) == "running"


def test_unknown_reachability_is_not_reported_as_degraded() -> None:
    """``None`` reachability is "we could not tell", not a fault.

    Found on the live appliance immediately after deploying 009: every
    deployment created before container labels existed reports
    ``endpoint_reachable: null``, and `bool(None)` read that as unreachable.
    `deployment show` said running with no divergence while `deployment list`
    said degraded — the same deployment, two answers, one of them invented.
    """
    observed = {
        "status": "running",
        "per_node": {
            "a": {"status": "running", "endpoint_reachable": None, "inference_ready": None},
        },
    }
    assert commands._deployment_health(observed) == "running"


def test_a_silent_coordinator_is_not_blamed_on_the_agent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Name the right party for a missing version.

    Found running `node show` against a coordinator that predated the field: the
    key was absent from the response and the CLI reported "agent did not say",
    which would send an operator to upgrade the wrong host. Absent key and null
    value are different facts about different machines.
    """
    captured: list[str] = []
    monkeypatch.setattr(commands.typer, "echo", lambda msg="": captured.append(str(msg)))

    class _Response:
        status_code = 200

        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def json(self) -> dict:
            return self._payload

    class _Client:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def get(self, *_args: object, **_kwargs: object) -> _Response:
            return _Response(self._payload)

    # Old coordinator: the key is not in the response at all.
    monkeypatch.setattr(
        commands, "_client", lambda **_kw: _Client({"status": "reachable", "observed_at": None})
    )
    commands._render_node_observed("01ABC")
    assert any("coordinator predates" in line for line in captured)

    captured.clear()

    # Current coordinator, agent that did not report a version.
    monkeypatch.setattr(
        commands,
        "_client",
        lambda **_kw: _Client(
            {"status": "reachable", "observed_at": None, "contract_version": None}
        ),
    )
    commands._render_node_observed("01ABC")
    assert any("agent did not say" in line for line in captured)


def test_health_falls_back_to_the_overall_status_without_per_node() -> None:
    assert commands._deployment_health({"status": "unknown"}) == "unknown"
    assert commands._deployment_health({"status": "running", "per_node": {}}) == "running"


# --------------------------------------------------------------- observed


def test_an_unauthenticated_endpoint_is_rendered_emphatically(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An open inference endpoint must not read as normal."""
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "endpoint_authenticated": False}},
            },
            "divergences": [],
        }
    )
    assert "UNAUTHENTICATED" in _out(capsys)


def test_an_authenticated_endpoint_says_so_plainly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "endpoint_authenticated": True}},
            },
            "divergences": [],
        }
    )
    out = _out(capsys)
    assert "requires a key" in out
    assert "UNAUTHENTICATED" not in out


def test_an_agent_that_did_not_say_produces_no_authentication_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`None` is unknown, and must not be rendered as either answer."""
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "endpoint_authenticated": None}},
            },
            "divergences": [],
        }
    )
    out = _out(capsys)
    assert "UNAUTHENTICATED" not in out
    assert "requires a key" not in out


def test_an_unreachable_endpoint_is_annotated_on_its_node(
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "endpoint_reachable": False}},
            },
            "divergences": [],
        }
    )
    assert "endpoint unavailable" in _out(capsys)


def test_drift_is_reported_with_a_count(capsys: pytest.CaptureFixture[str]) -> None:
    commands._render_observed(
        {"observed": {"status": "running"}, "divergences": [{"kind": "x"}, {"kind": "y"}]}
    )
    out = _out(capsys)
    assert "drift" in out
    assert "2 difference" in out


def test_no_drift_says_no_rather_than_staying_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Absence of drift is an answer and must be shown as one."""
    commands._render_observed({"observed": {"status": "running"}, "divergences": []})
    assert "drift              no" in _out(capsys)


def test_a_malformed_observed_block_does_not_crash_the_renderer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An older coordinator's shape must degrade, not raise, in a display path."""
    commands._render_observed({"observed": "not-a-dict", "divergences": "nope"})
    assert "Status" in _out(capsys)


# ------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("value", "expected"),
    [("01KZJ5JJESQ48BX3JNR3ZR4QZM", "01KZJ5JJESQ4"), ("short", "short")],
)
def test_ids_are_shortened_for_display_without_losing_short_ones(value: str, expected: str) -> None:
    assert commands._short_id(value) == expected


def test_a_missing_timestamp_renders_as_a_dash_not_as_none() -> None:
    """ "None" in a table reads as a value; a dash reads as absence."""
    rendered = commands._short_time(None)
    assert "None" not in rendered
    assert rendered.strip() in {"-", "\u2014", ""}


def test_a_timestamp_is_rendered_without_its_timezone_noise() -> None:
    rendered = commands._short_time("2026-08-09T12:52:20.178976-04:00")
    assert "2026-08-09" in rendered
    assert "178976" not in rendered


def test_a_deployment_that_is_not_serving_says_where_the_reason_is(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The instant an operator reads "not serving" is when they want the reason.

    Before the product printed it, the only route to the reason was SSH. Printing
    the next command here is the difference between the product answering the
    next question and merely raising it.
    """
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "inference_ready": False}},
            },
            "divergences": [],
        }
    )
    assert "deployment runtime" in _out(capsys)


def test_a_healthy_deployment_is_not_told_where_to_look(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A suggestion printed on every healthy deployment is one nobody reads."""
    commands._render_observed(
        {
            "observed": {
                "status": "running",
                "per_node": {"n1": {"status": "running", "inference_ready": True}},
            },
            "divergences": [],
        }
    )
    assert "deployment runtime" not in _out(capsys)
