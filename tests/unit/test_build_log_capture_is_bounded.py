"""What a failed build keeps of its log, and what it refuses to keep.

The evidence a build failure carries goes into the operation record, and an
operation record is read forever. Every bound here exists because the thing it
bounds is unbounded in production: a build step can be hundreds of kilobytes, a
build log can be hundreds of megabytes, and docker's own reason for a failed
``RUN`` quotes the whole failing command back.

These test the module functions directly rather than through the container
engine, for the reason ``test_dockerfile_rendering`` gives: the fake engine
renders nothing and builds nothing, so a test written against it would assert
only that the fake agrees with itself.
"""

from __future__ import annotations

import pytest
from docker.errors import BuildError

from tensorstead.agent.container_engine.docker_py import (
    _BUILD_LOG_CHARS,
    _BUILD_LOG_LINES,
    _BUILD_REASON_CHARS,
    _BUILD_STEP_ECHO_CHARS,
    _build_failure,
    _build_log_lines,
    _elide,
    _failing_step_index,
)

pytestmark = pytest.mark.unit

_REFERENCE = "local/runtime:test"


def _failure(chunks: list[dict], *, steps: list[str], reason: str = "boom") -> object:
    return _build_failure(BuildError(reason, iter(chunks)), reference=_REFERENCE, steps=steps)


def test_the_tail_is_kept_and_the_head_is_dropped() -> None:
    """A compiler names its error last, so the end of the log is the useful end."""
    chunks = [{"stream": f"line {index}\n"} for index in range(_BUILD_LOG_LINES * 3)]
    error = _failure(chunks, steps=[])

    tail = error.detail["build_log_tail"]
    assert f"line {_BUILD_LOG_LINES * 3 - 1}" in tail
    assert "line 0\n" not in tail
    assert len(tail.splitlines()) <= _BUILD_LOG_LINES


def test_one_enormous_line_is_bounded_too() -> None:
    """The line bound alone is no bound at all — a single line has no length limit."""
    error = _failure([{"stream": "x" * (_BUILD_LOG_CHARS * 4) + "\n"}], steps=[])

    assert len(error.detail["build_log_tail"]) <= _BUILD_LOG_CHARS


def test_the_failing_step_is_echoed_only_in_part() -> None:
    """Identifying the step must not reproduce it (a step can be ~259K here)."""
    huge = "echo " + "A" * 300_000
    error = _failure([{"stream": "Step 2/2 : RUN [...]\n"}], steps=[huge])

    assert error.detail["failing_step_index"] == 0
    assert len(error.detail["failing_step"]) <= _BUILD_STEP_ECHO_CHARS


def test_dockers_reason_is_elided_in_the_middle() -> None:
    """The head identifies the command, the tail carries the exit code."""
    reason = "The command '" + "B" * 200_000 + "' returned a non-zero code: 2"
    error = _failure([], steps=[], reason=reason)

    assert "The command '" in error.message
    assert "returned a non-zero code: 2" in error.message
    assert len(error.message) < _BUILD_REASON_CHARS * 2


def test_elide_keeps_both_ends_and_says_how_much_it_dropped() -> None:
    text = "HEAD" + "." * 500 + "TAIL"
    elided = _elide(text, 100)

    assert elided.startswith("HEAD")
    assert elided.endswith("TAIL")
    assert "elided" in elided
    assert _elide("short", 100) == "short"


def test_a_failure_before_the_first_step_is_not_blamed_on_a_step() -> None:
    """``FROM`` is instruction 1 and no recorded step at all."""
    error = _failure([{"stream": "Step 1/3 : FROM base\n"}], steps=["a", "b"])

    assert "failing_step_index" not in error.detail
    assert "outside the recorded steps" in error.message


def test_a_log_with_no_step_marker_names_no_step() -> None:
    """BuildKit and a builder that died before starting both look like this."""
    error = _failure([{"stream": "failed to solve: no such image\n"}], steps=["a"])

    assert _failing_step_index(["failed to solve: no such image"]) is None
    assert "failing_step_index" not in error.detail
    assert "outside the recorded steps" not in error.message


def test_an_error_carrying_no_log_at_all_still_produces_a_reason() -> None:
    """Not every builder exception is a ``BuildError``; none may raise a second time."""
    error = _build_failure(RuntimeError("daemon gone"), reference=_REFERENCE, steps=["a"])

    assert _build_log_lines(RuntimeError("daemon gone")) == []
    assert "daemon gone" in error.message
    assert error.detail["build_log_tail"] == ""
    assert error.reference == _REFERENCE


def test_the_terminal_error_detail_is_part_of_the_log() -> None:
    """docker reports the last word in ``errorDetail``, not in ``stream``."""
    chunks = [
        {"stream": "Step 2/2 : RUN [...]\n"},
        {"error": "boom", "errorDetail": {"code": 2, "message": "nvcc failed with exit code 2"}},
    ]
    error = _failure(chunks, steps=["compile"])

    assert "nvcc failed with exit code 2" in error.detail["build_log_tail"]
