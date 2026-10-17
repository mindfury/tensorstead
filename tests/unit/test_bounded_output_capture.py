"""Build, probe and runtime output is bounded while it is captured, not after.

CodeGuard CG-07: buildx ran with ``capture_output=True``, a classic build's
error log was accumulated whole, and ``logs(tail=N)`` returned the last N lines
as one byte string -- one newline-free record of any size being one line. Each
was trimmed only after everything had been read into memory.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from tensorstead.agent.container_engine import docker_py
from tensorstead.agent.container_engine.bounded_output import BoundedTail

pytestmark = pytest.mark.unit

_MiB = 1024 * 1024


def test_many_small_chunks_stay_bounded_throughout() -> None:
    tail = BoundedTail(max_chars=1000)
    peak = 0
    for i in range(100_000):
        tail.append(f"line {i}\n")
        peak = max(peak, len(tail.text))
    assert peak <= 1000
    assert tail.truncated
    assert tail.lines()[-1] == "line 99999"


def test_a_giant_newline_free_record_is_bounded() -> None:
    tail = BoundedTail(max_chars=4096, max_line_chars=100)
    tail.append("x" * (64 * _MiB))
    assert len(tail.text) == 4096
    (line,) = tail.lines()
    assert len(line) <= 101


def test_a_real_process_flooding_output_is_held_to_the_bound() -> None:
    """50 MB from a real child process, including one 20 MB newline-free record."""
    script = (
        "import sys\n"
        "w = sys.stdout.buffer.write\n"
        "for _ in range(30): w(b'y' * 1_000_000 + b'\\n')\n"
        "w(b'z' * 20_000_000)\n"
        "sys.stderr.write('#6 ERROR: the real reason\\n')\n"
    )
    code, lines = docker_py._run_capped([sys.executable, "-c", script], timeout=60)
    assert code == 0
    held = sum(len(line) for line in lines)
    assert held <= 2 * _MiB, f"held {held} bytes"
    assert any("the real reason" in line for line in lines), "the diagnostic tail was lost"


def test_a_process_past_its_deadline_is_killed() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        docker_py._run_capped([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)


def test_runtime_and_probe_logs_are_streamed_and_bounded() -> None:
    requested: dict[str, Any] = {}

    class _Container:
        def logs(self, **kwargs: Any) -> Any:
            requested.update(kwargs)
            # One giant record, as docker would stream it in frames.
            return iter([b"q" * _MiB for _ in range(64)])

    text = docker_py._bounded_logs(_Container(), tail=500, max_chars=4096)

    assert requested["stream"] is True and requested["tail"] == 500
    assert len(text) == 4096


def test_containers_get_a_rotated_log_driver() -> None:
    config = docker_py._BOUNDED_LOG_CONFIG
    assert config["type"] == "json-file"
    assert config["config"]["max-size"] and config["config"]["max-file"]
