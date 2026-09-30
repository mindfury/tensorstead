"""A recorded build step is one RUN instruction, whatever it contains (016).

The first real DSpark image build failed before producing anything:

    dockerfile parse error on line 84: unknown instruction: chmod

The recorded step was correct. The *rendering* was wrong. ``RUN {step}`` — shell
form — prefixes only the step's first physical line, so a step that installs a
script with a heredoc and then runs ``chmod +x`` on it emitted that ``chmod`` at
Dockerfile top level, where the parser read it as an instruction it did not know.

The class of defect is the one this product keeps finding: a recorded artifact
and the thing built from it stopped matching, with nothing comparing them. Here
the spec said "run this shell command" and the renderer silently reinterpreted
part of it as Dockerfile grammar.

These tests target the renderer directly rather than going through the fake
container engine, because the fake renders no Dockerfile at all — it could not
have caught this, and a test written against it would assert only that the fake
agrees with itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tensorstead.agent.container_engine.docker_py import DockerEngine, render_dockerfile

pytestmark = pytest.mark.unit

# The shape that failed on hardware: a heredoc that writes a script, followed by
# a shell command acting on what the heredoc produced.
_HEREDOC_STEP = """cat <<'EOF' > /usr/local/bin/tensorstead-dspark-prepare
#!/bin/bash
set -euo pipefail
echo preparing
EOF
chmod +x /usr/local/bin/tensorstead-dspark-prepare"""


def _instructions(dockerfile: str) -> list[str]:
    """Every line Docker would read as an instruction — i.e. every top-level line."""
    return [line for line in dockerfile.splitlines() if line and not line.startswith(" ")]


def test_a_multiline_step_emits_no_top_level_line_of_its_own() -> None:
    """The exact hardware failure: ``chmod`` must not become an instruction."""
    dockerfile = render_dockerfile(
        base_image="nvcr.io/nvidia/vllm:26.07-py3", steps=[_HEREDOC_STEP]
    )

    assert not any(line.startswith("chmod") for line in _instructions(dockerfile)), (
        f"a line of the recorded step reached Dockerfile top level:\n{dockerfile}"
    )


def test_each_recorded_step_is_exactly_one_instruction() -> None:
    """Structural, not incidental: N steps render as N RUN instructions.

    Asserted by count rather than by looking for the one command that happened
    to fail on hardware, so any future way of leaking a step's interior fails
    here too.
    """
    steps = [_HEREDOC_STEP, "pip install xgrammar", "echo one\necho two\necho three"]

    dockerfile = render_dockerfile(base_image="base:1", steps=steps, entrypoint=["/bin/prepare"])

    lines = _instructions(dockerfile)
    assert len(lines) == len(steps) + 2, f"expected FROM + {len(steps)} RUN + ENTRYPOINT:\n{lines}"
    assert lines[0].startswith("FROM ")
    assert all(line.startswith("RUN ") for line in lines[1:-1])
    assert lines[-1].startswith("ENTRYPOINT ")


def test_the_step_reaches_the_shell_byte_for_byte() -> None:
    """Rendering must not edit the recipe — the shell gets what was recorded.

    A renderer that escaped newlines into ``\\n`` continuations, or stripped
    them, would pass the test above while silently changing what the heredoc
    writes to disk.
    """
    dockerfile = render_dockerfile(base_image="base:1", steps=[_HEREDOC_STEP])

    argv = json.loads(dockerfile.splitlines()[1].removeprefix("RUN "))

    assert argv[:2] == ["/bin/sh", "-c"]
    assert argv[2] == _HEREDOC_STEP


@pytest.mark.parametrize(
    "step",
    [
        'echo "quoted"',
        "echo 'single'",
        "printf 'a\\tb\\n'",
        "echo $HOME && echo ${VAR:-default}",
        'sed -i "s/a/b/" /etc/file',
    ],
)
def test_shell_metacharacters_survive_rendering(step: str) -> None:
    """Quotes and backslashes are the other way a rendered recipe stops matching."""
    dockerfile = render_dockerfile(base_image="base:1", steps=[step])

    assert json.loads(dockerfile.splitlines()[1].removeprefix("RUN "))[2] == step


def test_buildx_builds_from_the_same_rendered_dockerfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring claim above, for the builder that is now the default.

    BuildKit takes a file rather than a stream, so the assertion moves to the
    file buildx is pointed at — but the thing being asserted is identical: the
    bytes the builder sees come from ``render_dockerfile`` and nowhere else.
    Without this the 016 fix would be verified only on the path that no longer
    runs by default.
    """
    import subprocess

    from tensorstead.agent.container_engine import docker_py

    seen: dict[str, str] = {}
    monkeypatch.setattr(docker_py, "_docker_cli", lambda: "/usr/bin/docker")
    monkeypatch.setattr(docker_py, "_buildx_available", lambda _cli: True)

    def _run(argv: list[str], **_kwargs: Any) -> Any:
        seen["dockerfile"] = Path(argv[argv.index("--file") + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_py.subprocess, "run", _run)

    class _Images:
        def get(self, _ref: str) -> Any:
            return type("Image", (), {"id": "sha256:built"})()

    class _Client:
        images = _Images()

    DockerEngine(client=_Client()).build_image(
        reference="local/dspark:0.1.1",
        base_image="base:1",
        steps=[_HEREDOC_STEP],
        entrypoint=["/usr/local/bin/tensorstead-dspark-prepare"],
    )

    assert not any(line.startswith("chmod") for line in _instructions(seen["dockerfile"])), (
        f"a line of the recorded step reached Dockerfile top level:\n{seen['dockerfile']}"
    )


def test_build_image_renders_through_the_same_function() -> None:
    """The fix has to be *wired in*, not merely present.

    A correct renderer that ``build_image`` does not call would leave the
    hardware failure exactly where it was while every test above passed.
    """
    sent: dict[str, Any] = {}

    class _Images:
        def build(self, *, fileobj: Any, **kwargs: Any) -> tuple[Any, list[Any]]:
            sent["dockerfile"] = fileobj.read().decode("utf-8")
            return type("Image", (), {"id": "sha256:built"})(), []

    class _Client:
        images = _Images()

    identifier = DockerEngine(client=_Client(), prefer_buildx=False).build_image(
        reference="local/dspark:0.1.1",
        base_image="base:1",
        steps=[_HEREDOC_STEP],
        entrypoint=["/usr/local/bin/tensorstead-dspark-prepare"],
    )

    assert identifier == "sha256:built"
    assert sent["dockerfile"] == render_dockerfile(
        base_image="base:1",
        steps=[_HEREDOC_STEP],
        entrypoint=["/usr/local/bin/tensorstead-dspark-prepare"],
    )
    assert not any(line.startswith("chmod") for line in _instructions(sent["dockerfile"]))
