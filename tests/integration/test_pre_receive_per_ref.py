"""The server-side pre-receive gate judges every pushed ref on its own tree.

CodeGuard CG-13: the hook pooled changed paths across refs but materialised only
the *last* ref's tree, so a violating file in one ref of a multi-ref push was
checked against another ref's clean copy -- or skipped when that ref lacked it.
Diff errors were swallowed and filenames word-split.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_HOOK = Path(
    os.environ.get("TENSORSTEAD_TEST_PRE_RECEIVE")
    or Path(__file__).resolve().parents[2] / "ops" / "githooks" / "server" / "pre-receive"
)
# The project's own ruff (the venv beside this interpreter), not whatever is on PATH.
_RUFF = str(Path(sys.executable).parent / "ruff")

if shutil.which("git") is None or shutil.which("bash") is None or not Path(_RUFF).exists():
    pytest.skip("needs git, bash and ruff", allow_module_level=True)


def _ruff_version() -> str:
    return subprocess.run(
        [_RUFF, "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[1]


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "TENSORSTEAD_HOOK_RUFF": _RUFF,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=check
    )


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "server.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    hook = bare / "hooks" / "pre-receive"
    shutil.copy(_HOOK, hook)
    hook.chmod(0o755)

    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    (work / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (work / "uv.lock").write_text(f'[[package]]\nname = "ruff"\nversion = "{_ruff_version()}"\n')
    (work / "ok.py").write_text("x = 1\n")
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", "base")
    _git(work, "remote", "add", "origin", str(bare))
    pushed = _git(work, "push", "-q", "origin", "main", check=False)
    assert pushed.returncode == 0, pushed.stderr
    return bare, work


def _branch(work: Path, name: str, files: dict[str, str]) -> None:
    _git(work, "checkout", "-q", "-b", name, "main")
    for path, text in files.items():
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_text(text)
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", name)
    _git(work, "checkout", "-q", "main")


_BAD = "x   =     1\n"  # unformatted


@pytest.mark.parametrize("order", [("bad", "clean"), ("clean", "bad")])
def test_a_violating_ref_is_refused_whatever_its_position(
    repos: tuple[Path, Path], order: tuple[str, str]
) -> None:
    _, work = repos
    _branch(work, "bad", {"ok.py": _BAD, "only_in_bad.py": _BAD})
    _branch(work, "clean", {"ok.py": "x = 2\n"})
    result = _git(work, "push", "origin", *order, check=False)
    assert result.returncode != 0, result.stderr
    assert "formatting" in result.stderr


def test_a_violation_only_in_a_path_the_other_ref_lacks_is_refused(
    repos: tuple[Path, Path],
) -> None:
    _, work = repos
    _branch(work, "bad", {"only_in_bad.py": _BAD})
    _branch(work, "clean", {"ok.py": "x = 3\n"})
    result = _git(work, "push", "origin", "bad", "clean", check=False)
    assert result.returncode != 0
    assert "refs/heads/bad" in result.stderr


def test_clean_pushes_new_refs_and_deletions_pass(repos: tuple[Path, Path]) -> None:
    _, work = repos
    _branch(work, "clean", {"pkg/new.py": "y = 1\n"})
    assert _git(work, "push", "origin", "clean", check=False).returncode == 0
    assert _git(work, "push", "origin", "--delete", "clean", check=False).returncode == 0


def test_awkward_filenames_are_checked_not_split(repos: tuple[Path, Path]) -> None:
    _, work = repos
    _branch(work, "odd", {"has space/and\nnewline.py": _BAD})
    result = _git(work, "push", "origin", "odd", check=False)
    assert result.returncode != 0
    assert "formatting" in result.stderr


def test_a_diff_that_cannot_run_refuses(repos: tuple[Path, Path]) -> None:
    bare, _ = repos
    bogus = "1" * 40
    result = subprocess.run(
        ["bash", str(bare / "hooks" / "pre-receive")],
        cwd=bare,
        input=f"{'0' * 40} {bogus} refs/heads/x\n",
        env={**os.environ, "GIT_DIR": str(bare), "TENSORSTEAD_HOOK_RUFF": _RUFF},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "could not diff" in result.stderr


def test_ruff_version_skew_fails_closed(repos: tuple[Path, Path]) -> None:
    _, work = repos
    _branch(
        work,
        "skew",
        {"uv.lock": '[[package]]\nname = "ruff"\nversion = "0.0.1"\n', "z.py": "z = 1\n"},
    )
    result = _git(work, "push", "origin", "skew", check=False)
    assert result.returncode != 0
    assert "the ref pins 0.0.1" in result.stderr
