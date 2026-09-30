"""An import names a file in the managed store, never a path.

The design narrowed the rule for *recorded build specs*. It did not narrow the
separate rule that no parameter accepts an arbitrary path, and the first draft
of the import route did exactly that — caught by `test_mcp_no_shell`, not by
review. These tests pin the containment that replaced it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.routes.images import _managed_archive

pytestmark = pytest.mark.contract


def _request(store: Path) -> Any:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(image_store_path=str(store))))


def test_a_plain_name_inside_the_store_resolves(tmp_path: Path) -> None:
    (tmp_path / "vllm.tar").write_bytes(b"x")

    assert _managed_archive(_request(tmp_path), "vllm.tar") == (tmp_path / "vllm.tar").resolve()


@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "sub/dir.tar",
        "..",
        ".",
        "",
    ],
)
def test_anything_path_shaped_is_refused(tmp_path: Path, name: str) -> None:
    """An agent may name an archive an operator placed there, and nothing else."""
    with pytest.raises(ImageBuildError):
        _managed_archive(_request(tmp_path), name)


def test_a_name_that_is_not_present_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ImageBuildError) as caught:
        _managed_archive(_request(tmp_path), "absent.tar")
    assert "absent.tar" in str(caught.value)


def test_a_symlink_escaping_the_store_is_refused(tmp_path: Path) -> None:
    """Containment must survive a link, not only a literal path."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.tar").write_bytes(b"x")
    store = tmp_path / "store"
    store.mkdir()
    (store / "link.tar").symlink_to(outside / "secret.tar")

    with pytest.raises(ImageBuildError):
        _managed_archive(_request(store), "link.tar")
