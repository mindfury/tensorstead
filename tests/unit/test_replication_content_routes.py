"""The source side of the ranged protocol: manifest and file routes.

Both are sub-resources of the content endpoint and hold to its rules: the
replication token, never the management one; only an ``available`` replica;
and a requested path is a peer's input, contained within the replica.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.acquisition import ModelAcquisitionService
from tensorstead.agent.app import build_agent_app
from tensorstead.agent.replication import directory_digest

pytestmark = pytest.mark.unit

_MODEL = "huggingface:org/model"
_REPL = {"Authorization": "Bearer repl"}
_BASE = f"/agent/v1/models/{_MODEL}/content"


@pytest.fixture
def held(tmp_path: Path) -> tuple[TestClient, ModelAcquisitionService, Path]:
    acquisition = ModelAcquisitionService(store_dir=tmp_path / "s", marker_dir=tmp_path / "m")
    staging = acquisition.staging_path(_MODEL)
    (staging / "sub").mkdir(parents=True)
    (staging / "sub" / "weights.bin").write_bytes(bytes(range(256)) * 4)
    (staging / "config.json").write_bytes(b"{}")
    acquisition.promote(_MODEL, staging, content_digest=directory_digest(staging))
    app = build_agent_app(
        management_token="mgmt",
        replication_token="repl",
        acquisition=acquisition,
        model_source=object(),
        container_engine=object(),
        service_manager=object(),
    )
    return TestClient(app), acquisition, tmp_path


def test_the_manifest_lists_every_entry_with_its_size(held: tuple) -> None:
    client, _, _ = held
    response = client.get(f"{_BASE}/manifest", headers=_REPL)
    assert response.status_code == 200
    assert response.json() == {
        "entries": [
            {"path": "config.json", "type": "f", "size": 2},
            {"path": "sub", "type": "d", "size": 0},
            {"path": "sub/weights.bin", "type": "f", "size": 1024},
        ]
    }


def test_a_byte_range_is_served_as_partial_content(held: tuple) -> None:
    client, _, _ = held
    response = client.get(
        f"{_BASE}/file",
        params={"path": "sub/weights.bin"},
        headers={**_REPL, "Range": "bytes=10-19"},
    )
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 10-19/1024"
    assert response.headers["content-length"] == "10"
    assert response.content == bytes(range(10, 20))


def test_a_whole_file_is_served_without_a_range(held: tuple) -> None:
    client, _, _ = held
    response = client.get(f"{_BASE}/file", params={"path": "config.json"}, headers=_REPL)
    assert response.status_code == 200
    assert response.content == b"{}"


def test_an_unsatisfiable_range_is_refused(held: tuple) -> None:
    client, _, _ = held
    response = client.get(
        f"{_BASE}/file", params={"path": "config.json"}, headers={**_REPL, "Range": "bytes=0-99"}
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */2"


@pytest.mark.parametrize(
    "path", ["../../m/x.json", "/etc/passwd", "sub", "missing", "sub/../config.json", ""]
)
def test_only_a_regular_file_inside_the_replica_is_served(held: tuple, path: str) -> None:
    client, _, _ = held
    response = client.get(f"{_BASE}/file", params={"path": path}, headers=_REPL)
    assert response.status_code == 404


def test_a_link_anywhere_on_the_path_is_not_followed(held: tuple) -> None:
    client, acquisition, tmp_path = held
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"must not be served")
    root = acquisition.local_path(_MODEL)
    (root / "via-dir").symlink_to(outside)
    (root / "via-file").symlink_to(outside / "secret")

    for path in ("via-dir/secret", "via-file"):
        response = client.get(f"{_BASE}/file", params={"path": path}, headers=_REPL)
        assert response.status_code == 404
        assert b"must not be served" not in response.content


def test_a_replica_that_is_not_available_is_neither_described_nor_served(held: tuple) -> None:
    client, acquisition, _ = held
    acquisition.mark_failed(_MODEL, detail="test")
    assert client.get(f"{_BASE}/manifest", headers=_REPL).status_code == 409
    response = client.get(f"{_BASE}/file", params={"path": "config.json"}, headers=_REPL)
    assert response.status_code == 409


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer mgmt"}])
def test_both_routes_require_the_replication_token(held: tuple, headers: dict) -> None:
    client, _, _ = held
    assert client.get(f"{_BASE}/manifest", headers=headers).status_code in (401, 403)
    response = client.get(f"{_BASE}/file", params={"path": "config.json"}, headers=headers)
    assert response.status_code in (401, 403)
