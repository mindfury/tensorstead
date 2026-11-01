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


def test_the_manifest_names_a_running_content_listeners_port(held: tuple) -> None:
    from types import SimpleNamespace

    client, _, _ = held
    client.app.state.content_listener = SimpleNamespace(port=8444)
    assert client.get(f"{_BASE}/manifest", headers=_REPL).json()["content_port"] == 8444

    client.app.state.content_listener = SimpleNamespace(port=None)
    assert "content_port" not in client.get(f"{_BASE}/manifest", headers=_REPL).json()


@pytest.mark.parametrize("port", [0, 65536, -1, True, "8444", 84.4])
def test_an_advertised_content_port_is_checked_as_untrusted_input(port: object) -> None:
    from tensorstead.agent.replication import ReplicationError, advertised_content_port

    with pytest.raises(ReplicationError, match="invalid content port"):
        advertised_content_port({"content_port": port}, "https://peer:8443")


def test_no_advertised_content_port_means_the_main_port() -> None:
    from tensorstead.agent.replication import advertised_content_port

    assert advertised_content_port({}, "https://peer:8443") is None
    assert advertised_content_port({"content_port": 8444}, "https://peer:8443") == 8444


def test_the_content_listener_is_off_unless_a_port_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorstead.agent.app import content_listener_from_env

    monkeypatch.delenv("TENSORSTEAD_CONTENT_PORT", raising=False)
    assert content_listener_from_env() is None


def test_the_content_listener_takes_the_agents_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorstead.agent.app import content_listener_from_env

    monkeypatch.setenv("TENSORSTEAD_CONTENT_PORT", "8444")
    monkeypatch.delenv("TENSORSTEAD_CONTENT_BIND_HOST", raising=False)
    monkeypatch.delenv("TENSORSTEAD_AGENT_TLS_CERT", raising=False)
    with pytest.raises(RuntimeError, match="TENSORSTEAD_AGENT_TLS_CERT"):
        content_listener_from_env()

    monkeypatch.setenv("TENSORSTEAD_AGENT_TLS_CERT", "/etc/tensorstead/agent.crt")
    monkeypatch.setenv("TENSORSTEAD_AGENT_TLS_KEY", "/etc/tensorstead/agent.key")
    config = content_listener_from_env()
    assert config is not None
    assert (config.host, config.port, config.certfile, config.keyfile) == (
        "0.0.0.0",
        8444,
        "/etc/tensorstead/agent.crt",
        "/etc/tensorstead/agent.key",
    )


@pytest.mark.parametrize("raw", ["0", "70000", "port", "-1"])
def test_a_content_port_that_is_not_a_port_is_refused(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    from tensorstead.agent.app import content_listener_from_env

    monkeypatch.setenv("TENSORSTEAD_CONTENT_PORT", raw)
    with pytest.raises(RuntimeError, match="must be a port number"):
        content_listener_from_env()


def test_the_content_listener_needs_the_replication_token(tmp_path: Path) -> None:
    from tensorstead.agent.content_listener import ContentListenerConfig

    with pytest.raises(ValueError, match="replication token"):
        build_agent_app(
            management_token="mgmt",
            content_listener=ContentListenerConfig("127.0.0.1", 0, "c", "k"),
        )
