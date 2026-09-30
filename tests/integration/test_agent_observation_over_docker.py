"""The agent's observation route against a real container.

The closest reproduction of the 2026-08-10 incident available without a Spark:
a real Docker container really serving HTTP on a published port, observed
through the real agent route with the real docker-py backend and the real vLLM
adapter. Only the model is absent, and the model was never what the defect was
about.

The arc is the incident's, in order:

1. the runtime serves      -> running, reachable, inference_ready
2. the runtime dies, and the container stays, as Docker leaves it
                           -> not_running, with the exit code
3. a listener remains but no longer serves
                           -> running and reachable, inference_ready false

Step 3 is the state Tensorstead reported as healthy for the length of the
incident. Nothing here is faked: the failure of `inference_ready` comes from an
HTTP request to a real socket that answers with the wrong status.

Skipped when no Docker daemon is reachable. No GPU and no vLLM.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.agent.app import build_agent_app

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer mgmt"}
_IMAGE = "python:3.12-alpine"
_DEPLOYMENT_ID = "01KZJ7AV2QGYD6SNH1QVYH0S80"
_CONTAINER = f"tensorstead-{_DEPLOYMENT_ID}"

# A stand-in for a runtime's OpenAI-compatible surface. `--serving` decides
# whether /v1/models answers 200 or 503, which is the difference between a
# runtime that is up and one that is merely listening.
_SERVER = """
import sys, http.server
serving = "--serving" in sys.argv
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models" and serving:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[{"id":"qwen36-27b"}]}')
        else:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"model not loaded"}')
    def log_message(self, *a):
        pass
http.server.HTTPServer(("0.0.0.0", 8000), H).serve_forever()
"""


def _docker_or_skip() -> Any:
    docker = pytest.importorskip("docker", reason="docker SDK not installed")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.skip(f"no reachable Docker daemon: {type(exc).__name__}: {exc}")
    return client


@pytest.fixture(scope="module")
def client() -> Any:
    docker_client = _docker_or_skip()
    try:
        docker_client.images.get(_IMAGE)
    except Exception:
        docker_client.images.pull(_IMAGE)
    return docker_client


@pytest.fixture
def runtime(client: Any) -> Any:
    """A real container serving a runtime-shaped API on a published port."""
    started: list[Any] = []

    def _start(*, serving: bool = True, revision: int = 9) -> tuple[Any, int]:
        with contextlib.suppress(Exception):
            client.containers.get(_CONTAINER).remove(force=True)

        command = ["python", "-c", _SERVER]
        if serving:
            command.append("--serving")

        container = client.containers.create(
            _IMAGE,
            command=command,
            name=_CONTAINER,
            # Published on an ephemeral host port, which is what the agent
            # probes. The declared endpoint is filled in below once Docker has
            # chosen it.
            ports={"8000/tcp": None},
            labels={
                "tensorstead.deployment_id": _DEPLOYMENT_ID,
                "tensorstead.revision": str(revision),
                "tensorstead.runtime_type": "vllm",
            },
        )
        started.append(container)
        container.start()
        container.reload()

        binding = container.attrs["NetworkSettings"]["Ports"]["8000/tcp"][0]
        host_port = int(binding["HostPort"])

        # The agent reads the endpoint from a label; Docker only tells us the
        # port after start, so the container is recreated with it recorded.
        # Simpler than a fixed port, and it keeps the test parallel-safe.
        container.stop(timeout=1)
        container.remove(force=True)
        started.remove(container)

        container = client.containers.create(
            _IMAGE,
            command=command,
            name=_CONTAINER,
            ports={"8000/tcp": host_port},
            labels={
                "tensorstead.deployment_id": _DEPLOYMENT_ID,
                "tensorstead.revision": str(revision),
                "tensorstead.runtime_type": "vllm",
                "tensorstead.endpoint": f"0.0.0.0:{host_port}",
            },
        )
        started.append(container)
        container.start()
        _await_listening(host_port)
        return container, host_port

    yield _start

    for container in started:
        with contextlib.suppress(Exception):
            container.remove(force=True)


def _await_listening(port: int, *, timeout: float = 30.0) -> None:
    """Wait until the container's server returns a real HTTP response.

    A TCP connect is *not* sufficient here, which is itself worth recording:
    Docker Desktop's port proxy binds the host port as soon as the container
    starts, so `connect()` succeeds while the process inside is still coming up
    and the request that follows is reset. That is a real startup race, not a
    test artifact -- and the product reports it correctly, as reachable but not
    ready. This helper waits past it so each test observes the state it means to.
    """
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.0)
            return
        except Exception:
            time.sleep(0.2)
    pytest.fail(f"container never served HTTP on {port}")


@pytest.fixture
def agent(client: Any, tmp_path: Any) -> TestClient:
    """The real agent app on the real docker-py backend."""
    from tensorstead.agent.container_engine.docker_py import DockerEngine

    app = build_agent_app(
        management_token="mgmt",
        container_engine=DockerEngine(client=client),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    return TestClient(app)


def _observe(agent: TestClient) -> dict[str, Any]:
    response = agent.get(f"/agent/v1/deployments/{_DEPLOYMENT_ID}/observed", headers=_AUTH)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_a_serving_runtime_is_reported_serving(agent: TestClient, runtime: Any) -> None:
    """The healthy case, end to end, with nothing stubbed."""
    runtime(serving=True, revision=9)

    observed = _observe(agent)

    assert observed["status"] == "running"
    assert observed["endpoint_reachable"] is True
    assert observed["inference_ready"] is True
    assert observed["running_revision"] == 9
    assert observed["running_image_digest"] is not None


def test_a_dead_runtime_is_reported_dead(agent: TestClient, runtime: Any, client: Any) -> None:
    """The container outlives the process, and the agent is no longer fooled."""
    container, _port = runtime(serving=True)

    container.kill()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and client.containers.get(_CONTAINER).status != "exited":
        time.sleep(0.1)

    observed = _observe(agent)

    assert observed["status"] == "not_running", (
        "a real killed container was reported as running -- this is the "
        "2026-08-10 incident, reproduced against a real daemon"
    )
    # The revision survives the death of the process, because it lives on the
    # container rather than in the agent.
    assert observed["running_revision"] == 9
    assert observed["endpoint_reachable"] is None
    assert observed["inference_ready"] is None
    assert "exited" in (observed["detail"] or "")


def test_listening_but_not_serving_is_the_incident_state(agent: TestClient, runtime: Any) -> None:
    """The exact state Tensorstead called healthy for the whole incident.

    A process is up, the port accepts connections, and no inference is served.
    Before this change it reported `status: running`, `endpoint_reachable: true`
    and no divergence. The port is genuinely bound and the 503 is genuinely
    returned over TCP -- nothing about this failure is simulated.
    """
    runtime(serving=False, revision=9)

    observed = _observe(agent)

    assert observed["status"] == "running"
    assert observed["endpoint_reachable"] is True
    assert observed["inference_ready"] is False, (
        "a listening-but-not-serving runtime was reported ready -- the "
        "distinction this feature exists for did not survive contact with "
        "a real socket"
    )
    assert "503" in (observed["detail"] or "")
