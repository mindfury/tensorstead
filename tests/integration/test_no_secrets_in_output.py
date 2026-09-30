"""🚫 GUARDRAIL — no secret in any product output.

The requirement asks for **0 occurrences** of secret material across exports, operation
records, API responses (errors included), and product-emitted logs. This test
takes that literally: it creates a deployment whose model was acquired with a
gated credential, then sweeps every one of those surfaces for the exact secret
string.

Two design choices matter for what this test is worth:

- The sentinel is **distinctive**, so a match is a match and not a coincidence.
- Error paths are exercised on purpose. A redaction step that covers the happy
  path and forgets the exception handler is the usual way secrets escape, so the
  sweep includes refused acquisitions and failed lookups.

The reason this passes is structural rather than diligent: the value never
enters the database (the credential store holds a reference instead),
and no response model, export field, or operation record has a place to put one.
There is nothing here for a redaction step to miss, because there is no
redaction step.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from tests.fakes.node_agent import FakeNodeAgent
from tests.helpers import build_test_coordinator, poll_operation

pytestmark = pytest.mark.integration

_AUTH = {"Authorization": "Bearer test"}

# Distinctive enough that any occurrence is unambiguous.
_SECRET = "hf_zQ7wErTyUiOpAsDfGhJkLzXcVbNm1234567890"


@pytest.fixture
def client() -> TestClient:
    app, _ = build_test_coordinator()
    return TestClient(app, raise_server_exceptions=False)


def _agent(client: TestClient) -> FakeNodeAgent:
    """The fake node agent behind this coordinator (tests/helpers.py)."""
    agent: FakeNodeAgent = client.app.state.fake_agent  # type: ignore[attr-defined]
    return agent


def _assert_clean(text: str, where: str) -> None:
    assert _SECRET not in text, f"secret material found in {where}"


def _setup(client: TestClient) -> tuple[str, str]:
    """Gated model + credential + running deployment. Returns (deployment, model)."""
    agent = _agent(client)
    agent.gate_model("org/gated-model")

    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-01", "agent_endpoint": "https://10.0.0.11:8443"},
        headers=_AUTH,
    )
    node_id = str(resp.json()["id"])

    stored = client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )
    assert stored.status_code == 204

    acquired = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/gated-model",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    assert acquired.status_code == 202
    # The route returns 202 immediately and runs on a background thread, so the
    # model row is written asynchronously — poll the operation to terminal
    # before reading the model list.
    poll_operation(client, acquired.json()["operation_id"])

    models = client.get("/v1/models", headers=_AUTH).json()
    model_id = str(next(m["id"] for m in models if m["source_model_id"] == "org/gated-model"))

    created = client.post(
        "/v1/deployments",
        json={
            "name": "gated-llama",
            "model_id": model_id,
            "runtime_type": "vllm",
            "runtime_version": "0.6.0",
            "image_reference": "repo/vllm:tag",
            "runtime_config": {"tensor_parallel_size": 2},
            "participating_nodes": [node_id],
            "endpoint": "10.0.0.11:8000",
        },
        headers=_AUTH,
    )
    assert created.status_code == 202
    deployments = client.get("/v1/deployments", headers=_AUTH).json()
    dep_id = str(
        next(d["declared"]["id"] for d in deployments if d["declared"]["name"] == "gated-llama")
    )
    client.post(f"/v1/deployments/{dep_id}:start", headers=_AUTH)
    return dep_id, model_id


def test_no_secret_in_the_export(client: TestClient) -> None:
    """The export is a projection of a revision, which holds no secret."""
    dep_id, _ = _setup(client)

    export = client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH)
    assert export.status_code == 200
    _assert_clean(export.text, "the deployment export")
    _assert_clean(json.dumps(export.json()), "the serialized export")

    # Every retained revision, not only the current one.
    revisions = client.get(f"/v1/deployments/{dep_id}/revisions", headers=_AUTH)
    _assert_clean(revisions.text, "the revision list")
    for revision in revisions.json():
        rendered = client.get(
            f"/v1/deployments/{dep_id}/export?revision={revision['revision']}", headers=_AUTH
        )
        _assert_clean(rendered.text, f"the export of revision {revision['revision']}")


def test_no_secret_in_operation_records(client: TestClient) -> None:
    """Operation records carry progress and failure reasons, never secrets."""
    _setup(client)

    operations = client.get("/v1/operations", headers=_AUTH)
    assert operations.status_code == 200
    _assert_clean(operations.text, "the operation list")

    for operation in operations.json():
        detail = client.get(f"/v1/operations/{operation['id']}", headers=_AUTH)
        _assert_clean(detail.text, f"operation {operation['id']}")


def test_no_secret_in_api_responses_including_errors(client: TestClient) -> None:
    """Every response body, error paths included."""
    dep_id, model_id = _setup(client)

    responses = [
        client.get("/v1/credentials", headers=_AUTH),
        client.get("/v1/nodes", headers=_AUTH),
        client.get("/v1/models", headers=_AUTH),
        client.get(f"/v1/models/{model_id}", headers=_AUTH),
        client.get("/v1/deployments", headers=_AUTH),
        client.get(f"/v1/deployments/{dep_id}", headers=_AUTH),
        client.get(f"/v1/deployments/{dep_id}/status", headers=_AUTH),
        client.get("/v1/images", headers=_AUTH),
        # Error paths: a refusal, a missing entity, a rejected payload, and an
        # unauthorized call — the handlers most likely to echo an input back.
        client.get("/v1/deployments/01JZZZZZZZZZZZZZZZZZZZZZZZ", headers=_AUTH),
        client.get("/v1/credentials"),
        client.put(
            "/v1/credentials/huggingface/broken",
            json={"secret": _SECRET, "from_env": "ALSO_SET"},
            headers=_AUTH,
        ),
        client.put(
            "/v1/credentials/huggingface/broken",
            json={"from_env": "TENSORSTEAD_DEFINITELY_UNSET"},
            headers=_AUTH,
        ),
        client.request("DELETE", "/v1/credentials/huggingface/nonexistent", headers=_AUTH),
        client.post(
            "/v1/models:acquire",
            json={
                "source_id": "huggingface",
                "source_model_id": "org/gated-model",
                "revision": "e1f2a3b",
                "nodes": ["01JZZZZZZZZZZZZZZZZZZZZZZZ"],
                "credential": "nonexistent",
            },
            headers=_AUTH,
        ),
    ]
    for response in responses:
        _assert_clean(response.text, f"the response to {response.request.url}")


def test_no_secret_in_a_refused_acquisition(client: TestClient) -> None:
    """The refusal names the source and the nature — never the credential."""
    agent = _agent(client)
    agent.gate_model("org/other-gated")
    resp = client.post(
        "/v1/nodes",
        json={"name": "spark-02", "agent_endpoint": "https://10.0.0.12:8443"},
        headers=_AUTH,
    )
    node_id = str(resp.json()["id"])
    client.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )
    agent.hub.require_access_terms("org/other-gated")

    refused = client.post(
        "/v1/models:acquire",
        json={
            "source_id": "huggingface",
            "source_model_id": "org/other-gated",
            "revision": "e1f2a3b",
            "nodes": [node_id],
        },
        headers=_AUTH,
    )
    # The refusal is the terminal state of the accepted operation, not
    # a synchronous 403 — poll it to failed so the refusal is recorded, then the
    # operation-record sweep below exercises the refusal's failure_reason.
    assert refused.status_code == 202
    _assert_clean(refused.text, "an authorization refusal")
    poll_operation(client, refused.json()["operation_id"])

    # The failure was recorded, and the record is clean too.
    operations = client.get("/v1/operations", headers=_AUTH)
    _assert_clean(operations.text, "the operation record of a refused acquisition")


def test_no_secret_in_product_emitted_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing the product logs contains secret material."""
    with caplog.at_level(logging.DEBUG):
        dep_id, _ = _setup(client)
        client.get(f"/v1/deployments/{dep_id}/export", headers=_AUTH)
        client.get("/v1/operations", headers=_AUTH)
        # An error path, where a handler is most likely to log its input.
        client.post(
            "/v1/models:acquire",
            json={
                "source_id": "huggingface",
                "source_model_id": "org/gated-model",
                "revision": "e1f2a3b",
                "nodes": ["01JZZZZZZZZZZZZZZZZZZZZZZZ"],
            },
            headers=_AUTH,
        )

    _assert_clean(caplog.text, "product-emitted logs")


def test_the_database_never_holds_the_value(client: TestClient) -> None:
    """The store holds a reference; the value is not in any column.

    Read straight out of SQLite rather than through the service, so this is a
    statement about the persisted bytes and not about how they are rendered.
    """
    app, repository = build_test_coordinator()
    inner = TestClient(app)
    inner.put(
        "/v1/credentials/huggingface/personal",
        json={"secret": _SECRET, "default": True},
        headers=_AUTH,
    )

    # Reaching into the repository's connection deliberately: the claim is about
    # the bytes on disk, not about what the service layer chooses to return.
    conn = repository._conn
    tables = [
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]
    for table in tables:
        # The table name comes from the schema itself, never from input.
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        for row in rows:
            _assert_clean(repr(tuple(row)), f"the {table} table")


def test_the_credential_value_never_reaches_the_deployment_revision(client: TestClient) -> None:
    """A revision has no field a secret could occupy — a structural guarantee."""
    dep_id, _ = _setup(client)

    revisions = client.get(f"/v1/deployments/{dep_id}/revisions", headers=_AUTH).json()
    for revision in revisions:
        for value in revision.values():
            _assert_clean(repr(value), "a deployment revision field")


# ------------------------------------------------------------------ the CLI
def test_no_cli_option_can_carry_a_secret_value() -> None:
    """``credential set`` has no flag that takes the value itself.

    An argv flag would put the secret into shell history and into every process
    listing on the host for the lifetime of the call — output in every sense
    the requirement cares about, and the one place redaction cannot reach. The value
    therefore arrives by prompt or stdin only, and this asserts no future flag
    quietly reopens that door.

    ``--from-env`` and ``--from-file`` are deliberately allowed: they name where
    a value lives without carrying it.
    """
    import typer

    from tensorstead.cli.commands import credentials_app

    group = typer.main.get_command(credentials_app)
    set_command = group.commands["set"]  # type: ignore[attr-defined]

    value_bearing = {"secret", "token", "password", "passwd", "value", "key", "credential"}
    for param in set_command.params:
        assert param.name not in value_bearing, (
            f"`credential set` exposes a parameter {param.name!r} that could carry a "
            f"secret on the command line"
        )
        for opt in getattr(param, "opts", []):
            stem = opt.lstrip("-").replace("-", "_")
            assert stem not in value_bearing, (
                f"`credential set` exposes a flag {opt!r} that could carry a secret "
                f"on the command line"
            )


def test_cli_credential_set_reads_the_secret_from_stdin(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The piped path works and the secret never appears in the output."""
    from typer.testing import CliRunner

    from tensorstead.cli import commands as cli_commands
    from tensorstead.cli.main import app as cli_app

    monkeypatch.setattr(cli_commands, "_client", lambda **_kwargs: client)
    monkeypatch.setenv("TENSORSTEAD_MGMT_TOKEN", "test")

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["credential", "set", "huggingface", "personal", "--default"],
        input=f"{_SECRET}\n",
    )
    assert result.exit_code == 0, result.output
    _assert_clean(result.output, "the `credential set` output")

    listed = runner.invoke(cli_app, ["credential", "list"])
    assert listed.exit_code == 0, listed.output
    assert "personal" in listed.output
    _assert_clean(listed.output, "the `credential list` output")

    # The value did land in the store — the CLI is a real projection, not a
    # command that silently succeeded without doing anything.
    credentials = client.get("/v1/credentials", headers=_AUTH).json()
    assert [(c["name"], c["is_default"]) for c in credentials] == [("personal", True)]
