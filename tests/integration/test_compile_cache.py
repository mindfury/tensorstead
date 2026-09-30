"""The durable compile cache survives the container it was built in.

vLLM compiles on first use — Triton kernels, `torch.compile` artefacts — and
caches both inside the container. Tensorstead replaces the container on **every**
restart and every revision change, so that work was thrown away and redone each
time. Observed on the appliance on 2026-08-11: four Triton kernels compiling on
live traffic, minutes after a restart, on a deployment already started several
times that day.

The obvious fix — let an operator mount a volume — is the one the design refuses,
because a bind can point anywhere and a deployment that mounts an arbitrary host
path is no longer a true account of what is running. So the vocabulary changed
instead: the adapter states *what must be true* ("durable storage visible at
this in-container path") and the agent decides where it lives. The operator
never names a host path and `host_config` still refuses `volumes` and `binds`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorstead.agent import cache_store

pytestmark = pytest.mark.integration


def test_a_deployments_cache_outlives_its_container(tmp_path: Path) -> None:
    """The entire point: the directory is not owned by the container's lifetime."""
    first = cache_store.provision("01DEPLOY", root=tmp_path)
    assert first is not None
    Path(first, "compiled.bin").write_text("expensive")

    # The container is destroyed and re-created, as restart does.
    second = cache_store.provision("01DEPLOY", root=tmp_path)

    assert second == first
    assert Path(second, "compiled.bin").read_text() == "expensive"


def test_two_deployments_do_not_share_a_cache(tmp_path: Path) -> None:
    """Per-deployment, so disk stays accountable (`node resources` by purpose).

    A shared node-wide cache would be safe — these caches are content-addressed
    internally — and would use less disk. It was rejected for v1 because nobody
    could then say whose bytes those are, or reclaim them with the deployment
    they belong to.
    """
    a = cache_store.provision("01AAA", root=tmp_path)
    b = cache_store.provision("01BBB", root=tmp_path)

    assert a != b


def test_the_cache_is_not_world_writable(tmp_path: Path) -> None:
    """A compile cache holds executable artefacts, so its mode is a security control.

    The convenient answer to "the container may run as a different user" is
    0o777. It would be a route from any local account on the host to code
    executing inside the inference container, because the runtime loads and runs
    what it finds here — Triton cubins, compiled shared objects.

    The cache is therefore private to its owner, and a container whose user
    cannot be determined recompiles instead: slow, visible in the runtime log
    tail, and safe.
    """
    path = cache_store.provision("01PERMS", root=tmp_path)
    assert path is not None

    mode = os.stat(path).st_mode & 0o777

    assert not mode & 0o022, "the compile cache is writable by accounts other than its owner"
    assert not mode & 0o004, "the compile cache is world-readable"


def test_the_cache_is_handed_to_the_user_the_image_runs_as(tmp_path: Path) -> None:
    """Ownership is how a non-root container gets access without opening the mode.

    Only meaningful as root, so the assertion is on the call being accepted and
    the directory still being private; the chown itself needs privilege the test
    suite does not have and should not want.
    """
    path = cache_store.provision("01OWNED", root=tmp_path, owner_uid=os.getuid())
    assert path is not None
    assert os.stat(path).st_mode & 0o777 == 0o700


def test_an_unprovisionable_cache_does_not_stop_a_deployment(tmp_path: Path) -> None:
    """Slow beats down.

    A runtime with no durable cache recompiles. Refusing to start a deployment
    because a *performance* directory was unavailable would turn an optimisation
    into an outage, so the caller starts without one.
    """
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("this is a file")

    assert cache_store.provision("01BLOCKED", root=blocked) is None


def test_removing_a_deployment_discards_its_cache(tmp_path: Path) -> None:
    """Unlike model artifacts and images, which the design deliberately retains.

    Those are expensive to re-acquire and may be shared. A compile cache is
    derived from them and rebuilds itself, so keeping it after its deployment is
    gone leaks disk that nothing will ever claim.
    """
    path = cache_store.provision("01GONE", root=tmp_path)
    assert path is not None and Path(path).exists()

    assert cache_store.discard("01GONE", root=tmp_path) is True
    assert not Path(path).exists()


def test_discarding_a_cache_that_was_never_there_is_not_an_event(tmp_path: Path) -> None:
    """So `deployment remove` does not claim to have removed something it did not."""
    assert cache_store.discard("01NEVER", root=tmp_path) is False


def test_the_agent_provisions_a_cache_when_the_adapter_asks_for_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the create path: adapter asks, agent provides, engine mounts."""
    from fastapi.testclient import TestClient

    from tensorstead.agent.app import build_agent_app
    from tests.fakes.container_engine import FakeContainerEngine
    from tests.fakes.service_manager import FakeServiceManager
    from tests.helpers import make_vllm_model_dir

    monkeypatch.setenv("TENSORSTEAD_CACHE_PATH", str(tmp_path))
    app = build_agent_app(
        management_token="test",
        container_engine=FakeContainerEngine(),
        service_manager=FakeServiceManager(),
        store_dir=tmp_path / "store",
        marker_dir=tmp_path / "markers",
    )
    client = TestClient(app)

    # The pre-flight inspects the model dir before starting the container,
    # so the tree must actually exist on disk — not a /var/lib/... path that the
    # test host does not have.
    model_path = make_vllm_model_dir(tmp_path)
    deployment_id = "01J00000000000000000000004"
    response = client.post(
        "/agent/v1/deployments",
        json={
            "deployment_id": deployment_id,
            "revision": 1,
            "runtime_type": "vllm",
            "image_reference": "local/vllm:test",
            "runtime_config": {"tensor_parallel_size": 1},
            "model_path": model_path,
            "endpoint": "0.0.0.0:8000",
        },
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == 200, response.text

    container = app.state.container_engine.containers[f"tensorstead-{deployment_id}"]

    assert container.cache_path == str((tmp_path / deployment_id).resolve())
    assert Path(container.cache_path).is_dir()
    assert container.environment is not None
    assert container.environment["VLLM_CACHE_ROOT"].startswith(container.requirements.cache_at)


def test_a_numeric_image_user_is_resolved_and_a_named_one_is_not(tmp_path: Path) -> None:
    """``Config.User`` is free-form; only some of its forms are actionable.

    ``1000`` and ``1000:1000`` name a uid. ``vllm`` names an account that exists
    only inside the image, and resolving it would mean reading the image's own
    ``/etc/passwd``. Guessing would either fail silently or hand the cache to
    the wrong host account, so an unresolvable name leaves it root-owned.
    """
    from tensorstead.agent.routes.deployments import _image_uid
    from tests.fakes.container_engine import FakeContainerEngine

    engine = FakeContainerEngine()
    engine.image_users = {
        "img:numeric": "1000",
        "img:pair": "1000:1000",
        "img:named": "vllm",
        "img:empty": "",
    }

    assert _image_uid(engine, "img:numeric") == 1000
    assert _image_uid(engine, "img:pair") == 1000
    assert _image_uid(engine, "img:named") is None
    assert _image_uid(engine, "img:empty") is None
    assert _image_uid(engine, "img:unknown") is None


def test_an_engine_that_cannot_say_is_not_an_error(tmp_path: Path) -> None:
    """An older engine without ``image_user`` still deploys, just without chown."""
    from tensorstead.agent.routes.deployments import _image_uid

    assert _image_uid(object(), "anything") is None


def test_the_report_distinguishes_written_from_merely_provisioned(tmp_path: Path) -> None:
    """The question the feature shipped unable to answer.

    On 2026-08-11 a second start reused persisted ``torch.compile`` artefacts —
    initialisation fell from 143.6s to 48.2s — while Triton kernels still
    JIT-compiled on the request path. Two explanations fitted equally: the
    Triton cache was not being written, or the warning does not mean what it
    appears to. Nothing in the product could tell them apart, because a
    provisioned directory and a *used* one looked identical.

    They no longer do.
    """
    cache_store.provision("01MIXED", root=tmp_path)
    triton = tmp_path / "01MIXED" / "triton"
    triton.mkdir()
    vllm = tmp_path / "01MIXED" / "vllm" / "torch_compile_cache"
    vllm.mkdir(parents=True)
    (vllm / "artefact.py").write_bytes(b"x" * 2048)

    report = cache_store.inspect("01MIXED", root=tmp_path)
    by_name = {s["name"]: s for s in report["subtrees"]}

    assert report["present"] is True
    assert by_name["triton"]["files"] == 0, "an unwritten subtree must report as empty"
    assert by_name["vllm"]["files"] == 1
    assert by_name["vllm"]["bytes"] == 2048


def test_an_absent_cache_is_not_reported_as_an_empty_one(tmp_path: Path) -> None:
    """Never provisioned and provisioned-but-unused are different diagnoses."""
    report = cache_store.inspect("01NONE", root=tmp_path)

    assert report["present"] is False
    assert report["subtrees"] == []


def test_cache_dir_for_refuses_an_absolute_deployment_id(tmp_path: Path) -> None:
    """The escape this whole boundary exists to close.

    ``Path(root) / deployment_id`` silently discards ``root`` when
    ``deployment_id`` is itself absolute -- a pathlib join, not a containment
    check. Before this fix, an agent request naming an existing host
    directory as its deployment id caused ``provision`` to ``chmod`` it to
    ``0700`` and, with an image UID, ``chown`` it -- an SSH or system config
    directory, for instance. Confirmed as a live reproduction before the fix:
    ``provision`` returned the outside path and changed its mode.
    """
    victim = tmp_path / "victim"
    victim.mkdir()
    victim.chmod(0o755)

    with pytest.raises(ValueError, match="does not resolve beneath the cache root"):
        cache_store.cache_dir_for(str(victim), root=tmp_path / "cache-root")

    # The would-be victim directory was never touched.
    assert oct(victim.stat().st_mode)[-3:] == "755"


def test_cache_dir_for_refuses_traversal_out_of_the_root(tmp_path: Path) -> None:
    """A relative ``..`` escape is the same bug in a different disguise."""
    root = tmp_path / "cache-root"
    root.mkdir()

    with pytest.raises(ValueError, match="does not resolve beneath the cache root"):
        cache_store.cache_dir_for("../escape", root=root)


def test_provision_never_creates_outside_the_root(tmp_path: Path) -> None:
    """``provision`` degrades to ``None`` rather than acting outside its root.

    Mirrors the existing ``test_provision_degrades_to_none_...``-style
    contract for an unwritable root: an escape attempt is exactly as
    survivable as a permissions failure, never a host mutation.
    """
    victim = tmp_path / "victim"
    victim.mkdir()

    assert cache_store.provision(str(victim), root=tmp_path / "cache-root") is None
