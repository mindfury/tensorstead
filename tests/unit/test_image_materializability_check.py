"""How the docker backend proves an image is whole.

The check has to be a container *create*. That is not an implementation
detail, it is the entire finding: on the containerd image store an image whose
config blob never arrived still inspects successfully, answering with an empty
``Config``, an empty ``RootFS`` and an empty ``Architecture`` rather than an
error. A check built on inspection therefore passes on exactly the images this
exists to reject, which is how the fault reached production in the first place
-- ``import_image`` returned the expected id and ``image_digest`` resolved.

Creating a container is what reads the config blob, and creating a container is
what failed on the worker two minutes after distribution reported 200 OK.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorstead.agent.container_engine.base import ImageBuildError
from tensorstead.agent.container_engine.docker_py import DockerEngine

pytestmark = pytest.mark.unit

_IMAGE_ID = "sha256:b89f26d7ac24968f8f6b5675f23520a5fe3471b6d03d0d89b6484da221f9de51"

# Verbatim from the worker's agent journal, so a reader can match this test to
# the log line that motivated it.
_REAL_FAILURE = (
    "404 Client Error for http+docker://localhost/v1.53/containers/create: Not Found "
    '("failed to read config content: NotFound: content digest '
    'sha256:d670b497aa3cb567cc78260eacdfdaa682e5f4ef5d81c9d048200d28eb8050ec: not found")'
)


class _FakeContainer:
    def __init__(self) -> None:
        self.removed = False

    def remove(self, **_kwargs: Any) -> None:
        self.removed = True


class _FakeContainers:
    def __init__(self, *, fails: bool, removal_fails: bool = False) -> None:
        self.fails = fails
        self.removal_fails = removal_fails
        self.create_calls: list[dict[str, Any]] = []
        self.created: list[_FakeContainer] = []

    def create(self, **kwargs: Any) -> _FakeContainer:
        self.create_calls.append(dict(kwargs))
        if self.fails:
            raise RuntimeError(_REAL_FAILURE)
        container = _FakeContainer()
        if self.removal_fails:

            def _boom(**_kwargs: Any) -> None:
                raise RuntimeError("daemon refused the removal")

            container.remove = _boom  # type: ignore[method-assign]
        self.created.append(container)
        return container


class _FakeClient:
    def __init__(self, *, fails: bool, removal_fails: bool = False) -> None:
        self.containers = _FakeContainers(fails=fails, removal_fails=removal_fails)

    class images:
        @staticmethod
        def get(_reference: str) -> Any:
            raise AssertionError(
                "the check must not be built on inspection: an image missing its "
                "config still inspects successfully on the containerd image store"
            )


def test_a_whole_image_passes_by_creating_a_container() -> None:
    client = _FakeClient(fails=False)

    DockerEngine(client=client).verify_image_materializable(image_id=_IMAGE_ID)

    assert len(client.containers.create_calls) == 1, (
        "creating a container is the check -- it is what reads the config blob"
    )
    assert client.containers.create_calls[0]["image"] == _IMAGE_ID


def test_the_probe_container_is_never_started_and_gets_no_network() -> None:
    """Starting would add nothing and would run an unvetted image's entrypoint.

    Reading the config is what a create does; the answer is already known by
    the time it returns.
    """
    client = _FakeClient(fails=False)

    DockerEngine(client=client).verify_image_materializable(image_id=_IMAGE_ID)

    call = client.containers.create_calls[0]
    assert call["network_mode"] == "none"
    assert call["entrypoint"] == ["/bin/true"], (
        "the image's own entrypoint must be overridden, not inherited"
    )
    assert "detach" not in call or call["detach"] is False


def test_the_probe_container_is_removed_again() -> None:
    """A verification that leaves a container behind has changed the node."""
    client = _FakeClient(fails=False)

    DockerEngine(client=client).verify_image_materializable(image_id=_IMAGE_ID)

    assert client.containers.created[0].removed


def test_an_incomplete_image_is_refused_and_names_the_daemon_error() -> None:
    """The refusal has to carry the daemon's own words.

    The operator-visible symptom was a bare 500 from the deployment route. A
    message that says only "verification failed" would move the mystery rather
    than end it.
    """
    client = _FakeClient(fails=True)

    with pytest.raises(ImageBuildError) as caught:
        DockerEngine(client=client).verify_image_materializable(image_id=_IMAGE_ID)

    message = str(caught.value)
    assert "failed to read config content" in message
    assert "must not be reported as distributed" in message
    assert caught.value.reference == _IMAGE_ID


def test_a_container_that_cannot_be_removed_does_not_fail_a_good_image() -> None:
    """Deliberately unlike ``run_once``'s probe, and worth saying why.

    A probe's answer is discarded when cleanup fails because a probe that
    leaves state behind has changed the thing it described. Here the answer is
    "this image works", the leftover is a never-started container, and refusing
    a good image over it would strand a correct distribution.
    """
    client = _FakeClient(fails=False, removal_fails=True)

    DockerEngine(client=client).verify_image_materializable(image_id=_IMAGE_ID)
