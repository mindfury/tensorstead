"""Fake container engine.

Models the agent-side container execution seam. The
real backend is docker-py behind the narrow container-engine adapter; the fake
stands in for it so agent tests need no Docker daemon (tier 1).

It models the behaviour that matters for the conformance suite: a started
container is running, its platform-specific image digest is recorded, and the
adapter never parses CLI output — the Engine REST API is the authority beneath
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tensorstead.agent.container_engine.base import ContainerState


@dataclass
class FakeContainer:
    id: str
    name: str
    image_reference: str
    image_digest: str
    running: bool = False
    endpoint: str | None = None
    command_args: list[str] | None = None
    entrypoint: list[str] | None = None
    environment: dict[str, str] | None = None
    labels: dict[str, str] = field(default_factory=dict)
    exit_code: int | None = None
    restart_count: int = 0
    logs: str = ""
    # What the runtime adapter said this container needs.
    requirements: object | None = None
    # Host directory backing the durable cache, if one was provisioned.
    cache_path: str | None = None
    # Further model directories mounted read-only -- a speculative-decoding
    # drafter the operator acquired. Recorded rather than
    # dropped so a test can assert the agent mounted what it approved, and
    # equally that it mounted nothing when the reference was a repo id.
    extra_model_paths: list[str] = field(default_factory=list)
    # What is *actually* running, when a test needs it to differ from what was
    # configured -- an entrypoint that rebuilt its own command line.
    processes: list[str] | None = None


class FakeContainerEngine:
    """A stateful in-memory container engine implementing the narrow seam."""

    def __init__(self) -> None:
        self.containers: dict[str, FakeContainer] = {}
        self._next_id = 0
        # Images the node already holds, as a locally built or imported image
        # would be. Keyed by reference, valued by identifier.
        self.local_images: dict[str, str] = {}
        # References a pull cannot satisfy. A locally produced image names no
        # registry repository, so this is the real behaviour, not a contrivance.
        self.unpullable: set[str] = set()
        self.pulled: list[str] = []
        # Ids passed to remove_image, in call order -- so a test can assert a
        # rejected import was actually cleaned up, not merely refused.
        self.removed_images: list[str] = []
        self.import_image_calls: list[str] = []
        self.import_image_returns: str = "sha256:imported"
        # Image ids that load and resolve but from which no container can be
        # created -- the arrival-path gap this fake models.
        self.unmaterializable: set[str] = set()
        # Ids passed to verify_image_materializable, in call order, so a test
        # can assert the check ran at all rather than only that it can fail.
        self.materializability_checked: list[str] = []
        # ``Config.User`` per image reference. Empty means every image runs as
        # root, which is Docker's default and the common case for vLLM images.
        self.image_users: dict[str, str] = {}
        # References a container still holds, which a non-forced removal must
        # refuse rather than override.
        self.in_use_images: set[str] = set()

    def pull_image(self, reference: str) -> str:
        """Return the platform-specific digest for ``reference``.

        In the fake this is deterministic; the real backend gets it from the
        Engine REST API.
        """
        self.pulled.append(reference)
        if reference in self.unpullable:
            from tensorstead.agent.container_engine.base import ImagePullError

            raise ImagePullError(f"no such registry repository: {reference}", reference=reference)
        return self._platform_digest(reference)

    def image_digest(self, reference: str) -> str | None:
        """Return the identifier of an image already on the node, or None."""
        return self.local_images.get(reference)

    def remove_image(self, *, image_id: str, force: bool = False) -> None:
        """Remove an image, modelling the daemon's two refusals.

        ``in_use_images`` is what a container still holds; ``force`` overrides
        that, as the identity-mismatch path requires. An image the fake does not
        hold raises ``ImageNotPresentError``, so the absent case is exercised
        rather than assumed.
        """
        from tensorstead.agent.container_engine.base import (
            ImageInUseError,
            ImageNotPresentError,
        )

        if not force:
            if image_id in self.in_use_images:
                raise ImageInUseError(f"{image_id} is in use by a container", reference=image_id)
            if image_id not in self.local_images and image_id not in self.local_images.values():
                raise ImageNotPresentError(f"no such image {image_id}", reference=image_id)
        self.removed_images.append(image_id)
        self.local_images.pop(image_id, None)

    def list_images(self) -> list[dict[str, str]]:
        """One row per locally held reference."""
        return [{"reference": ref, "digest": digest} for ref, digest in self.local_images.items()]

    def verify_image_materializable(self, *, image_id: str) -> None:
        """Accept every image unless a test names one as unusable.

        Models the real gap rather than a contrived one: an image in
        ``unmaterializable`` is one the daemon holds and resolves under the
        right id, and which only fails when a container is created from it
        because its config content cannot be read.
        """
        self.materializability_checked.append(image_id)
        if image_id in self.unmaterializable:
            from tensorstead.agent.container_engine.base import ImageBuildError

            raise ImageBuildError(
                f"failed to read config content for {image_id}", reference=image_id
            )

    def import_image(self, *, archive_path: str) -> str:
        """Report whatever ``import_image_returns`` is set to, ignoring content.

        Set by the test before calling, matching a real ``docker load``: the
        engine has no way to know an archive's resulting id without loading
        it -- this fake models that by simply
        not looking at ``archive_path`` at all.
        """
        self.import_image_calls.append(archive_path)
        return self.import_image_returns

    def _platform_digest(self, reference: str) -> str:
        """The digest for ``reference`` without contacting a registry.

        Creating a container must not re-pull: the image was already resolved
        by the caller, and a second pull would make a locally built image fail
        at container creation even after the resolution step got it right.
        """
        return self.local_images.get(reference) or f"sha256:platform-{hash(reference) & 0xFFFF:04x}"

    def create_container(
        self,
        *,
        name: str,
        image: str,
        endpoint: str,
        model_path: str | None = None,
        command_args: list[str] | None = None,
        entrypoint: list[str] | None = None,
        environment: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        requirements: object | None = None,
        cache_path: str | None = None,
        extra_model_paths: list[str] | None = None,
    ) -> FakeContainer:
        self._next_id += 1
        # The real backend folds the adapter's declared environment into the
        # container's (docker_py._apply_requirements). A fake that skipped that
        # would report no cache variables on a container the real engine would
        # have set them on -- the fake-diverges-from-Docker trap that made the
        # 2026-08-10 incident possible in the first place.
        declared_env = getattr(requirements, "environment", None) or {}
        if declared_env:
            environment = {**declared_env, **(environment or {})}
        container = FakeContainer(
            id=f"c{self._next_id}",
            name=name,
            image_reference=image,
            image_digest=self._platform_digest(image),
            endpoint=endpoint,
            command_args=command_args,
            entrypoint=entrypoint,
            environment=environment,
            labels=dict(labels or {}),
            requirements=requirements,
            cache_path=cache_path,
            extra_model_paths=list(extra_model_paths or []),
        )
        self.containers[name] = container
        return container

    def start_container(self, name: str) -> None:
        container = self.containers[name]
        container.running = True
        container.exit_code = None

    def stop_container(self, name: str, *, exit_code: int = 0) -> None:
        """Stop a container, leaving it in place as Docker does.

        The container object surviving its process is the behaviour that made
        the 2026-08-10 incident possible, so the fake must keep modelling it: a
        stopped container is still resolvable by name and still reports a
        digest. Only ``running`` distinguishes it.
        """
        container = self.containers.get(name)
        if container:
            container.running = False
            container.exit_code = exit_code

    def remove_container(self, name: str) -> None:
        self.containers.pop(name, None)

    def inspect_container(self, name: str) -> ContainerState | None:
        container = self.containers.get(name)
        if container is None:
            return None
        return self._state_of(container)

    def list_managed_containers(self) -> list[ContainerState]:
        """Every ``tensorstead-`` container, running or not."""
        return [
            self._state_of(container)
            for name, container in sorted(self.containers.items())
            if name.startswith("tensorstead-")
        ]

    @staticmethod
    def _state_of(container: FakeContainer) -> ContainerState:
        return ContainerState(
            running=container.running,
            name=container.name,
            image_digest=container.image_digest,
            exit_code=None if container.running else container.exit_code,
            detail="running" if container.running else "exited",
            labels=dict(container.labels),
            restart_count=container.restart_count,
            command=list(container.entrypoint or []) + list(container.command_args or []),
            environment=dict(container.environment or {}),
        )

    def image_user(self, reference: str) -> str | None:
        """What ``Config.User`` would say for this image."""
        return self.image_users.get(reference)

    def container_processes(self, name: str) -> list[str] | None:
        """What is actually running, which a test sets to model a rogue entrypoint."""
        container = self.containers.get(name)
        if container is None or not container.running:
            return None
        if container.processes is not None:
            return list(container.processes)
        # Default: a well-behaved image execs the argv it was handed.
        return [" ".join(list(container.entrypoint or []) + list(container.command_args or []))]

    def container_logs(self, name: str, *, tail: int = 200) -> str | None:
        """The container's last ``tail`` lines, or None if it does not exist."""
        container = self.containers.get(name)
        if container is None:
            return None
        lines = container.logs.splitlines()
        return "\n".join(lines[-max(1, tail) :])

    def get_digest(self, name: str) -> str | None:
        container = self.containers.get(name)
        return container.image_digest if container else None
