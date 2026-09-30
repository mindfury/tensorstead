"""Container-engine seam.

A narrow operation set behind which the docker-py backend hides. No
Docker type escapes this module — the domain and the agent's routes never see a
``docker`` object. The seam models the behaviours the agent depends on:

- pulling an image and returning its **platform-specific** digest
  rather than the multi-arch manifest-list digest;
- creating, starting, stopping, and removing a container;
- reading back the digest of a created container.

The real backend (``docker_py.py``) treats the Engine REST API as the
authority beneath and never parses CLI output.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from typing import Any, Protocol

from tensorstead.ports.runtime_adapter import ContainerRequirements


@dataclass(frozen=True)
class ContainerState:
    """What a container *is*, as distinct from whether the engine knows its name.

    The distinction is the entire point. ``containers.get(name)`` resolves
    an exited container as readily as a running one, so a digest read through it
    proves only that the container was created at some point. Observation needs
    ``running`` to come from the container's actual state, and ``exit_code`` and
    ``detail`` so a dead runtime says why rather than merely reporting absence.
    """

    running: bool
    # The container's own name. Optional because a single-container inspection
    # already knows what it asked for; it matters when a *list* is returned and
    # each element has to identify itself.
    name: str | None = None
    image_digest: str | None = None
    exit_code: int | None = None
    # The engine's own status word ("exited", "restarting", "created"). Carried
    # verbatim rather than mapped to our vocabulary: when a runtime is
    # crash-looping, "restarting" is the fact an operator needs, and it has no
    # equivalent in a two-value running/not-running model.
    detail: str | None = None
    # Container labels, which is where the agent records the coordinator's
    # revision and declared endpoint. Labels live on the container
    # itself, so they survive an agent restart without the agent keeping state.
    labels: dict[str, str] = field(default_factory=dict)
    # How many times the engine has restarted this container. A crash-looping
    # runtime and a healthy one are both "not running" at any given instant;
    # only this distinguishes them.
    restart_count: int | None = None
    # The argv the runtime was actually launched with, entrypoint included.
    # Read back from the container rather than recomputed from the deployment
    # revision: the point is to show what *is* running, and a value derived
    # from config could only ever agree with config.
    command: list[str] = field(default_factory=list)
    # The container's actual environment, read back rather than recomputed
    # from the deployment revision -- the same reasoning as
    # ``command``. Exists so observation can test readiness with, and report
    # on, the credential genuinely installed for *this* container instead of
    # a node-wide fallback that may not be what the runtime was actually
    # given. Never rendered in an API response
    # or log line by anything that reads it.
    environment: dict[str, str] = field(default_factory=dict)


class ContainerEngine(Protocol):
    """The narrow container-execution surface the agent needs."""

    def pull_image(self, reference: str) -> str:
        """Pull ``reference`` and return its platform-specific digest.

        Raises ``ImagePullError`` when the digest cannot be resolved. The
        caller records the failure rather than substituting a different tag
        (the record must describe what actually happened).
        """

    def image_digest(self, reference: str) -> str | None:
        """Return the digest of an image already present locally, or None.

        Distinct from ``pull_image``: it asks what the node *has*, and never
        reaches a registry. A locally produced image exists only on
        the node that built or imported it, so resolving it through a pull can
        never succeed — the reference names no registry repository. Without
        this, an image the product itself produced could be built but never
        run, which made the whole local-image path unreachable from the deployment
        path.
        """

    def list_images(self) -> builtins.list[dict[str, str]]:
        """Every image the daemon holds, as ``{"reference", "digest"}`` rows.

        Exists so the coordinator can compare its image records against the
        node rather than trusting them. Nothing in the product removed an
        image until explicit deletion was made real, so every out-of-band ``docker rmi``
        orphaned a record silently, and the estate reached 50 records against
        25 real objects with nothing comparing them.

        One row per tag, so an object carrying three tags appears three times —
        which is the shape the coordinator's records are in, one per reference.
        An untagged image is reported under its id.
        """

    def build_image(
        self,
        *,
        reference: str,
        base_image: str,
        steps: list[str],
        entrypoint: list[str] | None = None,
    ) -> str:
        """Build ``reference`` from ``base_image`` plus ordered ``steps``.

        Returns the produced image's content-addressable identifier. That is
        **not** a registry digest — a locally built image has none — and
        presenting it as one is forbidden.

        The steps are a recorded artifact, never an interactive session: there
        is no shell into a build and no operation accepting a command to run on
        the host itself.
        """

    def import_image(self, *, archive_path: str) -> str:
        """Load an image archive and return the produced image identifier.

        For estates that cannot reach a registry, and for vendor-supplied
        images. The caller verifies the archive against its stated digest
        before the image is recorded as an available Tensorstead image,
        and partially acquired models are verified under the same rule.
        That verification necessarily happens *after* this call, not
        before it -- the engine has no way to compute an archive's resulting
        image id without loading it. A caller whose comparison fails must
        call ``remove_image`` with the id this returned: an unrecorded image
        left in the daemon is a poisoned tag a later start could pick up, not
        merely wasted storage.
        """

    def verify_image_materializable(self, *, image_id: str) -> None:
        """Refuse ``image_id`` unless a container can actually be made from it.

        A loaded image can carry the right identity and still be unusable. The
        daemon's content store is addressed by digest, so a manifest whose
        config and layer blobs never arrived is still *recorded* under the id
        the manifest names: ``import_image`` returns the expected id,
        ``image_digest`` resolves, and ``images`` lists it. Only creating a
        container reads the config blob, and that is where it fails::

            failed to read config content: NotFound:
            content digest sha256:d670b4...: not found

        which the agent then reported as a bare HTTP 500 from the deployment
        route, pointing at the deployment rather than at the image.

        Worse, the incomplete content is *sticky*. A later ``docker load`` of
        the same image deduplicates against the store, adds the tag, and
        reports success without repairing anything -- so a re-distribution
        under a fresh tag reproduces the fault exactly, and looks like a second
        independent failure rather than the same one. Nothing short of removing
        the image clears it.

        This is therefore the seam's own check, not the caller's: identity
        answers *which* image arrived and this answers *whether it arrived
        whole*. Both checks must hold before distribution may be reported
        successful.

        Raises ``ImageBuildError`` when no container can be created. The caller
        removes the image on failure, exactly as it does for an identity
        mismatch -- an image that cannot be materialized is a poisoned tag in
        the same way, and additionally poisons every later load of the same id.
        """

    def remove_image(self, *, image_id: str, force: bool = False) -> None:
        """Remove an image the daemon holds, by id or by reference.

        Two callers, and they want opposite things of a shared object.

        The identity-mismatch path unwinds an ``import_image`` the caller then
        refused, and passes ``force=True``: an id an identity check just
        rejected must not survive because it happens to share a tag with
        something else on the node.

        The operator-facing delete passes ``force=False`` and needs
        the daemon's own refusals surfaced rather than overridden. It raises
        ``ImageInUseError`` when a container still holds the image, so the
        coordinator can keep its record and say so, and ``ImageNotPresentError``
        when the daemon does not have it, so the coordinator can tell "I
        removed it" from "it was already gone" and reap the stale record
        either way.

        This docstring used to say this was "not a general image-lifecycle
        primitive" and that ``ModelService.delete_image`` "only forgets the
        coordinator's record and does not reach the daemon at all". Both were
        true, and the second cost 430 GB of images that no record named any
        more. The reasoning rested on ordinary retention, which keeps
        images across *deployment* removal; explicit deletion is a
        separate operation. Retention by default is
        not a reason to make a delete a no-op.
        """

    def export_image(self, *, reference: str, archive_path: str) -> str:
        """Write ``reference`` to ``archive_path`` and return its digest.

        Raises ``ImageBuildError`` when the export succeeds but the archive it
        wrote is not a whole image -- every blob its own manifest names must be
        in it. An image can run on the node that produced it and still have no
        exportable content, in which case ``save`` returns cleanly having
        written only index and manifest.

        Used to distribute one produced image to the other participating
        nodes, so every node holds the same identifier. Building
        separately on each node would yield different identifiers for the same
        steps, which breaks the comparison silently.
        """

    def create_container(
        self,
        *,
        name: str,
        image: str,
        endpoint: str,
        model_path: str,
        command_args: list[str],
        entrypoint: list[str] | None = None,
        environment: dict[str, str] | None = None,
        labels: dict[str, str] | None = None,
        requirements: ContainerRequirements | None = None,
        cache_path: str | None = None,
        extra_model_paths: list[str] | None = None,
    ) -> str:
        """Create (but do not start) a container.

        Returns a stable identifier for the container. ``image`` is the
        digest-resolved reference; ``endpoint`` is the host:port the runtime
        binds; ``model_path`` is mounted read-only at that same path inside the
        container; and ``command_args`` are the runtime-launch arguments
        produced by the runtime adapter. ``entrypoint`` is optional
        because each runtime image owns its executable contract. ``environment``
        is for runtime-only secrets that must not appear in the process command
        line or coordinator state. ``labels`` is durable non-secret metadata the
        agent reads back at observation time — never credentials, which have a
        dedicated mechanism and must not be written to container metadata.
        """

    def start_container(self, name: str) -> None:
        """Start a previously-created container."""

    def stop_container(self, name: str) -> None:
        """Stop a running container."""

    def remove_container(self, name: str) -> None:
        """Remove a container entirely."""

    def inspect_container(self, name: str) -> ContainerState | None:
        """Return ``name``'s actual state, or None if no such container exists.

        The authority for whether a deployment is running. Distinct from
        ``get_digest``, which answers only "what image is this container built
        from" and is true of a container that died an hour ago.
        """

    def list_managed_containers(self) -> list[ContainerState]:
        """Every container in the ``tensorstead-`` namespace, running or not.

        Enumeration, not judgment. The agent cannot decide whether a container
        is *unexpected*: only the coordinator knows which deployments it
        recorded. So this reports what is on the node and the comparison
        happens where the records live (observed and declared
        stay separate until something deliberately compares them).

        Includes stopped containers. A runtime that died and left its container
        behind is exactly the case worth surfacing, and filtering to
        running ones would hide it.
        """

    def run_once(
        self,
        *,
        image: str,
        entrypoint: list[str],
        command: list[str],
        script: str | None = None,
        with_accelerator: bool = False,
        timeout_seconds: float = 300.0,
    ) -> str:
        """Run a container to completion and return its combined output.

        For asking an image about itself. The container is removed whatever
        happens, mounts no model, publishes no port, joins no network, and is
        never started as a deployment -- nothing about it is a serving instance,
        and the agent has no other reason to run a container that exits.

        ``script`` is written into the container and named as the last argument,
        rather than being passed as ``-c``. A probe is several lines of Python;
        threading it through a shell as one argument is where quoting defects
        live, and this seam should not have any.

        ``with_accelerator`` because a vLLM probe needs one: constructing its
        argument parser builds config defaults that require device detection,
        and without it the probe fails with "Failed to infer device type"
        (measured on the appliance). An adapter that does not
        need a device does not ask for one.
        """

    def image_user(self, reference: str) -> str | None:
        """The user an image declares it runs as, or None if it does not say.

        Needed because the agent provisions a durable cache directory on the
        host and the container must be able to write to it. The
        alternative -- a world-writable directory -- would be a path from any
        local account to code executing inside the inference container, since a
        compile cache holds executable artefacts.

        ``None`` means unknown, not root: an image that declares nothing runs
        as root by Docker's default, but an image declaring a *name* we cannot
        resolve to a uid is genuinely unknown and must not be guessed at.
        """

    def container_processes(self, name: str) -> list[str] | None:
        """Command lines of the processes actually running, or None.

        Distinct from ``ContainerState.command``, which is what the container
        was *configured* with. An entrypoint that ignores the arguments it was
        handed and builds its own leaves those two disagreeing, and only this
        side sees it: the configured argv looks exactly right.

        The contract forbids that -- an entrypoint may prepare, and may not decide --
        because a deployment whose declared ``runtime_config`` the runtime never
        read is a record that describes nothing. The rule needs something that
        can notice.

        Read through the Engine's process listing, not by executing anything in
        the container. ``None`` when the engine cannot say.
        """

    def container_logs(self, name: str, *, tail: int = 200) -> str | None:
        """Return the last ``tail`` lines the container wrote, or None if absent.

        The runtime's own account of itself. Observation established *that* a
        deployment stopped serving; this is the only thing that says why —
        vLLM's traceback, its complaint about an argument, its report of how
        much memory it wanted.

        **Read through, never stored.** The agent fetches on request and
        returns; nothing in the product persists a line of it. That is what
        keeps this on the management side of the boundary: a runtime may log
        whatever it likes, including request content, and the product must not
        become the place that content comes to rest.
        """

    def get_digest(self, name: str) -> str | None:
        """Return the platform-specific digest of ``name``, or None if absent.

        Answers a question about the image, not about liveness. Callers deciding
        whether a deployment is running MUST use ``inspect_container``.
        """


class ImagePullError(Exception):
    """An image reference could not be resolved to a platform-specific digest.

    Carries a structured ``code`` (``image_digest_unresolved``, per the agent
    contract) so the operation record can name what failed.
    """

    def __init__(self, message: str, *, reference: str) -> None:
        super().__init__(message)
        self.code = "image_digest_unresolved"
        self.reference = reference


class ImageInUseError(Exception):
    """An image could not be removed because a container still holds it.

    Distinct from a generic failure because the operator's next step differs:
    this is not "the delete broke", it is "something is still using this", and
    the record must survive to name the bytes.
    """

    def __init__(self, message: str, *, reference: str) -> None:
        super().__init__(message)
        self.code = "image_in_use"
        self.reference = reference


class ImageNotPresentError(Exception):
    """The daemon does not hold the image a caller asked to remove.

    Not an error the operator caused, and not a reason to keep a record: it is
    the signal that a record has outlived its object and should be reaped.
    """

    def __init__(self, message: str, *, reference: str) -> None:
        super().__init__(message)
        self.code = "image_not_present"
        self.reference = reference


class ImageBuildError(Exception):
    """A build, import, or export failed on the node.

    Carries the reference so the failure names what was being produced rather
    than only that something went wrong, and an optional ``detail`` for
    evidence that belongs in the operation record in structured form -- a build
    log tail and the failing step, which a message can only paste.

    ``detail`` must stay bounded. It is copied into the agent's JSON response,
    from there into the coordinator's operation record, and from there into
    every read of that operation forever.
    """

    def __init__(
        self, message: str, *, reference: str, detail: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.reference = reference
        self.detail = dict(detail or {})
