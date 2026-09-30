"""docker-py container backend.

Implements the narrow container-engine seam over the Docker Engine REST API via
``docker`` (docker-py). The Engine REST API is the authority beneath; we never
parse CLI output. When the API is unavailable, ``ImagePullError`` is raised so
the operation record names the failure rather than guessing.

``pull_image`` returns the **platform-specific** digest — the
digest of the image that actually ran on this host, not the multi-arch
manifest-list digest. On an arm64 host the two differ, and only the former
identifies the running image. We use the image id resolved by the
Engine for the *host architecture* rather than the manifest-list reference.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tensorstead.agent.container_engine.base import (
    ContainerState,
    ImageBuildError,
    ImageInUseError,
    ImageNotPresentError,
    ImagePullError,
)
from tensorstead.ports.runtime_adapter import ContainerRequirements

_LEGACY_API_PORT = 8000

# Read timeout for calls that move a whole image archive over the Docker socket
# Thirty minutes: a DSpark runtime image is tens of
# gigabytes and Docker unpacks it before answering, and the machines this runs
# on are the slowest link. Bounded on purpose -- an import that will never
# finish must still terminate and be reportable.
_ARCHIVE_TIMEOUT_SECONDS = 1800

# Bounds on a one-shot probe. The agent reads a probe's
# output into memory, so an image that prints without stopping would otherwise
# be limited only by the node's RAM. The payload is the last line, which a tail
# always keeps; the character cap is the second bound, for one enormous line.
_PROBE_LOG_LINES = 2000
_PROBE_OUTPUT_CHARS = 1_000_000
# A parser probe runs one process. Generous by two orders of magnitude, and
# still a ceiling.
_PROBE_PIDS_LIMIT = 256

# Bounds on what a *failed* build keeps of its log. A build log is the only
# artifact that says why a step exited non-zero, and this backend discarded it
# on both paths -- so a failed build reached the operator as a node name and an
# exit code, which is not diagnosable. The operator's recourse was to reproduce
# the build by hand over SSH: outside the product, outside the operation record,
# and without the resource guardrails a managed build runs under.
#
# The tail is the part that matters -- a compiler names its error last -- and it
# is bounded twice, by lines and by characters, because one build step can emit
# a single line of unbounded length.
_BUILD_LOG_LINES = 200
_BUILD_LOG_CHARS = 20_000
# How much of the log travels in the human-readable message, as opposed to the
# structured detail beside it. Smaller on purpose: the message is read first and
# in full, the detail is read when the message was not enough.
_BUILD_MESSAGE_LOG_LINES = 40
# A recorded step can be enormous -- a base64 build context is the shape this
# estate actually ships, at ~259K characters -- so naming which step failed must
# never paste the step back.
_BUILD_STEP_ECHO_CHARS = 400

# How long this node lets its own build run.
#
# Deliberately *below* the coordinator's 14400s hop bound. If the
# agent expires first it produces the diagnosable failure -- failing step, log
# tail -- whereas if the caller expires first all anyone gets is a timeout on a
# hop. Same total patience, better record; the difference is only which end
# gives up.
_BUILDX_TIMEOUT_SECONDS = 14100.0
# docker's own reason for a failed ``RUN`` quotes the whole command, and the
# command is the recorded step. Bounded for the same reason, and elided in the
# middle rather than the end: the head identifies the step and the tail carries
# the exit code, so a plain truncation loses precisely the useful half.
_BUILD_REASON_CHARS = 2_000

# The classic builder announces each instruction before running it.
_BUILD_STEP_MARKER = re.compile(r"^Step\s+(\d+)/(\d+)\s*:")

# BuildKit numbers the same instructions differently, and interleaves them.
#
# Classic prints one linear stream, so "the last Step marker seen" is the step
# that failed. BuildKit runs stages concurrently and tags every line with a
# vertex id -- ``#5 [2/4] RUN ...`` -- so the last marker in the stream is
# whichever vertex printed most recently, not necessarily the one that broke.
# Reading it that way would name the wrong step with total confidence, which is
# worse than naming none.
#
# So the vertex id is the join: map ``#N`` to its ``[i/n]``, then read the id
# off the ``#N ERROR`` line. Both regexes below exist for that single purpose.
_BUILDKIT_VERTEX_STEP = re.compile(r"^#(\d+)\s+\[(?:[^\]]*\s)?(\d+)/(\d+)\]")
_BUILDKIT_VERTEX_ERROR = re.compile(r"^#(\d+)\s+ERROR\b")

# Where a probe script is staged before being bind-mounted into its container.
#
# **Not /tmp**, and the reason is not obvious enough to leave unwritten. The
# agent unit sets `PrivateTmp=true`, so the agent's /tmp is a private mount
# namespace. A path written there does not exist from the *Docker daemon's*
# point of view -- and Docker, asked to bind-mount a source it cannot see,
# silently creates an empty **directory** at the destination. The container then
# ran `python3 /probe.py` against a directory and reported "can't find
# `__main__` module in '/probe.py'", which reads as a broken script rather than
# a namespace disagreement.
#
# The agent's managed state directory is a real host path both processes see.
# Found on the first hardware exercise of the probe; unreachable in tests,
# because a fake Docker client shares the test process's view of the filesystem
# and so can never disagree with it.
_PROBE_STAGING_DIR = Path("/var/lib/tensorstead/state")


def _container_api_port(requirements: ContainerRequirements | None) -> int:
    """The port the runtime listens on inside its container.

    Every shipped adapter declares this, and a guardrail test asserts they do.
    The fallback covers only a call that supplied no requirements at all — the
    older shape — and is deliberately the historical value so that path
    behaves exactly as it did rather than acquiring a new opinion.
    """
    if requirements is not None and requirements.api_port:
        return int(requirements.api_port)
    return _LEGACY_API_PORT


def _elide(text: str, limit: int) -> str:
    """Bound ``text`` keeping both ends, since both ends carry meaning here."""
    if len(text) <= limit:
        return text
    head, tail = limit // 3, limit - limit // 3
    return f"{text[:head]}\n[... {len(text) - limit} characters elided ...]\n{text[-tail:]}"


class _Unset:
    """Distinguishes "not yet resolved" from a cached ``None``."""


_UNSET = _Unset()


def _docker_cli() -> str | None:
    """Absolute path to the ``docker`` CLI, or ``None`` if it is not installed.

    Resolved rather than invoked as a bare name: an absolute path is what makes
    the subprocess call auditable, and it doubles as the availability check.
    """
    return shutil.which("docker")


def _buildx_available(cli: str) -> bool:
    """Whether this host has a working buildx builder.

    Asked by running it, not by inspecting a version string: buildx can be
    installed as a plugin and still have no usable builder instance, and the
    only reliable answer is whether it responds.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path, fixed argv, no shell
            [cli, "buildx", "version"],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _build_log_lines(exc: BaseException) -> list[str]:
    """Every line the builder printed, recovered from a docker-py ``BuildError``.

    ``BuildError`` carries ``build_log`` — the same stream the successful path
    returns — and it is the only place the *reason* for a non-zero step exists.
    docker-py's own ``str()`` is the terminal ``errorDetail`` message, which for
    a failed ``RUN`` says which command returned which code and nothing about
    what the command printed. That is the difference between a failure an
    operator can act on and one they can only reproduce by hand.

    ``build_log`` is a live ``itertools.tee`` over a socket, so iterating it can
    raise where the successful path never would. Whatever it yielded before that
    is kept: a partial log is worth incomparably more than none, and this
    function exists on an error path that must not raise a second error over the
    first.
    """
    log = getattr(exc, "build_log", None)
    if log is None:
        return []
    lines: list[str] = []
    # Suppressed, not handled: a truncated log is still the evidence, and this
    # runs on an error path that must not raise a second failure over the first.
    with contextlib.suppress(Exception):
        for chunk in log:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("stream") or "")
            detail = chunk.get("errorDetail")
            if isinstance(detail, dict) and detail.get("message"):
                text = f"{text}{detail['message']}"
            elif not text and chunk.get("error"):
                text = str(chunk["error"])
            lines.extend(line for line in text.splitlines() if line.strip())
    return lines


def _failing_step_index(lines: list[str]) -> int | None:
    """Which recorded step the builder was on when it stopped, 0-based.

    ``render_dockerfile`` emits ``FROM`` followed by exactly one ``RUN`` per
    recorded step, so instruction *N* is recorded step *N - 2* under either
    builder. Negative means it never got past ``FROM``; ``None`` means no marker
    was seen at all, which is what a builder that failed before starting looks
    like.

    BuildKit is tried first and by vertex correlation rather than by recency,
    for the reason given at ``_BUILDKIT_VERTEX_STEP``. The classic scan is the
    fallback and is unchanged.
    """
    instruction = _buildkit_failing_instruction(lines)
    if instruction is not None:
        return instruction - 2
    for line in reversed(lines):
        match = _BUILD_STEP_MARKER.match(line.strip())
        if match:
            return int(match.group(1)) - 2
    return None


def _buildkit_failing_instruction(lines: list[str]) -> int | None:
    """The 1-based instruction number BuildKit reported an error against.

    ``None`` when the output is not BuildKit's, or when it is but no vertex
    reported an error -- a build killed by a signal, say. Returning ``None``
    rather than guessing keeps the caller's "no step named" path honest.
    """
    vertex_step: dict[str, int] = {}
    failed: str | None = None
    for raw in lines:
        line = raw.strip()
        step = _BUILDKIT_VERTEX_STEP.match(line)
        if step:
            vertex_step[step.group(1)] = int(step.group(2))
            continue
        error = _BUILDKIT_VERTEX_ERROR.match(line)
        if error:
            # Last error wins: a failing vertex can print more than one, and a
            # later one is not a different failure.
            failed = error.group(1)
    if failed is None:
        return None
    return vertex_step.get(failed)


def _build_failure(exc: BaseException, *, reference: str, steps: list[str]) -> ImageBuildError:
    """Turn a builder exception into a failure that names what actually broke.

    Three facts go in, and all three were missing when a real build failed on
    this estate: which recorded step failed, what the builder printed before it
    did, and docker's own terminal reason. The operation record carried the node
    and the exit code, which named the failure without describing it.

    The failing step is *identified* but only echoed in truncated form. A build
    step here can be hundreds of kilobytes — an encoded build context is the
    shape this estate ships, absent a real one (the record holds steps, not
    files) — and a failure record that pastes it back is unreadable for the same
    reason the original was.
    """
    return _build_failure_from_lines(
        _build_log_lines(exc), reason=str(exc), reference=reference, steps=steps
    )


def _build_failure_from_lines(
    lines: list[str], *, reason: str, reference: str, steps: list[str]
) -> ImageBuildError:
    """Format a build failure from log lines that are already in hand.

    Split out of ``_build_failure`` when buildx arrived: the classic path
    recovers its log from a docker-py exception, while buildx hands it over as
    subprocess output. One formatter for both, so the two builders cannot
    produce differently-shaped failure records for the same defect.
    """
    tail = lines[-_BUILD_LOG_LINES:]
    log_tail = "\n".join(tail)[-_BUILD_LOG_CHARS:]

    detail: dict[str, Any] = {"reference": reference, "build_log_tail": log_tail}
    index = _failing_step_index(lines)
    where = ""
    if index is not None and 0 <= index < len(steps):
        detail["failing_step_index"] = index
        detail["failing_step"] = steps[index][:_BUILD_STEP_ECHO_CHARS]
        where = f" at recorded step {index + 1} of {len(steps)}"
    elif index is not None:
        # ``FROM``, or an ``ENTRYPOINT`` past the last recorded step. Named as
        # not-a-step rather than silently misattributed to one.
        where = " outside the recorded steps (base image or entrypoint)"

    message = f"build of {reference!r} failed{where}: {_elide(reason, _BUILD_REASON_CHARS)}"
    if tail:
        shown = tail[-_BUILD_MESSAGE_LOG_LINES:]
        body = "\n".join(shown)[-_BUILD_LOG_CHARS:]
        message = f"{message}\n--- build log, last {len(shown)} of {len(lines)} lines ---\n{body}"
    return ImageBuildError(message, reference=reference, detail=detail)


def render_dockerfile(
    *, base_image: str, steps: list[str], entrypoint: list[str] | None = None
) -> str:
    """Render one recorded build spec as a Dockerfile.

    Each recorded step becomes exactly **one** ``RUN`` instruction, in exec form
    naming the shell explicitly.

    Shell form — ``RUN {step}`` — prefixes only the step's *first physical
    line*. A step containing a heredoc, or any newline at all, therefore emitted
    its continuation at Dockerfile top level, where the parser read it as an
    instruction: the first real DSpark build died on ``unknown instruction:
    chmod`` from a step whose trailing ``chmod +x`` was ordinary shell input
    The step was recorded correctly and rendered wrongly.

    Exec form carries the whole step as a single JSON string argument, so its
    newlines stay shell input and cannot change Dockerfile grammar. This is not
    a widening: shell form on Linux already compiles to ``/bin/sh -c``, so the
    same shell interprets the same text — the only difference is where the
    instruction boundary falls.
    """
    lines = [f"FROM {base_image}"]
    lines += [f"RUN {json.dumps(['/bin/sh', '-c', step])}" for step in steps]
    if entrypoint:
        lines.append("ENTRYPOINT " + json.dumps(list(entrypoint)))
    return "\n".join(lines)


def _archive_omits_its_own_blobs(archive_path: str) -> str | None:
    """Name the content an archive's manifest declares but does not carry.

    Returns ``None`` when the archive is self-contained, otherwise a short
    description of what is missing.

    This exists because ``docker save`` can succeed and produce an archive with
    no image content in it. On Docker 29 the containerd image store is the
    default, and an image produced by the **classic** builder -- which is what
    docker-py's ``images.build`` drives, there being no BuildKit support in the
    SDK -- is registered in a form that runs locally but whose layer blobs are
    not in the content store. Exporting it yields index and manifest and
    nothing else. Measured on this estate's own hardware:

        classic-builder image, 31 GB on disk   ->  docker save = 16,896 bytes
        the same recipe built with BuildKit    ->  docker save = 9,814,834,688
        a pulled base image                    ->  docker save = 10,533,214,208

    The 16 KB archive transfers fine, loads without error, and reports the
    expected image id on the far side, because the id comes from the manifest
    and the manifest is the part that *is* present.

    The check is self-referential on purpose: every digest the archive's own
    index and manifests name must be present in the archive. That needs no
    comparison against the daemon, so it cannot drift from what the daemon
    thinks, and it is exactly the property "this archive is a whole image".
    """
    import tarfile

    try:
        with tarfile.open(archive_path, "r:*") as tar:
            present = {name.lstrip("./") for name in tar.getnames()}

            def _blob(digest: str) -> str:
                algorithm, _, hexdigest = digest.partition(":")
                return f"blobs/{algorithm}/{hexdigest}"

            def _read(name: str) -> Any:
                handle = tar.extractfile(name)
                if handle is None:
                    return None
                return json.loads(handle.read().decode("utf-8"))

            if "index.json" not in present:
                # Not an OCI layout. Older archive shapes are not produced by
                # the daemons this runs against; unrecognised is not the same as
                # broken, so this declines to judge rather than guessing.
                return None

            index = _read("index.json") or {}
            missing: list[str] = []
            for descriptor in index.get("manifests", []):
                digest = descriptor.get("digest", "")
                if not digest or _blob(digest) not in present:
                    missing.append(f"manifest {digest or '<unnamed>'}")
                    continue
                manifest = _read(_blob(digest)) or {}
                # An index may point at another index (a multi-platform image).
                # Its entries are checked on the next pass rather than treated
                # as a config-bearing manifest.
                for nested in manifest.get("manifests", []):
                    nested_digest = nested.get("digest", "")
                    if not nested_digest or _blob(nested_digest) not in present:
                        missing.append(f"manifest {nested_digest or '<unnamed>'}")
                config_digest = (manifest.get("config") or {}).get("digest")
                if config_digest and _blob(config_digest) not in present:
                    missing.append(f"config {config_digest}")
                for layer in manifest.get("layers", []):
                    layer_digest = layer.get("digest", "")
                    if layer_digest and _blob(layer_digest) not in present:
                        missing.append(f"layer {layer_digest}")

            if not missing:
                return None
            shown = ", ".join(missing[:3])
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            return f"{len(missing)} blob(s) its manifest names are absent: {shown}{more}"
    except Exception as exc:
        # An archive that cannot be read as one is its own answer.
        return f"the archive could not be read: {exc}"


class DockerEngine:
    """A ``ContainerEngine`` backed by the Docker Engine REST API.

    ``client`` is a ``docker.DockerClient`` (docker-py). Importing ``docker``
    and constructing the client happen lazily at call time so the module
    imports cleanly on a host without the Docker SDK, keeping agent
    dependencies optional (pyproject ``agent`` group).
    """

    def __init__(
        self,
        client: Any | None = None,
        staging_dir: Path | None = None,
        prefer_buildx: bool | None = None,
    ) -> None:
        # Whether a caller *supplied* a client, as distinct from this class
        # having lazily built one. ``self._client is not None`` conflated the
        # two and made the archive timeout inert in production: see
        # ``_docker_for_archives``.
        self._client_was_supplied = client is not None
        self._client = client
        self._archive_client: Any | None = None
        # Explicit for tests; on a deployed agent the managed state directory
        # always exists because the Ansible role creates it. The fallback is for
        # development hosts that have neither that directory nor a Docker daemon
        # whose filesystem view could differ from ours.
        self._staging_dir = staging_dir
        # Which builder ``build_image`` uses. ``None`` means "ask the host",
        # resolved once and cached in ``_buildx_cli``.
        #
        # Explicit rather than always-autodetect because otherwise this engine's
        # behaviour depends on whether the machine it happens to run on has
        # buildx installed -- which makes a caller that injected a fake client
        # still reach out to the host, and makes the same test pass or fail
        # depending on the developer's laptop. Autodetection belongs in
        # production; determinism belongs everywhere.
        self._prefer_buildx = prefer_buildx
        self._buildx_cli: str | _Unset | None = _UNSET

    def _docker(self) -> Any:
        if self._client is not None:
            return self._client
        import docker  # type: ignore[import-untyped]  # docker-py ships no stubs

        self._client = docker.from_env()
        return self._client

    def _docker_for_archives(self) -> Any:
        """A client whose read timeout suits moving a whole image archive.

        The SDK's 60s default is right for ordinary engine calls and wrong for
        ``images.load``, which sends a multi-gigabyte archive over the Unix
        socket and then waits for a single response while Docker unpacks it.
        The first successful peer transfer died there::

            archive for 'local/dspark-deepseek-v4-flash:0.1.1' could not be
            loaded: Read timed out. (read timeout=60)

        after five minutes of work that had already succeeded.

        A **separate** client, not a raised global default: a lightweight call
        that hangs should still fail in a minute rather than thirty. Bounded
        rather than disabled, for the same reason -- an import that will never
        finish must still end.

        Streaming calls do not need this. ``images.build`` reads progress
        continuously, so the read timeout is never approached; the five-minute
        build on the source node completed fine.

        **The condition below is ``_client_was_supplied``, not
        ``self._client is not None``, and the difference was a live defect.**
        ``_docker`` caches its lazily built 60s client in the same attribute, so
        the earlier test returned it to every archive call that followed an
        ordinary one -- which is every real one: ``build_and_distribute`` calls
        ``build_image`` on the source before ``export_image``. The timeout was
        declared and never reached. Only a test on a *fresh* engine passed, and
        that is the test that had been written.
        """
        if self._client_was_supplied:
            return self._client
        if self._archive_client is None:
            import docker

            self._archive_client = docker.from_env(timeout=_ARCHIVE_TIMEOUT_SECONDS)
        return self._archive_client

    def pull_image(self, reference: str) -> str:
        try:
            image = self._docker().images.pull(reference)
        except Exception as exc:  # docker.errors.* — engine-level failure
            raise ImagePullError(
                f"could not pull image {reference!r}: {exc}",
                reference=reference,
            ) from exc
        # ``image.id`` is the platform-specific image id for the host the
        # Engine resolved. Normalise to the ``sha256:`` form.
        image_id: str = image.id
        digest = image_id if image_id.startswith("sha256:") else f"sha256:{image_id}"
        return digest

    def image_digest(self, reference: str) -> str | None:
        """Return the local image's identifier, or None when it is absent.

        ``images.get`` consults only what the daemon already holds, so a
        locally built reference resolves and a registry reference that has
        never been pulled returns None rather than raising.
        """
        try:
            image = self._docker().images.get(reference)
        except Exception:
            return None
        image_id = getattr(image, "id", None)
        if not image_id:
            return None
        return image_id if image_id.startswith("sha256:") else f"sha256:{image_id}"

    def build_image(
        self,
        *,
        reference: str,
        base_image: str,
        steps: list[str],
        entrypoint: list[str] | None = None,
    ) -> str:
        """Build via the engine's own build API; no Dockerfile is left on disk.

        ``entrypoint`` makes the produced image self-starting: whatever it
        names runs when the container starts, before anything the deployment
        supplies. Emitted in exec form so no shell is interposed and the
        arguments are exactly what was recorded.
        """
        dockerfile = render_dockerfile(base_image=base_image, steps=steps, entrypoint=entrypoint)

        # BuildKit first, and this is the whole of the fix.
        #
        # docker-py has no BuildKit support, so ``images.build`` drives the
        # *classic* builder. Under Docker 29's containerd image store a
        # classic-built image runs locally and exports as index-plus-manifest
        # with none of its layer blobs -- 16 KB standing in for 31 GB, measured
        # on this estate. It therefore cannot be distributed to a second node,
        # which is what made ``image_build`` unusable for every multi-node
        # deployment and sent the operator to a hand-run ``docker buildx``
        # followed by an import.
        #
        # There is no API route to BuildKit: it is the CLI or nothing. That is a
        # real departure from this module's rule of never parsing CLI output,
        # taken deliberately and confined to this one method -- the alternative
        # is a build path that cannot produce a distributable image, which is
        # not a build path.
        cli = self._resolve_buildx()
        if cli is not None:
            return self._build_with_buildx(
                cli, dockerfile=dockerfile, reference=reference, steps=steps
            )

        # No buildx: the classic builder, exactly as before. Not an error,
        # because a locally-built image that is never distributed still works
        # and this is what every build did until now. If it *is* distributed,
        # the blob check refuses it loudly rather than shipping a hollow archive
        # the other half of that same work, so the fallback is no worse than today.
        try:
            image, _logs = self._docker().images.build(
                fileobj=io.BytesIO(dockerfile.encode("utf-8")),
                tag=reference,
                rm=True,
                pull=False,
            )
        except Exception as exc:  # docker.errors.BuildError and friends
            # The exception carries the build log; ``str(exc)`` does not. Raising
            # only the latter is what made a failed build undiagnosable from the
            # operation record.
            raise _build_failure(exc, reference=reference, steps=steps) from exc
        return str(image.id)

    def _resolve_buildx(self) -> str | None:
        """The docker CLI to build through, or ``None`` to use the classic path.

        Resolved at most once. ``prefer_buildx=False`` refuses it outright;
        ``True`` still requires the CLI to actually be there, because a
        preference is not a capability.
        """
        if self._prefer_buildx is False:
            return None
        if not isinstance(self._buildx_cli, _Unset):
            return self._buildx_cli
        cli = _docker_cli()
        resolved = cli if (cli is not None and _buildx_available(cli)) else None
        self._buildx_cli = resolved
        return resolved

    def _build_with_buildx(
        self, cli: str, *, dockerfile: str, reference: str, steps: list[str]
    ) -> str:
        """Build through BuildKit so the result carries its own layers.

        ``--load`` is the point: it writes the finished image into the local
        image store, where ``docker save`` can export it whole. Without it
        buildx leaves the result in its cache and the export is empty again,
        which is the bug wearing a different hat.

        ``--progress=plain`` because the default renderer draws a live TTY
        display whose escape sequences are unreadable as a log -- and the log is
        what the build log exists to preserve.

        The context is an empty directory, deliberately. Recorded build specs
        carry steps and no files, so there is nothing to send, and an
        empty context means nothing on this node's disk can leak into an image
        by being in the wrong working directory.
        """
        with tempfile.TemporaryDirectory(prefix="tensorstead-buildx-") as context:
            path = Path(context) / "Dockerfile"
            path.write_text(dockerfile, encoding="utf-8")
            argv = [
                cli,
                "buildx",
                "build",
                "--progress=plain",
                "--load",
                "--tag",
                reference,
                "--file",
                str(path),
                context,
            ]
            try:
                completed = subprocess.run(  # noqa: S603 - absolute path, fixed argv, no shell
                    argv,
                    capture_output=True,
                    check=False,
                    text=True,
                    errors="replace",
                    timeout=_BUILDX_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise ImageBuildError(
                    f"build of {reference!r} exceeded this node's "
                    f"{_BUILDX_TIMEOUT_SECONDS:.0f}s bound and was terminated",
                    reference=reference,
                    detail={"reference": reference, "timeout_seconds": _BUILDX_TIMEOUT_SECONDS},
                ) from exc
            except (OSError, subprocess.SubprocessError) as exc:
                raise ImageBuildError(
                    f"build of {reference!r} could not be started: {exc}", reference=reference
                ) from exc

            # BuildKit writes its progress to stderr and the built image's own
            # output to stdout. Both are the build log; keeping only one loses
            # either the step markers or the compiler's message.
            lines = [
                line
                for stream in (completed.stdout, completed.stderr)
                for line in (stream or "").splitlines()
                if line.strip()
            ]
            if completed.returncode != 0:
                raise _build_failure_from_lines(
                    lines,
                    reason=f"docker buildx build exited {completed.returncode}",
                    reference=reference,
                    steps=steps,
                )

        # buildx reports no image id on stdout in a form worth parsing, so it is
        # read back from the daemon by the tag just written. A tag that does not
        # resolve after a successful build means something else moved it, which
        # is worth failing on rather than returning an id for the wrong image.
        try:
            return str(self._docker().images.get(reference).id)
        except Exception as exc:
            raise ImageBuildError(
                f"buildx reported success for {reference!r} but the image cannot be "
                f"resolved afterwards: {exc}",
                reference=reference,
            ) from exc

    def _probe_staging_dir(self) -> str | None:
        """A directory the Docker daemon can also see. See _PROBE_STAGING_DIR."""
        if self._staging_dir is not None:
            return str(self._staging_dir)
        return str(_PROBE_STAGING_DIR) if _PROBE_STAGING_DIR.is_dir() else None

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
        """Run a container to completion and return its output."""
        import docker

        create_args: dict[str, Any] = {
            "image": image,
            "entrypoint": list(entrypoint),
            "command": list(command),
            # Nothing to reach and nothing to reach it. A question about an
            # image needs no network, and denying one keeps a probe from
            # becoming a way to make outbound calls from a node.
            "network_mode": "none",
            "detach": True,
            # A parser probe forks nothing. A bound here costs nothing when
            # that holds and contains an image that misbehaves when it does not.
            "pids_limit": _PROBE_PIDS_LIMIT,
        }
        if with_accelerator:
            create_args["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        staged: Path | None = None
        if script is not None:
            # Written to a host file and bind-mounted read-only. The alternative
            # is passing it as an argument, which puts multi-line source through
            # a shell.
            staged_dir = tempfile.mkdtemp(
                prefix="tensorstead-probe-", dir=self._probe_staging_dir()
            )
            staged = Path(staged_dir) / "probe.py"
            staged.write_text(script, encoding="utf-8")
            staged.chmod(0o644)
            create_args["volumes"] = {str(staged): {"bind": "/probe.py", "mode": "ro"}}
            create_args["command"] = [*command, "/probe.py"]

        container = None
        removal_failure: str | None = None
        try:
            container = self._docker().containers.create(**create_args)
            container.start()
            result = container.wait(timeout=timeout_seconds)
            # Bounded read. The probe's payload is its last line, so a tail
            # keeps everything that matters while a runaway image cannot make
            # the agent read its output until the node runs out of memory.
            raw = container.logs(stdout=True, stderr=True, tail=_PROBE_LOG_LINES)
            output = raw.decode("utf-8", "replace")[-_PROBE_OUTPUT_CHARS:]
            if result.get("StatusCode", 1) != 0:
                raise ImageBuildError(
                    f"probe of {image!r} exited {result.get('StatusCode')}: {output[-2000:]}",
                    reference=image,
                )
        except ImageBuildError:
            raise
        except Exception as exc:
            raise ImageBuildError(f"could not probe {image!r}: {exc}", reference=image) from exc
        finally:
            # Removed however this ended. A probe that leaves containers behind
            # is a probe that changes the thing it was asked to describe.
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception as exc:  # surfaced below, not swallowed
                    removal_failure = str(exc)
            if staged is not None:
                with contextlib.suppress(Exception):
                    shutil.rmtree(staged.parent, ignore_errors=True)

        # A failed removal is a failed probe, even though the answer arrived
        # Suppressing it let the route return `known` while
        # an unlabelled container stayed on the node -- the product reporting
        # success for an operation that left the host in a state nobody records.
        # Only reached when the probe itself succeeded: on the error path the
        # original cause propagates, which is the more useful of the two.
        if removal_failure is not None:
            raise ImageBuildError(
                f"probe of {image!r} answered but its container could not be removed "
                f"({removal_failure}); the answer is discarded because a probe that "
                f"leaves state behind has changed the node it was asked to describe",
                reference=image,
            )
        return str(output)

    def import_image(self, *, archive_path: str) -> str:
        try:
            with open(archive_path, "rb") as handle:
                # `handle`, not `handle.read()`: docker-py's `load_image` passes
                # `data` straight through to requests, which streams a file-like
                # object in chunks instead of loading it whole, rather than reading
                # the whole archive into memory.
                images = self._docker_for_archives().images.load(handle)
        except Exception as exc:
            # The timeout is named in the message. A read timeout that does not
            # say which limit was reached leaves an operator unable to tell a
            # too-short bound from a genuinely stuck engine.
            raise ImageBuildError(
                f"{exc} (managed archive timeout {_ARCHIVE_TIMEOUT_SECONDS}s)",
                reference=archive_path,
            ) from exc
        loaded = list(images)
        if not loaded:
            raise ImageBuildError("archive contained no image", reference=archive_path)
        return str(loaded[0].id)

    def verify_image_materializable(self, *, image_id: str) -> None:
        """Create a container from ``image_id`` and remove it again.

        Creating is the whole check and starting would add nothing: reading the
        config blob is what a create does and what an incomplete image cannot
        do. It needs no accelerator, no network, and no model mount, so it is
        cheap enough to run on the arrival path of every distributed image.

        ``image_digest``/``inspect`` deliberately are *not* used here. On the
        containerd image store an image whose config never arrived still
        inspects successfully -- it answers with an empty ``Config``, an empty
        ``RootFS`` and an empty ``Architecture`` rather than an error -- so a
        check built on inspection reads as a pass on exactly the images this
        exists to reject.
        """
        container = None
        try:
            container = self._docker().containers.create(
                image=image_id,
                # Never started, so the entrypoint is irrelevant; overridden
                # only so a create cannot inherit an image's own long argv.
                entrypoint=["/bin/true"],
                command=[],
                network_mode="none",
            )
        except Exception as exc:
            raise ImageBuildError(
                f"{image_id!r} loaded but no container can be created from it: {exc}; "
                f"the image arrived incomplete and must not be reported as distributed",
                reference=image_id,
            ) from exc
        finally:
            if container is not None:
                # A verification that leaves a container behind has changed the
                # node it was asked to describe -- the same rule ``run_once``
                # follows. Unlike a probe, the answer here is
                # already known by this point and a stray never-started
                # container is not worth failing an otherwise good image over,
                # so this is suppressed rather than raised.
                with contextlib.suppress(Exception):
                    container.remove(force=True)

    def remove_image(self, *, image_id: str, force: bool = False) -> None:
        # force=True (identity-mismatch path): an id an identity check just
        # rejected must not survive because it happens to share a layer/tag
        # with something else on the node. noprune=False (the default) still
        # lets now-dangling parent layers go, which is the ordinary
        # storage-reclaim behaviour.
        #
        # force=False (operator delete): the daemon's refusals are the
        # answer, not an obstacle. Removing by *reference* untags, and the
        # object goes when its last tag does -- so deleting one of three tags
        # on one object frees nothing and is still correct, which is what the
        # coordinator's per-reference records mean.
        try:
            self._docker_for_archives().images.remove(image_id, force=force)
        except Exception as exc:
            code = _image_removal_failure(exc)
            if code == "not_present":
                raise ImageNotPresentError(
                    f"the daemon does not hold {image_id!r}", reference=image_id
                ) from exc
            if code == "in_use":
                raise ImageInUseError(
                    f"{image_id!r} is still held by a container: {exc}", reference=image_id
                ) from exc
            raise ImageBuildError(
                f"could not remove image {image_id!r}: {exc}", reference=image_id
            ) from exc

    def list_images(self) -> list[dict[str, str]]:
        """Every image the daemon holds, one row per tag."""
        try:
            images = self._docker().images.list(all=True)
        except Exception as exc:
            raise ImageBuildError(
                f"could not list images on this node: {exc}", reference="<list>"
            ) from exc

        rows: list[dict[str, str]] = []
        for image in images:
            image_id = getattr(image, "id", None)
            if not image_id:
                continue
            digest = image_id if image_id.startswith("sha256:") else f"sha256:{image_id}"
            tags = [t for t in (getattr(image, "tags", None) or []) if t]
            # An untagged image is reported under its id so it is nameable at
            # all. "<none>:<none>" is what the daemon prints and is not a
            # reference anything can act on.
            for tag in tags or [digest]:
                rows.append({"reference": tag, "digest": digest})
        return rows

    def export_image(self, *, reference: str, archive_path: str) -> str:
        # The same archive-scale work in the other direction: the source agent
        # exports before a peer can fetch, and a large export would hit the
        # same 60s default. Found by asking which other call moves an archive
        # rather than by waiting for it to fail on hardware too.
        try:
            client = self._docker_for_archives()
            image = client.images.get(reference)
            with open(archive_path, "wb") as handle:
                for chunk in image.save(named=True):
                    handle.write(chunk)
        except Exception as exc:
            raise ImageBuildError(
                f"{exc} (managed archive timeout {_ARCHIVE_TIMEOUT_SECONDS}s)",
                reference=reference,
            ) from exc

        # A successful ``save`` is not a usable archive. The
        # export that started this estate's TP=2 failure returned cleanly and
        # wrote 16,896 bytes for a 31 GB image -- index and manifest, no
        # content -- because the image had been produced by the classic
        # builder under the containerd image store. Refused here, at the
        # source, so the message names the export rather than leaving the
        # destination to refuse an image whose fault is on this node.
        omission = _archive_omits_its_own_blobs(archive_path)
        if omission is not None:
            raise ImageBuildError(
                f"exporting {reference!r} produced an archive that is not a whole image: "
                f"{omission}. The image runs on this node but cannot be distributed from "
                f"it; rebuild it with BuildKit, which writes the content an export needs",
                reference=reference,
            )
        return str(image.id)

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
        """Create an inference container with its model, GPU, and API port.

        The model store belongs to the host agent.  A container cannot see it
        unless it is explicitly bind-mounted.  Likewise, DGX Spark exposes its
        accelerator to Docker only when the Docker GPU request is present.

        ``extra_model_paths`` are further model directories the runtime reads --
        a speculative-decoding drafter being the case that forced it. They
        arrive already resolved and checked by the caller; this engine mounts
        what it is given and does not decide what is legitimate to mount.
        """
        import docker

        port = _endpoint_port(endpoint)
        create_args: dict[str, Any] = {
            "name": name,
            "command": command_args,
            "volumes": {model_path: {"bind": model_path, "mode": "ro"}},
            # The declared endpoint selects only the *host-side* port Docker
            # publishes. Which port the runtime listens on inside the container
            # is the adapter's fact, and this line used to hold vLLM's 8000 for
            # every runtime -- so llama.cpp, which listens on 8080, mapped its
            # recorded endpoint to nothing.
            "ports": {f"{_container_api_port(requirements)}/tcp": port},
            "device_requests": [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        }
        # Mounted at the same path inside the container, exactly as the
        # deployment's own model is, so the path the runtime was configured
        # with is the path it finds. Read-only for the same reason: weights are
        # something a runtime reads.
        for extra in extra_model_paths or ():
            if extra != model_path:
                create_args["volumes"][extra] = {"bind": extra, "mode": "ro"}
        if entrypoint is not None:
            create_args["entrypoint"] = entrypoint
        if labels:
            create_args["labels"] = labels
        # Read-write, and the only such mount. The model store above is
        # read-only on purpose; this is where a compiling runtime is permitted
        # to put what it built.
        if cache_path and requirements is not None and requirements.cache_at:
            create_args["volumes"][cache_path] = {
                "bind": requirements.cache_at,
                "mode": "rw",
            }
        # Requirements' own free-form environment (host_config.environment,
        # an operator-declared, unvalidated passthrough) is applied first;
        # the managed credential the route resolved is merged in after, so
        # it wins any key collision rather than losing to it. This used to
        # run the other way around -- environment=... set here, then
        # _apply_requirements's own merge applied host_config.environment on
        # top of it -- which let a free-form environment entry silently
        # override the credential. The schema
        # now refuses that input outright; this is defense in depth for
        # whatever reaches this call without having gone through it.
        _apply_requirements(create_args, requirements, docker)
        if environment:
            merged_env = dict(create_args.get("environment") or {})
            merged_env.update(environment)
            create_args["environment"] = merged_env
        container = self._docker().containers.create(image, **create_args)
        return str(container.id)

    def start_container(self, name: str) -> None:
        self._docker().containers.get(name).start()

    def stop_container(self, name: str) -> None:
        try:
            container = self._docker().containers.get(name)
        except Exception:
            return  # already gone — stopping an absent container is a no-op
        container.stop()

    def remove_container(self, name: str) -> None:
        try:
            container = self._docker().containers.get(name)
        except Exception:
            return
        container.remove(force=True)

    def inspect_container(self, name: str) -> ContainerState | None:
        """Report ``name``'s actual state from the Engine API.

        ``containers.get`` resolves a container in any state — created, exited,
        dead — so its success is not evidence of liveness. Only
        ``attrs["State"]["Running"]`` is. Reading the former as the latter is
        what let a deployment whose runtime had died report itself healthy.
        """
        try:
            container = self._docker().containers.get(name)
        except Exception:
            return None

        return self._state_of(container)

    def list_managed_containers(self) -> list[ContainerState]:
        """Enumerate the ``tensorstead-`` namespace via the Engine API.

        ``all=True`` so a container whose runtime died is still reported: that
        is the case worth surfacing, not the one to filter out.

        Docker's ``name`` filter is an unanchored substring match, so it would
        also return ``my-tensorstead-test``. The prefix is re-checked here rather
        than trusted to the daemon's matching rules.
        """
        try:
            containers = self._docker().containers.list(all=True)
        except Exception:
            # An unreachable daemon is not evidence that nothing is running.
            # Reporting an empty list would let the coordinator conclude the
            # namespace is clean, so say nothing instead of saying "none".
            return []
        return [
            self._state_of(container)
            for container in containers
            if str(getattr(container, "name", "") or "").startswith(_MANAGED_PREFIX)
        ]

    def _state_of(self, container: Any) -> ContainerState:
        """Build a ``ContainerState`` from a docker-py container object."""
        state = (container.attrs or {}).get("State", {})
        # ``status`` is the human-facing word ("exited", "restarting"); the
        # ``Running`` boolean is the authority. A restarting container is
        # deliberately *not* running: it is crash-looping, and reporting it as
        # running is the failure mode this whole change exists to remove.
        running = bool(state.get("Running", False)) and not state.get("Restarting", False)
        exit_code = state.get("ExitCode")
        config = (container.attrs or {}).get("Config", {})

        restart_count = (container.attrs or {}).get("RestartCount")

        return ContainerState(
            running=running,
            name=str(container.name) if getattr(container, "name", None) else None,
            image_digest=self._container_image_digest(container),
            exit_code=int(exit_code) if isinstance(exit_code, int) and not running else None,
            detail=str(container.status) if container.status else None,
            labels={str(k): str(v) for k, v in (config.get("Labels") or {}).items()},
            restart_count=int(restart_count) if isinstance(restart_count, int) else None,
            command=_effective_argv(config),
            environment=_parsed_env(config),
        )

    def image_user(self, reference: str) -> str | None:
        """Read ``Config.User`` from a local image, or None when absent."""
        try:
            image = self._docker().images.get(reference)
        except Exception:
            return None
        user = ((image.attrs or {}).get("Config", {}) or {}).get("User")
        return str(user) if user else None

    def container_processes(self, name: str) -> list[str] | None:
        """The running command lines, from the Engine's own process listing."""
        try:
            container = self._docker().containers.get(name)
            listing = container.top()
        except Exception:
            return None
        titles = list(listing.get("Titles") or [])
        rows = list(listing.get("Processes") or [])
        if not rows:
            return None
        # The command column is last on every platform this runs on, but find
        # it by name where the engine says so rather than trusting position.
        index = titles.index("CMD") if "CMD" in titles else -1
        return [str(row[index]) for row in rows if row]

    def container_logs(self, name: str, *, tail: int = 200) -> str | None:
        """Return the container's last ``tail`` lines, or None if it is absent.

        ``stdout`` and ``stderr`` together, because a runtime that dies during
        startup writes its reason to whichever it happens to use, and asking
        for one is how you get a blank answer to a real failure.

        Fetched and returned; never written down. See the seam's docstring for
        why that is a boundary of this design rather than a convenience.
        """
        try:
            container = self._docker().containers.get(name)
        except Exception:
            return None
        try:
            raw = container.logs(tail=max(1, tail), stdout=True, stderr=True, timestamps=False)
        except Exception as exc:
            # A container can exist with no readable log (a driver that does
            # not support reading, for instance). Saying so beats returning an
            # empty string, which reads as "the runtime said nothing".
            return f"[logs unavailable: {type(exc).__name__}: {exc}]"
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def get_digest(self, name: str) -> str | None:
        try:
            container = self._docker().containers.get(name)
        except Exception:
            return None
        return self._container_image_digest(container)

    @staticmethod
    def _container_image_digest(container: Any) -> str | None:
        """The ``sha256:``-normalised identifier of a container's image."""
        image = container.image
        if image is None:
            return None
        digest = getattr(image, "id", None)
        if not digest:
            return None
        return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def _apply_requirements(
    create_args: dict[str, Any], requirements: ContainerRequirements | None, docker: Any
) -> None:
    """Fold the adapter's declared container needs into the create call.

    Mutates ``create_args`` in place rather than returning a new mapping, so
    the caller's own keys stay in one readable block above and this stays a
    clearly separable concern beneath it.

    Environment here is the *requirement's* environment (``host_config.environment``,
    an operator-declared passthrough), merged onto whatever ``create_args``
    already holds -- which, at the point this runs, is not yet the managed
    credential. This function used to be the last write to
    ``create_args["environment"]``, so a free-form entry silently won over
    the credential a caller had already set.
    The credential is now applied by the caller *after* this returns,
    specifically so it wins that collision instead of losing it -- this
    function no longer carries that guarantee, the caller does.
    """
    if requirements is None:
        return
    if requirements.network_mode:
        # Host networking removes the published port map with it: on the host
        # stack there is nothing to publish, and leaving `ports` set makes
        # Docker reject the create outright rather than quietly ignore it.
        create_args["network_mode"] = requirements.network_mode
        if requirements.network_mode == "host":
            create_args.pop("ports", None)
    if requirements.devices:
        create_args["devices"] = [f"{d}:{d}:rwm" for d in requirements.devices]
    if requirements.capabilities:
        # Named capabilities only. The blanket all-permissions flag would also
        # work and would grant everything besides, leaving a record that cannot
        # say what its container may do.
        create_args["cap_add"] = list(requirements.capabilities)
    if requirements.shm_size:
        create_args["shm_size"] = requirements.shm_size
    if requirements.ipc_mode:
        create_args["ipc_mode"] = requirements.ipc_mode
    if requirements.extra_ports:
        ports = dict(create_args.get("ports") or {})
        # The deployment's own endpoint wins. A runtime requirement must not be
        # able to move the port the product tells operators to connect to.
        for container_port, host_port in requirements.extra_ports.items():
            ports.setdefault(container_port, host_port)
        create_args["ports"] = ports
    if requirements.ulimits:
        create_args["ulimits"] = [
            docker.types.Ulimit(name=name, soft=soft, hard=hard)
            for name, (soft, hard) in requirements.ulimits.items()
        ]
    if requirements.environment:
        merged = dict(create_args.get("environment") or {})
        merged.update(requirements.environment)
        create_args["environment"] = merged


def _effective_argv(config: dict[str, Any]) -> list[str]:
    """Entrypoint and command as one argv, the way the process actually ran.

    Docker keeps them apart; the runtime does not experience them apart. An
    operator comparing "what did we ask for" against "what is running" needs
    the joined form, and joining it here means every caller sees the same
    answer rather than each one reassembling it slightly differently.
    """
    parts: list[str] = []
    for key in ("Entrypoint", "Cmd"):
        value = config.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str) and value:
            parts.append(value)
    return parts


def _parsed_env(config: dict[str, Any]) -> dict[str, str]:
    """``Config.Env`` (``["KEY=value", ...]``) as a dict, split on the first ``=``.

    A value may legitimately contain ``=`` itself; ``str.split("=", 1)``
    keeps everything after the first one intact rather than truncating it.
    """
    result: dict[str, str] = {}
    for entry in config.get("Env") or []:
        if not isinstance(entry, str) or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        result[key] = value
    return result


# The namespace this product owns on a node. Every container it creates is
# named ``tensorstead-<deployment_id>``; nothing else in the namespace was put
# there by the product.
_MANAGED_PREFIX = "tensorstead-"


def _endpoint_port(endpoint: str) -> int:
    """Return a validated TCP port from the declared ``host:port`` endpoint."""
    try:
        parsed = urlsplit(f"//{endpoint}")
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"endpoint must be host:port, got {endpoint!r}") from exc
    if not parsed.hostname or port is None:
        raise ValueError(f"endpoint must be host:port, got {endpoint!r}")
    return port


def _image_removal_failure(exc: Exception) -> str:
    """Classify a docker-py image-removal failure as ``not_present``/``in_use``/``other``.

    Matched on the daemon's status code where docker-py exposes one and on its
    message otherwise, because the exception classes are only importable when
    docker-py is installed and this module is written to degrade without it
    (the same reason every other call here catches ``Exception``).
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404:
        return "not_present"
    if status == 409:
        return "in_use"
    text = str(exc).lower()
    if "no such image" in text or "not found" in text:
        return "not_present"
    if "conflict" in text or "is being used" in text or "is using its referenced image" in text:
        return "in_use"
    return "other"
