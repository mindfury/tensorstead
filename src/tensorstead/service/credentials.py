"""Credential service.

Named per-source credentials, exactly one of which is the default. The service
owns three rules that would otherwise live in an operator's head:

- **The store holds a reference, never a value**. ``set`` hands the
  value to the credential provider and persists only what the provider hands
  back. No endpoint returns a secret because **there is no read path** — the one
  method that yields a value, ``resolve_for_acquisition``, exists for the
  coordinator→agent hop and is never routed.
- **Exactly one default per source**. A source's first credential
  becomes its default, so a source is never left with credentials but no
  default; promoting another demotes the incumbent.
- **The default is never silently reassigned**. Deleting it reports
  the consequence — either the source now has none, or a named successor was
  promoted because the caller asked for it.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from typing import Any

from tensorstead.domain.errors import CredentialResolutionError, NotFoundError, StillReferencedError
from tensorstead.domain.models import Credential, InferenceCredential, ModelSource
from tensorstead.ports.repository import Repository

# secret_from's own operational namespace. A reference form exists so an
# operator can point at a credential for something Tensorstead talks to (a
# gated Hugging Face token, an inference key) without typing the value —
# never at Tensorstead's own management/replication/inference secrets
# (the recorded design: the own-secret boundary). Every env var
# this product reads for
# itself is named with this prefix (agent/app.py, coordinator/app.py); an
# MCP-reachable caller asking to bind ``TENSORSTEAD_MGMT_TOKEN`` as an
# "inference credential" is not a legitimate use of this form under any
# caller, so the restriction applies unconditionally rather than only when
# reached through MCP.
_OWN_ENV_PREFIX = "TENSORSTEAD_"

# Secrets are short. A cap turns a special file (``/dev/zero``, a device
# node, a fifo with no EOF) from an unbounded read into an immediate refusal
# rather than a hang.
_MAX_SECRET_FILE_BYTES = 65536


def secret_from(*, secret: str | None, from_env: str | None, from_file: str | None) -> str:
    """Resolve exactly one accepted secret form into a value.

    Shared by both credential services so neither grows its own copy of the
    "which form was supplied" rules. No form places the value on a command
    line, and the caller never persists what comes back.
    """
    given = [form for form in (secret, from_env, from_file) if form is not None]
    if len(given) != 1:
        raise ValueError("exactly one of secret, from_env, or from_file is required")

    if secret is not None:
        # Taken verbatim — a secret may legitimately contain whitespace.
        return secret
    if from_env is not None:
        if from_env.startswith(_OWN_ENV_PREFIX):
            raise ValueError(
                f"{from_env!r} is a Tensorstead operational variable, not an external "
                "credential -- it cannot be bound as one"
            )
        value = os.environ.get(from_env)
        if value is None:
            raise ValueError(f"environment variable {from_env!r} is not set")
        return value
    # Type narrowing for mypy: from_env was handled above, so this is
    # the remaining branch. Not a check on untrusted input.
    assert from_file is not None  # noqa: S101
    return _read_secret_file(from_file)


def _read_secret_file(path: str) -> str:
    """Read a credential file with the care its contents deserve.

    ``O_NOFOLLOW`` refuses a symlink atomically at open time rather than
    stat-then-open, which a race could still slip through. ``O_NONBLOCK`` is
    equally load-bearing and easy to miss: a FIFO with no writer blocks a
    plain open-for-read forever, which is a caller-triggered hang of the
    entire coordinator process, not merely an unbounded read -- found by
    this fix's own test hanging in exactly that way before this flag was
    added. It has no effect on a genuine regular file, which is the only
    thing this function accepts once past the check below. The regular-file
    check after opening rejects the FIFO (and any device or socket) that
    ``O_NOFOLLOW`` does not cover, and the size cap turns one with no EOF
    (``/dev/zero``) into a refusal instead of an unbounded read.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError(f"could not read credential file {path!r}: {exc}") from exc
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"credential file {path!r} is not a regular file")
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1  # ownership transferred to the file object
            data = handle.read(_MAX_SECRET_FILE_BYTES + 1)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > _MAX_SECRET_FILE_BYTES:
        raise ValueError(
            f"credential file {path!r} exceeds the {_MAX_SECRET_FILE_BYTES}-byte limit"
        )
    # A trailing newline is a file convention, not part of the secret.
    return data.decode("utf-8").rstrip("\n")


class CredentialService:
    """Set, list, and delete named per-source credential references."""

    def __init__(self, repository: Repository, provider: Any) -> None:
        self._repo = repository
        self._provider = provider

    # -------------------------------------------------------------------- set
    def set(
        self,
        source_id: str,
        name: str,
        *,
        secret: str | None = None,
        from_env: str | None = None,
        from_file: str | None = None,
        default: bool = False,
    ) -> Credential:
        """Store a credential value and persist a reference to it.

        Exactly one of ``secret``, ``from_env``, or ``from_file`` is accepted
        All three end in the same place — the provider's
        protected store — so the reference forms are about **who has to handle
        the value**, not about where it lands: the MCP surface offers only the
        reference forms, which is what keeps a secret out of an agent's context.
        """
        value = self._value_from(secret=secret, from_env=from_env, from_file=from_file)
        self._ensure_source(source_id)

        existing = self._repo.get_credential(source_id, name)
        is_default = self._resolve_default_flag(source_id, name, existing, default)
        if is_default:
            self._demote_other_defaults(source_id, name)

        secret_ref = self._provider.store(source_id, name, value)
        credential = Credential(
            source_id=source_id,
            name=name,
            secret_ref=secret_ref,
            is_default=is_default,
            set_at=datetime.now().astimezone(),
        )
        self._repo.save_credential(credential)
        return credential

    def _value_from(
        self, *, secret: str | None, from_env: str | None, from_file: str | None
    ) -> str:
        """Resolve the one accepted form into a value (delegates to the shared helper)."""
        return secret_from(secret=secret, from_env=from_env, from_file=from_file)

    def _resolve_default_flag(
        self,
        source_id: str,
        name: str,
        existing: Credential | None,
        requested: bool,
    ) -> bool:
        """Decide whether this credential is the source's default.

        Asked for, already held, or nobody else holds it — in that order. The
        last clause is what makes a source's first credential its default, so a
        source never ends up holding credentials with none of them applicable to
        an acquisition that names none.
        """
        if requested:
            return True
        if existing is not None and existing.is_default:
            return True
        return not any(
            c.is_default
            for c in self._repo.list_credentials()
            if c.source_id == source_id and c.name != name
        )

    def _demote_other_defaults(self, source_id: str, name: str) -> None:
        for credential in self._repo.list_credentials():
            if credential.source_id != source_id or credential.name == name:
                continue
            if credential.is_default:
                self._repo.save_credential(
                    Credential(
                        source_id=credential.source_id,
                        name=credential.name,
                        secret_ref=credential.secret_ref,
                        is_default=False,
                        set_at=credential.set_at,
                    )
                )

    def _ensure_source(self, source_id: str) -> None:
        """Register the model source on first use, as acquisition does."""
        if self._repo.get_model_source(source_id) is None:
            self._repo.save_model_source(
                ModelSource(
                    id=source_id,
                    supports_revision_pinning=True,
                    requires_credential=True,
                )
            )

    # ------------------------------------------------------------------- list
    def list(self) -> list[Credential]:
        """Names and default status only — never a value."""
        return self._repo.list_credentials()

    # ----------------------------------------------------------------- delete
    def delete(self, source_id: str, name: str, *, promote: str | None = None) -> dict[str, Any]:
        """Delete a credential and report what it did to the source's default.

        Deleting the default leaves the source with none unless ``promote``
        names a successor. Both outcomes are returned to the caller; neither is
        applied silently.
        """
        credential = self._repo.get_credential(source_id, name)
        if credential is None:
            raise NotFoundError(f"no credential named {name!r} for source {source_id!r}")

        was_default = credential.is_default
        self._provider.delete(credential.secret_ref)
        self._repo.delete_credential(source_id, name)

        if promote is not None:
            successor = self._repo.get_credential(source_id, promote)
            if successor is None:
                raise NotFoundError(f"no credential named {promote!r} for source {source_id!r}")
            self._demote_other_defaults(source_id, promote)
            self._repo.save_credential(
                Credential(
                    source_id=successor.source_id,
                    name=successor.name,
                    secret_ref=successor.secret_ref,
                    is_default=True,
                    set_at=successor.set_at,
                )
            )

        default_now = self._default_name(source_id)
        return {
            "source_id": source_id,
            "name": name,
            "was_default": was_default,
            "default_now": default_now,
            "consequence": self._consequence(source_id, was_default, default_now),
        }

    def _consequence(self, source_id: str, was_default: bool, default_now: str | None) -> str:
        if not was_default:
            if default_now is None:
                return f"source {source_id!r} has no default credential"
            return f"the default for source {source_id!r} is unchanged ({default_now})"
        if default_now is None:
            return (
                f"source {source_id!r} now has no default credential; an acquisition "
                f"naming none will proceed without one"
            )
        return f"{default_now} is now the default credential for source {source_id!r}"

    def _default_name(self, source_id: str) -> str | None:
        for credential in self._repo.list_credentials():
            if credential.source_id == source_id and credential.is_default:
                return credential.name
        return None

    # ------------------------------------------------- acquisition-time path
    def resolve_for_acquisition(self, source_id: str, name: str | None = None) -> str | None:
        """Return the secret value for one acquisition, or ``None``.

        **Not a management-surface method.** It is the acquisition-time path
        described in the design: the coordinator resolves the reference, passes
        the value to the agent for that one request, and neither side persists
        it. No route calls this.

        ``None`` means the source has no credential to apply — not an error
        here, because whether one is required is upstream's judgement, surfaced
        as ``authorization_refused`` when it is.
        """
        if name is not None:
            credential = self._repo.get_credential(source_id, name)
            if credential is None:
                raise NotFoundError(f"no credential named {name!r} for source {source_id!r}")
            return str(self._provider.resolve(credential.secret_ref))

        for credential in self._repo.list_credentials():
            if credential.source_id == source_id and credential.is_default:
                return str(self._provider.resolve(credential.secret_ref))
        return None


class InferenceCredentialService:
    """Inference credentials owned by the product.

    The credential authenticating inference clients to a runtime is a
    *deployment setting*, and this product already owns every other setting a
    deployment has. Splitting one runtime's configuration across two owners was
    the inconsistency; an Ansible vault plus a host environment file is also
    exactly the "undocumented host state" this design forbids essential
    deployment knowledge from living in.

    Owning the record is not the same as being in the data path. The
    coordinator never uses this value to serve a request — it resolves it at
    deployment-start time and hands it to the agent, which injects it into the
    runtime and retains nothing. Inference clients still reach the runtime
    directly.

    The value lives in the credential provider, never the database, and there
    is no read path. Storage holds a reference.
    """

    def __init__(self, repository: Repository, provider: Any) -> None:
        self._repo = repository
        self._provider = provider

    def set(
        self,
        name: str,
        *,
        value: str | None = None,
        from_env: str | None = None,
        from_file: str | None = None,
    ) -> dict[str, Any]:
        """Store an inference credential under ``name``."""
        secret = secret_from(secret=value, from_env=from_env, from_file=from_file)
        ref = self._provider.store("inference", name, secret)
        self._repo.save_inference_credential(
            InferenceCredential(name=name, secret_ref=ref, set_at=datetime.now().astimezone())
        )
        return {"name": name, "status": "stored"}

    def list(self) -> list[dict[str, Any]]:
        """Names and set times only. Never a value — there is no read path."""
        return [
            {"name": c.name, "set_at": c.set_at.isoformat()}
            for c in self._repo.list_inference_credentials()
        ]

    def delete(self, name: str) -> dict[str, Any]:
        """Delete a credential, refusing while a deployment still binds it.

        The same referential-integrity rule that applies to models and
        images: never leave a deployment pointing at something that no longer
        exists.
        """
        existing = self._repo.get_inference_credential(name)
        if existing is None:
            raise NotFoundError(f"inference credential {name!r} is not stored")

        referrers = [
            d.name
            for d in self._repo.list_deployments()
            if self._repo.get_bound_inference_credential(d.id) == name
        ]
        if referrers:
            raise StillReferencedError(
                f"inference credential {name!r} is still used by: {', '.join(sorted(referrers))}",
                detail={"referrers": sorted(referrers)},
            )

        self._provider.delete(existing.secret_ref)
        self._repo.delete_inference_credential(name)
        return {"name": name, "status": "deleted"}

    def bind(self, deployment_id: str, name: str | None) -> dict[str, Any]:
        """Bind a deployment to a credential, or clear the binding.

        Clearing does not disable authentication: it returns the deployment to
        whatever the node was provisioned with, which is the pre-existing
        behaviour and the migration path for deployments predating this.
        """
        if name is None:
            self._repo.unbind_inference_credential(deployment_id)
            return {"deployment_id": deployment_id, "inference_credential": None}
        if self._repo.get_inference_credential(name) is None:
            raise NotFoundError(f"inference credential {name!r} is not stored")
        self._repo.bind_inference_credential(deployment_id, name)
        return {"deployment_id": deployment_id, "inference_credential": name}

    def resolve_for_deployment(self, deployment_id: str) -> str | None:
        """Resolve the bound credential's value for one start, or None.

        Called immediately before the agent call and never retained. Returning
        ``None`` means "this node's own provisioning applies", not
        "unauthenticated" -- and means that only when nothing is bound at
        all. A binding that exists but cannot be resolved -- its record
        vanished, or the provider could not produce a value -- raises
        ``CredentialResolutionError`` instead of returning ``None``, because
        those two situations are not the same fact and must not collapse
        into the same "no binding" answer a caller then treats as safe to
        fall back from -- a recorded finding showed that collapse in the field.
        """
        name = self._repo.get_bound_inference_credential(deployment_id)
        if name is None:
            return None
        record = self._repo.get_inference_credential(name)
        if record is None:
            raise CredentialResolutionError(
                f"deployment {deployment_id!r} is bound to credential {name!r}, "
                "which no longer exists",
                detail={"deployment_id": deployment_id, "credential_name": name},
            )
        try:
            resolved = self._provider.resolve(record.secret_ref)
        except Exception as exc:
            raise CredentialResolutionError(
                f"deployment {deployment_id!r}'s bound credential {name!r} "
                f"could not be resolved: {exc}",
                detail={"deployment_id": deployment_id, "credential_name": name},
            ) from exc
        if not resolved:
            raise CredentialResolutionError(
                f"deployment {deployment_id!r}'s bound credential {name!r} "
                "resolved to an empty value",
                detail={"deployment_id": deployment_id, "credential_name": name},
            )
        return str(resolved)
