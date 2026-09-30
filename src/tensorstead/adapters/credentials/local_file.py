"""Local-file credential provider.

The v1 implementation of the credential-provider port: secret **values** live in
a permission-restricted directory on the coordinator host, and the database
holds only a reference into it.

That split is what makes "no secret in any output" structural rather
than a discipline. There is no column, no response model, and no export field
that a value could leak from, because the value never enters the database in the
first place — the only thing that does is an opaque pointer.

Permissions are set explicitly rather than left to the umask: the store
directory is ``0o700`` and each value file is created ``0o600`` **at open time**,
so there is no window in which a freshly written secret is world-readable.
"""

from __future__ import annotations

import base64
import contextlib
import os
from pathlib import Path

from tensorstead.domain.errors import NotFoundError

_SCHEME = "local-file"

# Owner-only, set explicitly so the process umask cannot widen them.
_DIR_MODE = 0o700
_FILE_MODE = 0o600

# Where the store lives by default. This belongs to the adapter, not to the
# coordinator: the coordinator holds no knowledge of any local filesystem
# layout, which is what keeps it free of a colocation assumption
# (asserted by tests/contract/test_no_colocation_assumption.py).
_DEFAULT_ROOT_ENV = "TENSORSTEAD_CREDENTIALS_DIR"
_DEFAULT_ROOT = "/var/lib/tensorstead/credentials"


def default_root() -> Path:
    """The store location, overridable by ``TENSORSTEAD_CREDENTIALS_DIR``."""
    return Path(os.environ.get(_DEFAULT_ROOT_ENV, _DEFAULT_ROOT))


class LocalFileCredentialProvider:
    """Store secret values as owner-only files under ``root``.

    ``root`` defaults to :func:`default_root` and is created on first write, not
    at construction: building a coordinator must not require write access to the
    credential store just to inspect the app, and a deployment that never sets a
    credential never creates the directory.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else default_root()

    # ------------------------------------------------------------------ port
    def store(self, source_id: str, name: str, value: str) -> str:
        """Persist ``value`` under an owner-only file and return its reference.

        The reference is what the database persists. It encodes the
        ``source_id``/``name`` pair so an operator can see which file belongs to
        which credential — the pointer is not itself sensitive; the file
        permissions are what protect the value.
        """
        self._ensure_root()
        token = _token(source_id, name)
        path = self._root / token
        # O_CREAT with an explicit mode, then an explicit chmod for the case
        # where the file already existed with wider permissions.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        try:
            os.write(fd, value.encode())
        finally:
            os.close(fd)
        os.chmod(path, _FILE_MODE)
        return f"{_SCHEME}:{token}"

    def resolve(self, secret_ref: str) -> str:
        """Return the value behind ``secret_ref`` for one acquisition.

        Called only on the acquisition path: the value crosses the
        coordinator→agent hop in memory and is never persisted on either side.
        No management route reaches this method.
        """
        path = self._path_for(secret_ref)
        try:
            return path.read_text()
        except OSError as exc:
            raise NotFoundError(f"no stored value for credential reference {secret_ref!r}") from exc

    def delete(self, secret_ref: str) -> None:
        """Delete the stored value; absent is not an error (deletion is the goal)."""
        path = self._path_for(secret_ref)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    # -------------------------------------------------------------- internals
    def _ensure_root(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        os.chmod(self._root, _DIR_MODE)

    def _path_for(self, secret_ref: str) -> Path:
        scheme, _, token = secret_ref.partition(":")
        if scheme != _SCHEME or not token:
            raise NotFoundError(f"not a local-file credential reference: {secret_ref!r}")
        # Reject a token that would escape the store directory.
        if "/" in token or "\\" in token or token in (".", ".."):
            raise NotFoundError(f"not a local-file credential reference: {secret_ref!r}")
        return self._root / token


def _token(source_id: str, name: str) -> str:
    """Encode ``source_id``/``name`` into one filesystem-safe filename.

    A source id or credential name may contain characters that are not safe as a
    single path segment. Base64-url encoding is reversible, so the store stays
    legible to an operator inspecting it directly.
    """
    return base64.urlsafe_b64encode(f"{source_id}/{name}".encode()).decode().rstrip("=")
