"""Credential-provider port.

The seam between the coordinator and wherever a secret value actually lives.
The provider returns a **reference**, never a value: the
database and the management surface hold only a pointer into the provider's
protected store, which is what makes the absence of a secret in any output a
structural property rather than a discipline.

The v1 implementation is a permission-restricted local file
(``adapters/credentials/local_file.py``); the port exists so alternate
providers can be added later without touching the domain or service layer.
"""

from __future__ import annotations

from typing import Protocol


class CredentialProvider(Protocol):
    """Store and resolve secret *values* outside the database.

    Implementations own the secret bytes and their permission-restricted
    storage. They must never write a secret to a log, an operation record, or
    an export, and must never return a value across the management surface
    (there is intentionally no read path that yields one).
    """

    def store(self, source_id: str, name: str, value: str) -> str:
        """Persist ``value`` and return a reference to it.

        The reference is what the database persists; ``value`` itself is never
        stored in a place the management surface can read back.
        """

    def resolve(self, secret_ref: str) -> str:
        """Return the secret value behind ``secret_ref`` for use at request time.

        Only ever called to supply a credential to a node agent during
        acquisition; the value crosses the coordinator→agent hop in-memory and
        is never persisted there.
        """

    def delete(self, secret_ref: str) -> None:
        """Delete the stored value behind ``secret_ref``."""
