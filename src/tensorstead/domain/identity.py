"""Durable identity for managed objects.

A ULID is chosen so identities are sortable, time-ordered, and collision-free
without a coordinating sequence, and are stable across node and coordinator
restarts. This module depends on no platform-specific package;
the ULID is produced from a pure Python implementation so the domain package
stays import-clean of anything OS- or vendor-specific.
"""

from __future__ import annotations

import os
import time

# 26-char Crockford base32 alphabet for the time+counter portion. The standard
# ULID alphabet omits I, L, O, U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ALPHABET_REVERSE = {c: i for i, c in enumerate(_ALPHABET)}
_ULID_LEN = 26


def new_ulid() -> str:
    """Return a new ULID string.

    The first 10 characters encode milliseconds since the Unix epoch; the
    remaining 16 encode a random 80-bit value. This yields the standard 128-bit
    ULID layout, monotonic within a timestamp is *not* guaranteed (a random
    suffix is used rather than a counter), which is sufficient for durable
    identity — sorting is a bonus of the time prefix, not a contract.
    """
    ms = int(time.time() * 1000)
    random_bytes = os.urandom(10)
    payload = (ms << 80) | int.from_bytes(random_bytes, "big")
    return _encode(payload)


def _encode(value: int) -> str:
    chars = []
    for _ in range(_ULID_LEN):
        value, rem = divmod(value, 32)
        chars.append(_ALPHABET[rem])
    return "".join(reversed(chars))


def is_valid_ulid(candidate: str) -> bool:
    """Return whether ``candidate`` is a syntactically valid ULID."""
    if len(candidate) != _ULID_LEN:
        return False
    return all(ch in _ALPHABET_REVERSE for ch in candidate)


def assert_valid_ulid(candidate: str, *, what: str = "id") -> None:
    """Raise ``ValueError`` unless ``candidate`` is a valid ULID."""
    if not is_valid_ulid(candidate):
        raise ValueError(f"invalid {what}: {candidate!r} is not a ULID")
