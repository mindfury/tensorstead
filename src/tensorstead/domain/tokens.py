"""Compare bearer credentials without ever raising on what a client sent.

``hmac.compare_digest`` on two ``str`` values raises ``TypeError`` when either
holds a non-ASCII character, so a header such as ``Bearer \\xff`` produced an
uncaught exception and a traceback instead of the ordinary 401 (CodeGuard
CG-19). Comparing bytes never raises. HTTP header values arrive as latin-1, so
encoding a received ``str`` back to latin-1 recovers exactly the bytes sent.
"""

from __future__ import annotations

import hmac


def _wire_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    # Characters above U+00FF cannot have come off the wire as a header value;
    # replacing them keeps the comparison total -- and still a mismatch.
    return value.encode("latin-1", errors="replace")


def tokens_match(supplied: str | bytes | None, expected: str) -> bool:
    """Constant-time equality of ``supplied`` and ``expected``; never raises."""
    if supplied is None:
        return False
    return hmac.compare_digest(_wire_bytes(supplied), expected.encode("utf-8"))
