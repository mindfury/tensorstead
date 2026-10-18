"""Admission budgets for every Tensorstead HTTP surface (CodeGuard CG-15).

No request-body limit, nesting limit or bound on queued background work was
configured, so repeated or oversized requests could consume the management
plane. Three bounds, each refusing explicitly rather than degrading:

- ``BodyLimitMiddleware``: a request body over ``TENSORSTEAD_MAX_REQUEST_BYTES``
  (default 16 MiB) is refused with 413 -- on its declared length, or as soon as
  a chunked body passes the limit -- and JSON nested deeper than
  ``MAX_JSON_DEPTH`` with 400, before any route parses it (the stdlib parser
  would otherwise hit RecursionError and surface a 500).
- ``WorkAdmission``: background operations (acquire, build, image pull) take a
  slot before they are accepted; with none free the request is refused with
  429 ``busy`` instead of joining an unbounded executor queue.
- connection concurrency, set where the servers are started (uvicorn
  ``limit_concurrency``), refused with 503.

Per-client rate limiting is not done here: the management plane is
single-tenant behind one token, and a front proxy is the place for it (see
docs/operations.md).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from tensorstead.domain.errors import DomainError

DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 64
DEFAULT_MAX_BACKGROUND_OPERATIONS = 8
DEFAULT_CONNECTION_LIMIT = 64


def max_request_bytes() -> int:
    return int(os.environ.get("TENSORSTEAD_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES))


def connection_limit() -> int:
    return int(os.environ.get("TENSORSTEAD_MAX_CONNECTIONS", DEFAULT_CONNECTION_LIMIT))


def json_depth_exceeds(body: bytes, limit: int = MAX_JSON_DEPTH) -> bool:
    """Whether ``body``'s bracket nesting passes ``limit``; one linear scan, strings skipped."""
    depth = 0
    in_string = escaped = False
    for byte in body:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            depth += 1
            if depth > limit:
                return True
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1
    return False


async def _refuse(send: Any, status: int, code: str, message: str) -> None:
    body = json.dumps({"code": code, "message": message, "detail": {}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    """Refuse oversized or over-nested request bodies before the app sees them."""

    def __init__(self, app: Any, max_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_bytes if self.max_bytes is not None else max_request_bytes()
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await _refuse(send, 413, "request_too_large", f"request body exceeds {limit} bytes")
            return

        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                await _refuse(send, 413, "request_too_large", f"request body exceeds {limit} bytes")
                return
            chunks.append(chunk)
            more = message.get("more_body", False)
        body = b"".join(chunks)

        if b"json" in headers.get(b"content-type", b"") and json_depth_exceeds(body):
            await _refuse(
                send, 400, "invalid_request", f"JSON nested deeper than {MAX_JSON_DEPTH} levels"
            )
            return

        delivered = False

        async def replay() -> Any:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class AdmissionRefusedError(DomainError):
    """Too much background work is already running; try again shortly."""

    code = "busy"


class WorkAdmission:
    """A fixed number of background-operation slots, taken without waiting."""

    def __init__(self, slots: int | None = None) -> None:
        count = slots or int(
            os.environ.get(
                "TENSORSTEAD_MAX_BACKGROUND_OPERATIONS", DEFAULT_MAX_BACKGROUND_OPERATIONS
            )
        )
        self._slots = threading.BoundedSemaphore(count)
        self.capacity = count

    def admit(self, what: str) -> None:
        if not self._slots.acquire(blocking=False):
            raise AdmissionRefusedError(
                f"{self.capacity} background operations are already running; {what} was not "
                f"accepted -- retry when one finishes",
                detail={"capacity": self.capacity},
            )

    def release(self) -> None:
        self._slots.release()
