"""One validation-error response for every Tensorstead HTTP surface.

FastAPI's default 422 echoes each rejected value back in ``detail[].input``
(and sometimes ``ctx``), so a malformed credential request returned the secret
it carried (CodeGuard CG-12). This keeps what an operator needs -- where, what
rule, which kind -- and drops every input value, for every field rather than a
list of sensitive ones that would go stale. It is also the structured
``{code, message}`` shape the CLI renders; the default body was reported as
"not this product".
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def safe_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    """Each error's location, message and type -- never its input or context."""
    safe: list[dict[str, Any]] = []
    for error in exc.errors():
        loc = [str(part) for part in error.get("loc", ())]
        kind = str(error.get("type", ""))
        if kind == "extra_forbidden" and loc:
            # The offending key is client-chosen text, not a declared field:
            # a secret pasted where a field name goes would otherwise come
            # straight back. The depth and the rule still say what is wrong.
            loc[-1] = "<unexpected field>"
        safe.append({"loc": loc, "msg": str(error.get("msg", "")), "type": kind})
    return safe


async def handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    errors = safe_errors(exc) if isinstance(exc, RequestValidationError) else []
    return JSONResponse(
        status_code=422,
        content={
            "code": "invalid_request",
            "message": "the request does not match this operation's schema",
            "detail": {"errors": errors},
        },
    )
