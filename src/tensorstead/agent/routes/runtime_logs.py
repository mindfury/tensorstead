"""Agent runtime-observability route — ``GET .../{id}/runtime``.

An earlier change made the product able to say *that* a deployment stopped serving. It
still could not say **why**. On 2026-08-10 the answer existed the whole time, in
the runtime's own output on the node, reachable only by an operator with SSH —
which is exactly the position the management plane exists to make unnecessary.

This returns three things, none of which the product computes:

- the argv the runtime was actually launched with, read back from the container;
- how many times the engine has restarted it;
- the last N lines it wrote.

**Nothing here is stored.** The agent reads through to the container engine and
returns; no line of runtime output is persisted, recorded against a deployment,
or included in an export. That is the design's boundary and it is
structural rather than a matter of care: a runtime may log whatever it likes,
request content included, and the management plane must not become the place
that content comes to rest. Read on demand, held nowhere.

Strictly read-only. Reading a log mutates nothing.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict

from tensorstead.agent import cache_store
from tensorstead.agent.app import require_management
from tensorstead.agent.container_engine.base import ContainerState

router = APIRouter(prefix="/agent/v1", tags=["observation"])

# Sized from measurement rather than from a guess about how much a runtime
# writes. 200 was the guess -- "enough for a traceback and the banner above it"
# -- and against a real vLLM on 2026-08-11 it returned **zero** useful lines:
# 84% of that runtime's output was uvicorn access logging for a Prometheus
# scrape of /metrics, at a rate that buries anything else within seconds.
#
# 500 leaves roughly eighty real lines through the same noise. It is not a fix
# for a chatty runtime -- turning the access log off is, and the
# passthrough makes that reachable -- but a default that shows nothing on a
# healthy production deployment is a default that fails exactly when it is
# first tried.
_DEFAULT_TAIL = 500
_MAX_TAIL = 2000


class RuntimeReport(BaseModel):
    """What the runtime itself has to say."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    # ``None`` where the container does not exist; an empty list would claim
    # the runtime was launched with no arguments at all.
    argv: list[str] | None = None
    restart_count: int | None = None
    exit_code: int | None = None
    running: bool | None = None
    # The runtime's own output, verbatim and unparsed. The product does not
    # interpret it: a log line is evidence for a human, and a parser here would
    # be this product asserting it understands a runtime it does not ship.
    log_tail: str | None = None
    lines_requested: int = _DEFAULT_TAIL
    # What the durable compile cache actually holds. Whether a
    # runtime *used* the cache it was given is a different fact from whether one
    # was provisioned, and only the first answers "why is this still
    # recompiling". ``None`` from an agent too old to look.
    cache: dict[str, Any] | None = None
    # What is *actually* running, as distinct from ``argv``, which is what the
    # container was configured with.
    running_processes: list[str] | None = None
    # Whether the arguments the product passed survived into the running
    # process. ``None`` where it could not be established -- no process listing,
    # or nothing distinctive to look for -- which must not read as either
    # answer.
    argv_honoured: bool | None = None
    detail: str | None = None


@router.get(
    "/deployments/{deployment_id}/runtime",
    response_model=RuntimeReport,
    dependencies=[Depends(require_management)],
)
def get_runtime(
    deployment_id: str,
    request: Request,
    tail: int = Query(default=_DEFAULT_TAIL, ge=1, le=_MAX_TAIL),
) -> RuntimeReport:
    """Read the runtime's argv, restart count, and recent output. Mutates nothing."""
    engine = getattr(request.app.state, "container_engine", None)
    container_name = f"tensorstead-{deployment_id}"
    now = datetime.now().astimezone()

    if engine is None:
        return RuntimeReport(
            observed_at=now,
            lines_requested=tail,
            detail="this agent has no container engine configured",
        )

    container: ContainerState | None = None
    inspect = getattr(engine, "inspect_container", None)
    if inspect is not None:
        container = inspect(container_name)

    if container is None:
        return RuntimeReport(
            observed_at=now,
            lines_requested=tail,
            detail="no container exists for this deployment",
        )

    processes = _processes(engine, container_name)
    return RuntimeReport(
        observed_at=now,
        argv=list(container.command),
        restart_count=container.restart_count,
        exit_code=container.exit_code,
        running=container.running,
        log_tail=_logs(engine, container_name, tail),
        lines_requested=tail,
        cache=cache_store.inspect(deployment_id),
        running_processes=processes,
        argv_honoured=_argv_honoured(container.command, processes),
        detail=None,
    )


def _processes(engine: Any, container_name: str) -> list[str] | None:
    """The running command lines, tolerating an engine that cannot list them."""
    read = getattr(engine, "container_processes", None)
    if read is None:
        return None
    try:
        result: list[str] | None = read(container_name)
    except Exception:
        return None
    return result


def _argv_honoured(configured: list[str], processes: list[str] | None) -> bool | None:
    """Did the arguments the product passed survive into the running process?

    The rule is that an entrypoint may prepare and may not decide. One that rebuilds its
    own command line from environment variables leaves the deployment's declared
    ``runtime_config`` describing nothing, and the *configured* argv still looks
    perfectly correct -- which is why this compares against what is running.

    Deliberately a weak test, and honest about it. It asks whether **any**
    distinctive configured argument appears in any running command line. A
    runtime may legitimately reorder, expand or absorb arguments, so requiring
    all of them would cry wolf; requiring none of them to be missing would miss
    the case entirely. "None of what we passed is anywhere" is the signal that
    survives both.

    ``None`` when it cannot be established: no process listing, or nothing
    distinctive enough to look for. An unestablished fact rendered as either
    answer is the defect this whole codebase keeps finding.
    """
    if not processes:
        return None

    # Only the *values* the product chose, meaning what follows a ``--flag``.
    #
    # A first draft compared every non-flag token, which matched on ``vllm`` and
    # ``serve`` -- those come from the entrypoint and appear whatever arguments
    # are used, so a runtime ignoring every one of them still looked honoured.
    # My own test caught it, which is the argument for writing the failing case
    # rather than only the passing one.
    #
    # Short values are dropped too: a bare ``2`` matches almost any command
    # line by accident, and an accidental match here reads as a clean bill of
    # health.
    chosen = [
        value
        for flag, value in itertools.pairwise(configured)
        if flag.startswith("--") and not value.startswith("-") and len(value) > 3
    ]
    if not chosen:
        return None
    joined = " ".join(processes)
    return any(value in joined for value in chosen)


def _logs(engine: Any, container_name: str, tail: int) -> str | None:
    """Read the log tail, tolerating an engine that cannot provide one.

    ``None`` means the product could not obtain the runtime's output — an
    engine too old to expose it, or a read that failed. It is deliberately not
    an empty string, which would say the runtime ran silently. The whole value
    of this endpoint is in a failing runtime's last words; reporting silence it
    did not utter would send an operator looking in the wrong place.
    """
    read = getattr(engine, "container_logs", None)
    if read is None:
        return None
    try:
        result: str | None = read(container_name, tail=tail)
    except Exception as exc:
        return f"[logs unavailable: {type(exc).__name__}: {exc}]"
    return result
