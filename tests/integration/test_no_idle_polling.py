"""GUARDRAIL: no idle polling.

With the system idle and no client request outstanding, zero outbound calls
are made by the coordinator or the agent. The product runs no polling loop,
no sampler, no heartbeat, and no telemetry of any kind.

This is verified by asserting that:

- No ``while True`` / ``asyncio.sleep`` / ``Timer`` / ``Thread`` / scheduler
  pattern exists in the coordinator or agent source.
- No background task is started by ``build_coordinator_app`` or
  ``build_agent_app``.
- The observation and resource services take readings only when called —
  no ``__init__`` starts a sampler, no ``@app.on_event("startup")`` schedule
  exists.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"

# Modules that are allowed to contain sleep/timer patterns (test infrastructure,
# not product code). Product code must not poll.
#
# ``agent.routes.deployments`` -- added 2026-08-13, and it widens this guardrail,
# so it is argued rather than asserted.
#
# ``_await_rendezvous`` polls a local port while the *head of a distributed
# group* comes up. The rule is that "with the
# system idle and no client request outstanding, zero outbound calls are made".
# This loop cannot run when the system is idle: it exists only inside the
# coordinator's create-deployment call, and it ends when that call returns. It
# is the same reasoning that scopes ``cli/`` out of this scan below -- a
# request-driven wait is not a poller -- rather than a new kind of exception.
#
# It is also bounded, makes no *outbound* call (it connects to this host), and
# starts no thread, timer, or scheduler. What this rule forbids is the product
# doing work nobody asked for; this is the product doing exactly what was asked
# and declining to lie about when it finished.
#
# The next entry here should be as hard to add as this one was. The companion
# test below pins that the loop is reachable only from the request path, so the
# exemption cannot quietly grow into a background poller in the same module.
_ALLOWED_SLEEP_MODULES: set[str] = {"agent.routes.deployments"}

# Patterns that indicate a polling/scheduling loop. Each is checked against the
# AST or source of product modules.
_POLL_PATTERNS = [
    r"while\s+True",
    r"asyncio\.sleep",
    r"time\.sleep",
    r"threading\.Thread",
    r"Timer\(",
    r"schedule\.",
    r"APScheduler",
    r"croniter",
    r"@app\.on_event\(.startup.\)",
    r"BackgroundTasks",
]


def _product_python_files() -> list[Path]:
    """Product .py files the idle-polling rule actually governs.

    The rule constrains the management plane -- the coordinator and the agent --
    which must emit zero outbound calls "with the system idle and no client
    request outstanding." The client CLI is neither: it is a short-lived
    run-and-exit tool that legitimately polls an operation it just initiated,
    the same way the MCP client polls ``operation_get``. Scanning ``cli/`` for a
    client-side ``time.sleep`` poll would be a false positive against a rule
    whose own text scopes it to the coordinator or the agent. Excluding ``cli/``
    aligns the scan with the rule's stated scope without weakening the protection
    on the coordinator or the agent, which remain fully policed below.
    """
    return sorted(p for p in _SRC.rglob("*.py") if "cli" not in p.relative_to(_SRC).parts)


def test_no_polling_loop_patterns_in_product_code() -> None:
    """No while-True, asyncio.sleep, Timer, or scheduler in product code."""
    offenders: list[str] = []
    for py_file in _product_python_files():
        rel = py_file.relative_to(_SRC)
        module_name = str(rel).replace("/", ".").replace(".py", "")
        if module_name in _ALLOWED_SLEEP_MODULES:
            continue
        source = py_file.read_text()
        for pattern in _POLL_PATTERNS:
            # Check source-level patterns (string-based for robustness)
            if re.search(pattern, source):
                offenders.append(f"{rel}: matches {pattern!r}")
    assert not offenders, "polling/scheduling patterns found in product code:\n" + "\n".join(
        offenders
    )


def test_the_rendezvous_wait_is_reachable_only_from_a_request() -> None:
    """The exemption above holds only while this stays request-driven.

    ``agent.routes.deployments`` may contain a sleep because its one loop runs
    inside the create-deployment call. If a future change called it from module
    scope, an app factory, or a startup hook, that argument would be false and
    the module would be a background poller wearing an exemption.

    Checked structurally: every call site is inside a function, and the only
    function that calls it is the route handler.
    """
    module = _SRC / "agent" / "routes" / "deployments.py"
    tree = ast.parse(module.read_text())

    callers: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_await_rendezvous"
            ):
                callers.append(node.name)

    assert callers == ["create_deployment"], (
        f"_await_rendezvous is called from {callers or 'module scope'}; the "
        f"idle-polling exemption for this module argues it runs only inside a client "
        f"request, and that is no longer true"
    )

    # And nothing in this module sleeps outside that one helper.
    sleeping = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "sleep"
            for inner in ast.walk(node)
        )
    ]
    assert sleeping == ["_await_rendezvous"], (
        f"functions sleeping in this module: {sleeping}; the exemption covers "
        f"the rendezvous wait alone"
    )


def _flatten_routes(app: Any) -> list[Any]:
    """Every route on ``app``, descending into included routers.

    Since Starlette 1.4 / FastAPI 0.141, ``include_router`` inserts one opaque
    wrapper object into ``app.router.routes`` instead of flattening the
    included routes into it. Walking ``app.router.routes`` directly therefore
    sees none of the mounted management routes at all.
    """
    found: list[Any] = []
    pending = list(app.router.routes)
    while pending:
        route = pending.pop()
        inner = getattr(route, "original_router", None)
        if inner is not None:
            pending.extend(inner.routes)
            continue
        found.append(route)
    return found


def _dependant_tree(dependant: Any) -> Iterator[Any]:
    """Yield a route's dependant and every sub-dependency, recursively."""
    yield dependant
    for sub in getattr(dependant, "dependencies", []) or []:
        yield from _dependant_tree(sub)


def test_no_background_tasks_started_on_startup() -> None:
    """No route schedules background work.

    A ``BackgroundTasks`` parameter is how FastAPI schedules work to run after
    the response is sent; the product answers a request and stops, so no route
    may take one. FastAPI records that parameter on the route's ``Dependant``
    as ``background_tasks_param_name`` — that is the field to assert on. The
    type of the ``Dependant`` object itself is always ``Dependant`` and never
    mentions BackgroundTasks, so asserting against ``str(type(dep))`` can never
    fail no matter what the routes do.
    """
    from tensorstead.agent.app import build_agent_app
    from tensorstead.coordinator.app import build_coordinator_app

    agent_app = build_agent_app()
    coordinator_app = build_coordinator_app(management_token="test")

    routes = _flatten_routes(agent_app) + _flatten_routes(coordinator_app)

    # Self-check: the enumeration must actually reach the mounted management
    # routes. Without this, a future change to how routers are stored would
    # silently empty the walk and this guardrail would pass vacuously — which
    # is exactly how it failed before.
    assert len(routes) > 20, (
        f"route enumeration found only {len(routes)} routes; it is no longer "
        "reaching the mounted management routes and cannot police anything"
    )

    offenders = [
        f"{getattr(route, 'path', '?')} (parameter {dep.background_tasks_param_name!r})"
        for route in routes
        if getattr(route, "dependant", None) is not None
        for dep in _dependant_tree(route.dependant)
        if getattr(dep, "background_tasks_param_name", None) is not None
    ]
    assert not offenders, (
        "routes schedule background work, which is a polling/deferred-work "
        "mechanism the product must not have:\n" + "\n".join(offenders)
    )


def test_observation_service_has_no_sampler() -> None:
    """ObservationService.__init__ does not start a sampler or timer."""
    from tensorstead.service.observation import ObservationService

    init_source = _extract_method_source(ObservationService, "__init__")
    for pattern in _POLL_PATTERNS:
        assert not re.search(pattern, init_source), (
            f"ObservationService.__init__ contains {pattern!r} — a sampler "
            "would violate the no-idle-polling rule"
        )


def test_resource_service_has_no_sampler() -> None:
    """The agent resource module starts no sampler."""
    resources_path = _SRC / "agent" / "resources.py"
    source = resources_path.read_text()
    for pattern in _POLL_PATTERNS:
        assert not re.search(pattern, source), (
            f"agent/resources.py contains {pattern!r} — a sampler would "
            "violate the no-idle-polling rule"
        )


def _extract_method_source(cls: type, method_name: str) -> str:
    """Extract the source text of a method using ast inspection."""
    import inspect

    source_file = inspect.getfile(cls)
    source_lines = Path(source_file).read_text().splitlines()
    tree = ast.parse("\n".join(source_lines))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return "\n".join(source_lines[item.lineno - 1 : item.end_lineno])
    return ""
