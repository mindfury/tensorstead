"""Every field a coordinator route reads is reachable from the MCP surface.

`test_surface_parity` already asserts that every *operation* exists on the API,
the CLI and MCP. It does not look at *parameters*, and that gap has now been
found by an operator twice.

The concrete case: `entrypoint` was added to build specs
across the domain model, a migration, the repository, the service and the HTTP
route — and not the MCP tool. `buildspec.set` existed on all three surfaces, so
operation parity was satisfied and the suite stayed green, while the only
surface an agent can reach could not express the field at all. A capability that
is unreachable from the surface an operator actually uses is not a capability
they have.

## How this checks it

Five coordinator routes take an untyped ``payload: dict[str, Any]`` and read it
with ``payload.get("key")``. That is the population at risk: a typed request
model would at least appear in OpenAPI, but an untyped dict declares nothing, so
nothing but this compares the two sides.

So: extract the keys each such route reads, extract the JSON keys the matching
MCP tool sends, and diff. Source scanning is brittle in general and idiomatic
here — every guardrail in this directory reads source, because the alternative
is trusting that two files were changed together.

**The better fix is to type those five routes**, at which point this check
becomes a comparison of declared shapes rather than a scan for string literals.
That is recorded as follow-up design work rather than done here, because typing a route
changes its published contract and deserves its own change.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_ROUTES = Path(__file__).resolve().parents[2] / "src" / "tensorstead" / "coordinator" / "routes"

# Keys a route reads that no MCP caller should ever supply, with the reason.
# Each is something the product resolves for itself; an MCP parameter for one
# would be an invitation to set it.
_NOT_FOR_CALLERS: dict[str, str] = {
    # The most deliberate omission in the product:
    # the MCP tool accepts only the *reference* forms (`from_env`, `from_file`)
    # so an agent can store a credential without the secret ever entering its
    # context. Adding a `value` parameter here would undo the entire reason the
    # reference forms exist.
    "value": "MCP takes references only, so no secret enters an agent's context",
}


def _route_payload_keys() -> dict[tuple[str, str], set[str]]:
    """Every ``payload.get("key")`` a route reads, by (method, path)."""
    source = (_ROUTES / "__init__.py").read_text()
    found: dict[tuple[str, str], set[str]] = {}

    # Split on decorators so each function body is attributed to its own route.
    blocks = re.split(r"@router\.(get|put|post|delete|patch)\(", source)
    for method, block in zip(blocks[1::2], blocks[2::2], strict=False):
        path_match = re.search(r'"([^"]+)"', block)
        if path_match is None:
            continue
        path = path_match.group(1)
        body = block.split("@router.")[0]
        keys = set(re.findall(r'payload\.get\(\s*"([^"]+)"', body))
        if keys:
            found[(method.upper(), path)] = keys
    return found


def _mcp_sent_keys() -> dict[tuple[str, str], set[str]]:
    """Every JSON key an MCP tool sends, by (method, path).

    Scanned per *tool function*, not per ``client.request`` call. Several tools
    build a ``payload`` dict first and pass ``json=payload``, so a scan that
    only looked after the call reported those keys as unreachable when they are
    sent perfectly well -- four false positives on ``images:build`` the first
    time this ran. A guardrail that cries wolf gets ignored, which costs more
    than the rule protects.
    """
    from tensorstead.mcp import tools

    source = inspect.getsource(tools)
    found: dict[tuple[str, str], set[str]] = {}

    for block in source.split("@server.tool")[1:]:
        call = re.search(r'client\.request\(\s*"(\w+)",\s*f?"([^"]+)"', block)
        if call is None:
            continue
        method, path = call.group(1), call.group(2)
        # Both literal shapes: ``{"key": value}`` and ``payload["key"] = ...``.
        keys = set(re.findall(r'"([a-z_]+)"\s*:', block))
        keys |= set(re.findall(r'payload\[\s*"([a-z_]+)"\s*\]', block))
        found.setdefault((method.upper(), _normalise(path)), set()).update(keys)
    return found


def _normalise(path: str) -> str:
    """One spelling for a route, whichever side it came from.

    Two differences to reconcile, and missing the second is what made the first
    draft of this file pass while checking nothing: parameter names vary
    (``{name}`` vs an f-string's ``{name}``), and **route decorators omit the
    router's ``/v1`` prefix** while MCP tools include it. Without stripping the
    prefix no key ever matched, every route fell through the ``sent is None``
    branch, and the guardrail reported success having compared nothing.
    """
    path = re.sub(r"\{[^}]+\}", "{}", path).rstrip("/")
    return path[len("/v1") :] if path.startswith("/v1/") else path


def test_every_payload_field_a_route_reads_is_reachable_from_mcp() -> None:
    """A field the API accepts and MCP cannot send is unreachable to an agent."""
    mcp = _mcp_sent_keys()
    unreachable: list[str] = []

    for (method, path), keys in _route_payload_keys().items():
        sent = mcp.get((method, _normalise(path)))
        if sent is None:
            continue  # no MCP tool for this route; surface parity covers that
        for key in sorted(keys - sent - set(_NOT_FOR_CALLERS)):
            unreachable.append(f"{method} {path}: {key!r}")

    assert not unreachable, (
        "coordinator routes accept fields the MCP surface cannot send: "
        f"{unreachable}. An operator whose only access is MCP -- which is the "
        "case for the agent operator -- cannot reach these at all, and "
        "operation-level surface parity does not notice because the operation "
        "itself exists everywhere."
    )


def test_the_exemptions_still_describe_something_real() -> None:
    """An exemption for a key no route reads is a stale excuse."""
    read = {key for keys in _route_payload_keys().values() for key in keys}
    stale = sorted(set(_NOT_FOR_CALLERS) - read)

    assert not stale, f"exemptions for keys no route reads any more: {stale}"


def test_the_scan_actually_joins_the_two_sides() -> None:
    """A scanner that silently matched nothing would pass forever.

    The first draft of this file did exactly that: route decorators omit the
    router's ``/v1`` prefix, so no path ever matched an MCP tool, every route
    took the ``sent is None`` branch, and the check reported success having
    compared nothing. Reverting the fix it was written for did not fail it.

    Asserting each side separately is not enough -- both sides were fine. It is
    the **join** that has to be shown to happen, which is the same lesson as
    every other instance here: verify the thing you actually depend on, not the
    parts around it.
    """
    routes = _route_payload_keys()
    mcp = _mcp_sent_keys()

    assert len(routes) >= 4, f"payload-reading routes dropped to {len(routes)}; scan may be stale"

    joined = {key for key in routes if (key[0], _normalise(key[1])) in mcp}
    assert joined, (
        "no payload-reading route matched any MCP tool, so this guardrail is "
        "comparing nothing at all"
    )

    buildspec = ("PUT", "/buildspecs/{name}")
    assert buildspec in joined, "the route this guardrail exists for is not being compared"
    assert "entrypoint" in routes[buildspec]
