"""GUARDRAIL: no function whose body only discards a value.

An audit found this in `NodeHTTPClient._pin_fingerprint`:

    def _pin_fingerprint(self, node: Node) -> None:
        \"\"\"Validate the peer certificate fingerprint against the pinned one.\"\"\"
        _ = node.agent_cert_fingerprint

It was named as a check, documented as a check, called on every agent request,
and did nothing. Ruff's B018 catches the bare form (`node.agent_cert_fingerprint`
as a statement) but not this one, because assigning to `_` is the idiom for
"deliberately discarded" -- B018's own remediation message suggests exactly that
substitution. So the linter cannot see it and this test must.

A function that means to do nothing should say so with `pass`, `...`, or
`raise NotImplementedError`. Each of those is honest. Reading a value and
throwing it away looks like work.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"


def _is_discard(statement: ast.stmt) -> bool:
    """True for `_ = <expr>` and `_ , _ = ...` style throwaway assignments."""
    if not isinstance(statement, ast.Assign):
        return False
    return all(isinstance(target, ast.Name) and target.id == "_" for target in statement.targets)


def _significant_body(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """The body minus its docstring."""
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def test_no_function_body_is_only_a_discarded_value() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = _significant_body(node)
            if body and all(_is_discard(statement) for statement in body):
                where = path.relative_to(_SRC.parent.parent)
                offenders.append(f"{where}:{node.lineno} {node.name}")

    assert offenders == [], (
        "these functions only read a value and discard it, which reads as an "
        "implementation but performs no work. Use pass, ..., or "
        f"raise NotImplementedError instead: {offenders}"
    )
