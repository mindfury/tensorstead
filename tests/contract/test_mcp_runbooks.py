"""Runbooks served to agents over MCP.

The model-swap runbook exists because an agent repointed a running deployment
at another model. These tests hold it to being reachable from the MCP server,
pointed to from the server's instructions, identical to its docs copy, and
written in terms of tools that exist -- a runbook naming a tool that was
renamed would send the next agent looking for something that is not there.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from tensorstead.runbooks import get_runbook, list_runbooks

pytestmark = pytest.mark.contract

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def _server() -> Any:
    from tensorstead.mcp.server import build_server

    return build_server()


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in asyncio.run(_server().list_tools())}


def test_the_runbook_tools_are_published() -> None:
    tools = _tools()
    assert "runbook_list" in tools
    assert "runbook_get" in tools


def test_runbook_get_returns_the_model_swap_runbook() -> None:
    result = asyncio.run(_server().call_tool("runbook_get", {"name": "model-swap"}))
    assert "# Model swap" in str(result)


def test_an_unknown_runbook_names_the_available_ones() -> None:
    result = asyncio.run(_server().call_tool("runbook_get", {"name": "nope"}))
    assert "not_found" in str(result)
    assert "model-swap" in str(result)


def test_the_server_instructions_point_at_the_runbook() -> None:
    assert "runbook_get('model-swap')" in (_server().instructions or "")


def test_deployment_modify_warns_against_swapping_models() -> None:
    description = _tools()["deployment_modify"].description or ""
    assert "model_change_refused" in description
    assert "model-swap" in description


def test_every_runbook_has_an_identical_docs_copy() -> None:
    for runbook in list_runbooks():
        name = runbook["name"]
        assert (_DOCS / f"{name}.md").read_text(encoding="utf-8") == get_runbook(name), (
            f"docs/{name}.md and src/tensorstead/runbooks/{name}.md differ"
        )


def _error_codes() -> set[str]:
    from tensorstead.domain.errors import DomainError

    codes: set[str] = set()
    pending = list(DomainError.__subclasses__())
    while pending:
        cls = pending.pop()
        codes.add(cls.code)
        pending.extend(cls.__subclasses__())
    return codes


def test_every_tool_a_runbook_names_exists() -> None:
    tools = set(_tools())
    codes = _error_codes()
    pattern = re.compile(r"`((?:node|model|deployment|operation|runbook|image|runtime)_[a-z_]+)`")
    for runbook in list_runbooks():
        named = {
            name
            for name in pattern.findall(get_runbook(runbook["name"]) or "")
            # Error codes and fields (``model_id``, ``runtime_config``) share the
            # prefixes; by convention a field ends in _id, _config or _name.
            if name not in codes and not name.endswith(("_id", "_config", "_name"))
        }
        assert named, f"{runbook['name']} names no tools"
        assert named <= tools, f"{runbook['name']} names missing tools: {sorted(named - tools)}"


def test_every_tool_a_skill_names_exists() -> None:
    tools = set(_tools())
    codes = _error_codes()
    pattern = re.compile(r"`((?:node|model|deployment|operation|runbook|image|runtime)_[a-z_]+)`")
    skills = sorted((_DOCS.parent / "skills").glob("*/SKILL.md"))
    assert skills, "no skills found"
    for skill in skills:
        text = skill.read_text(encoding="utf-8")
        assert text.startswith("---\nname: "), f"{skill} has no frontmatter name"
        named = {
            name
            for name in pattern.findall(text)
            if name not in codes and not name.endswith(("_id", "_config", "_name"))
        }
        assert named <= tools, f"{skill} names missing tools: {sorted(named - tools)}"
