"""Runbooks served to agents through MCP (``runbook_list``, ``runbook_get``).

Each runbook is a Markdown file in this package, shipped in the wheel so the
MCP server can serve it from any install. Its first line is the title
(``# ...``) and its first paragraph the summary. ``docs/<name>.md`` carries the
same text for people, and a test keeps the two identical.
"""

from __future__ import annotations

from importlib import resources


def _files() -> dict[str, str]:
    root = resources.files(__name__)
    return {
        entry.name.removesuffix(".md"): entry.read_text(encoding="utf-8")
        for entry in root.iterdir()
        if entry.name.endswith(".md")
    }


def _title_and_summary(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    title = lines[0].removeprefix("# ").strip() if lines else ""
    paragraph: list[str] = []
    for line in lines[1:]:
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            break
    return title, " ".join(paragraph)


def list_runbooks() -> list[dict[str, str]]:
    """Every runbook's name, title and one-paragraph summary, by name."""
    listed = []
    for name, text in sorted(_files().items()):
        title, summary = _title_and_summary(text)
        listed.append({"name": name, "title": title, "summary": summary})
    return listed


def get_runbook(name: str) -> str | None:
    """The full text of one runbook, or None when there is none by that name."""
    return _files().get(name)
