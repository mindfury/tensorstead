"""Keep the tail of a stream within a fixed size *while* it is captured.

Build and probe output was captured whole and trimmed afterwards: buildx ran
with ``capture_output=True``, a classic build's error log was accumulated line
by line, and probe/runtime logs were fetched as the last N lines -- one
newline-free record of any size counts as one line. An image or build step
that floods its output could exhaust the agent's memory before any trimming ran
(CodeGuard CG-07). Everything here is bounded as it arrives.
"""

from __future__ import annotations

from collections.abc import Iterable

# Enough for any diagnosis this product renders (the failing step, the
# compiler's message), small enough to be irrelevant to the agent's memory.
DEFAULT_MAX_CHARS = 1024 * 1024
DEFAULT_MAX_LINE_CHARS = 16 * 1024


class BoundedTail:
    """The last ``max_chars`` characters of everything appended, nothing more."""

    def __init__(
        self, max_chars: int = DEFAULT_MAX_CHARS, max_line_chars: int = DEFAULT_MAX_LINE_CHARS
    ) -> None:
        self._max_chars = max_chars
        self._max_line_chars = max_line_chars
        self._text = ""
        self.truncated = False

    def append(self, chunk: str | bytes) -> None:
        text = chunk.decode("utf-8", "replace") if isinstance(chunk, bytes) else chunk
        if len(text) >= self._max_chars:
            self._text = text[-self._max_chars :]
            self.truncated = True
            return
        combined = self._text + text
        if len(combined) > self._max_chars:
            combined = combined[-self._max_chars :]
            self.truncated = True
        self._text = combined

    def extend(self, chunks: Iterable[str | bytes]) -> None:
        for chunk in chunks:
            self.append(chunk)

    @property
    def text(self) -> str:
        return self._text

    def lines(self) -> list[str]:
        """Non-blank lines, each cut to ``max_line_chars`` (keeping its end)."""
        return [
            line if len(line) <= self._max_line_chars else "…" + line[-self._max_line_chars :]
            for line in self._text.splitlines()
            if line.strip()
        ]
