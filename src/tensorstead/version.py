"""Application release version.

Keep this value aligned with ``[project].version`` in ``pyproject.toml``.  A
test enforces that relationship so the coordinator, agent, package metadata,
and build records all identify the same release.
"""

VERSION = "1.0.0"


def build_identity() -> dict[str, object]:
    """Which build this process is, or unknown.

    ``VERSION`` cannot answer it: it has read the same string for sixty
    consecutive builds, so an operator comparing a live appliance against a
    controller had nothing to compare. The build script writes ``_build.py``
    into the package immediately before building, so a wheel carries its own
    identity and the running process can state it.

    Every field is ``None`` in a source checkout and in any wheel not produced
    by that script. Unknown is the honest answer there, and deliberately not a
    guess -- a build number invented at import time would be exactly the kind
    of confident fiction this reports in order to prevent.
    """
    try:
        from tensorstead import _build  # type: ignore[attr-defined]
    except ImportError:
        return {"build_number": None, "git_revision": None, "built_at": None, "dirty": None}
    return {
        "build_number": getattr(_build, "BUILD_NUMBER", None),
        "git_revision": getattr(_build, "GIT_REVISION", None),
        "built_at": getattr(_build, "BUILT_AT", None),
        # A build made from a modified tree cannot be reproduced from its
        # revision, and an operator comparing revisions deserves to know.
        "dirty": getattr(_build, "SOURCE_TREE_DIRTY", None),
    }
