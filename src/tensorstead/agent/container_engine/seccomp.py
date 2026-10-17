"""Seccomp exceptions on top of Docker's default profile.

Docker applies its default profile to every container, and the API offers no way
to *extend* it: a container gets that profile, a complete replacement, or none at
all (``--privileged``, ``seccomp=unconfined``). So an exception is expressed as
the default profile shipped here plus the approved groups, sent whole.

Groups are named and defined here, never in configuration. A deployment record
says ``io_uring``; what that means in syscalls is this module's reviewed answer,
so an operator cannot widen the opening by spelling a longer list.

The shipped default is Docker 29.9.1's. A node running a different Docker release
gets that release's default for every container *without* an exception, and this
copy for a container with one -- a difference recorded here rather than hidden:
the profile has changed little across releases and only by additions.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any, Final

_DEFAULT_PROFILE: Final = "docker-default-v29.9.1.json"

# Reviewed syscall groups. ``io_uring`` is the three calls that create, drive
# and register buffers with a ring; Docker's default denied all three from
# release 25 on, after a run of kernel vulnerabilities reachable through them.
SYSCALL_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "io_uring": ("io_uring_setup", "io_uring_enter", "io_uring_register"),
}


class UnknownSeccompGroupError(ValueError):
    """A requirement named a group this module does not define."""


@cache
def _default_profile_text() -> str:
    return (
        resources.files(__package__)
        .joinpath("seccomp_profiles", _DEFAULT_PROFILE)
        .read_text(encoding="utf-8")
    )


def profile_with(groups: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Docker's default profile with ``groups`` allowed, as a new dict.

    Refuses a name it does not define rather than skipping it: a requirement
    that asked for an opening and silently did not get one would start a
    container whose record claims a confinement it does not have -- or, here,
    one that fails at load with an error naming neither seccomp nor this.
    """
    unknown = sorted(set(groups) - set(SYSCALL_GROUPS))
    if unknown:
        raise UnknownSeccompGroupError(f"no seccomp group named {', '.join(unknown)}")
    profile: dict[str, Any] = json.loads(_default_profile_text())
    names = sorted({name for group in groups for name in SYSCALL_GROUPS[group]})
    if names:
        profile["syscalls"].append({"names": names, "action": "SCMP_ACT_ALLOW"})
    return profile


def security_opt(groups: list[str] | tuple[str, ...]) -> list[str]:
    """The ``security_opt`` entry Docker takes for an inline seccomp profile."""
    return [f"seccomp={json.dumps(profile_with(groups), separators=(',', ':'))}"]
