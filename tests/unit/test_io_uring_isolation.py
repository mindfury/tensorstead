"""``allow_io_uring``: a reviewed, approval-gated seccomp exception.

NVIDIA's dual-Spark GLM-5.3 recipe loads its checkpoint with ``--load-format
b12x``, which reads through io_uring, and gets io_uring by running the container
``--privileged``. Under Docker's default seccomp profile the loader died with
"io_uring bounce initialization failed: Operation not permitted" (2026-10-16).

These tests hold the opening to its intended size: refused without an approval,
never on vLLM's argv, and -- when approved -- Docker's default profile plus the
three io_uring syscalls and nothing else.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorstead.adapters.runtimes.option_policy import (
    RELAXES_ISOLATION,
    authorize,
    classify,
    is_approvable,
)
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.agent.container_engine import seccomp
from tensorstead.agent.container_engine.docker_py import _apply_requirements
from tensorstead.ports.runtime_adapter import ContainerRequirements, NodePosition
from tensorstead.service.approvals import requested_code_options

pytestmark = pytest.mark.unit

_APPROVED = frozenset({"allow_io_uring"})
_IO_URING = {"io_uring_setup", "io_uring_enter", "io_uring_register"}


# ------------------------------------------------------------------ the policy


def test_allow_io_uring_is_classified_and_approvable() -> None:
    assert classify("allow_io_uring") == RELAXES_ISOLATION
    assert classify("allow-io-uring") == RELAXES_ISOLATION
    assert is_approvable("allow_io_uring")


def test_without_an_approval_it_is_refused_naming_the_remedy() -> None:
    authorized, reason = authorize("allow_io_uring")
    assert not authorized
    assert reason is not None
    assert "confinement" in reason
    assert "code_approval_create" in reason


def test_an_approval_for_another_option_does_not_open_it() -> None:
    authorized, _ = authorize("allow_io_uring", approved=frozenset({"trust_remote_code"}))
    assert not authorized


def test_the_coordinator_sees_it_as_needing_an_approval() -> None:
    assert requested_code_options({"allow_io_uring": True}) == {"allow_io_uring"}
    assert requested_code_options({"allow_io_uring": False}) == frozenset()


def test_it_cannot_arrive_through_the_passthrough() -> None:
    with pytest.raises(ValueError):
        VLLMAdapter().validate_config(
            {"extra_args": {"allow-io-uring": True}}, approved_options=_APPROVED
        )


# ------------------------------------------------------------------- the adapter


def test_refused_by_the_adapter_without_an_approval() -> None:
    with pytest.raises(ValueError, match="allow_io_uring"):
        VLLMAdapter().validate_config({"allow_io_uring": True})


def test_accepted_by_the_adapter_with_an_approval() -> None:
    validated = VLLMAdapter().validate_config({"allow_io_uring": True}, approved_options=_APPROVED)
    assert validated["allow_io_uring"] is True


def test_it_never_reaches_vllm_argv() -> None:
    adapter = VLLMAdapter()
    config = adapter.validate_config({"allow_io_uring": True}, approved_options=_APPROVED)
    args = adapter.build_launch_args(config, model_path="/var/lib/tensorstead/models/x")
    assert not any("io" in arg and "uring" in arg for arg in args)


@pytest.mark.parametrize(
    "position",
    [None, NodePosition(node_index=1, node_count=2, self_address="b", peer_addresses=["a", "b"])],
)
def test_approved_config_declares_the_seccomp_group(position: NodePosition | None) -> None:
    adapter = VLLMAdapter()
    config = adapter.validate_config({"allow_io_uring": True}, approved_options=_APPROVED)
    assert adapter.container_requirements(config, position).seccomp_allow == ["io_uring"]


def test_absent_it_declares_nothing() -> None:
    assert VLLMAdapter().container_requirements({}).seccomp_allow == []


# ----------------------------------------------------------- the agent's profile


def _default() -> dict:
    return seccomp.profile_with([])


def test_the_shipped_default_denies_io_uring() -> None:
    """Why this exists: Docker's default leaves io_uring to the deny-by-default action."""
    profile = _default()
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    allowed = {
        name
        for rule in profile["syscalls"]
        if rule["action"] == "SCMP_ACT_ALLOW"
        for name in rule["names"]
    }
    assert not allowed & _IO_URING


def test_the_exception_is_the_default_plus_exactly_three_syscalls() -> None:
    base = _default()
    widened = seccomp.profile_with(["io_uring"])
    assert widened["syscalls"][:-1] == base["syscalls"]
    assert widened["syscalls"][-1] == {"names": sorted(_IO_URING), "action": "SCMP_ACT_ALLOW"}
    assert {k: v for k, v in widened.items() if k != "syscalls"} == {
        k: v for k, v in base.items() if k != "syscalls"
    }


def test_the_shipped_default_is_not_mutated_between_calls() -> None:
    seccomp.profile_with(["io_uring"])
    assert len(_default()["syscalls"]) == len(seccomp.profile_with([])["syscalls"])


def test_an_undefined_group_is_refused_not_skipped() -> None:
    with pytest.raises(seccomp.UnknownSeccompGroupError, match="ptrace"):
        seccomp.profile_with(["ptrace"])


# ----------------------------------------------------------- the docker create


def test_docker_create_carries_the_inline_profile() -> None:
    create_args: dict = {}
    _apply_requirements(create_args, ContainerRequirements(seccomp_allow=["io_uring"]), None)
    (opt,) = create_args["security_opt"]
    assert opt.startswith("seccomp=")
    profile = json.loads(opt.removeprefix("seccomp="))
    assert set(profile["syscalls"][-1]["names"]) == _IO_URING
    assert "privileged" not in create_args


def test_docker_create_without_the_requirement_is_unchanged() -> None:
    create_args: dict = {}
    _apply_requirements(create_args, ContainerRequirements(), SimpleNamespace())
    assert "security_opt" not in create_args
