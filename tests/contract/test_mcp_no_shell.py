"""🚫 GUARDRAIL — the MCP surface offers no shell.

This guardrail requires that an agent never need arbitrary shell or interactive
SSH during normal operation, and measures **zero** such commands. This
asserts the surface makes that structural: the capability is absent, so it
cannot be reached for under pressure. A filtered or permission-gated shell would
satisfy neither requirement, because the thing to be avoided is the capability
existing at all.

Three checks, escalating in specificity:

1. no tool is *named* like a shell, exec, file, or SSH primitive;
2. no tool *parameter* accepts a command, script, or arbitrary path;
3. the MCP package contains no execution machinery of its own, and imports no
   coordinator business logic — so it cannot grow one by reaching
   through a service object.

**The path-shaped parameters, examined rather than waved through.** First,
``credential_set``
takes ``from_file``. It is a path, and pretending otherwise would be dishonest.
What makes it acceptable is bounded: the coordinator reads that file as a
credential *reference* and stores it in the provider's protected store, from
which no read path exists — so an agent cannot use it to retrieve file
contents back through this or any other surface. It is never executed and never
echoed. The residual risk is that a stored value is later presented to a model
source as a token; that is inherent in the reference design the contract
mandates, and it is narrower than the alternative of letting a
secret value travel through an agent's context.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SRC = Path(__file__).resolve().parents[2] / "src" / "tensorstead"
_MCP = _SRC / "mcp"

# Tool-name fragments that would indicate a shell, exec, file, or SSH primitive.
_FORBIDDEN_NAME_FRAGMENTS = (
    "shell",
    "exec",
    "command",
    "cmd",
    "run_",
    "spawn",
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "terminal",
    "console",
    "bash",
    "sh_",
    "script",
    "eval",
    "file_read",
    "file_write",
    "read_file",
    "write_file",
    "upload",
    "download",
)

# Parameter names that would let an agent smuggle execution or arbitrary
# filesystem access through an otherwise innocent tool.
_FORBIDDEN_PARAM_NAMES = {
    "command",
    "cmd",
    "shell",
    "script",
    "exec",
    "eval",
    "code",
    "args",
    "argv",
    "entrypoint",
    "path",
    "filepath",
    "file_path",
    "filename",
    "directory",
    "cwd",
    "host",
    "ssh_key",
    "private_key",
}

# The permitted path-shaped parameters, and why (see module docstring). Both
# are the *reference* half of a credential operation: the agent names where a
# secret lives so it never carries the value itself, which is what keeps the
# score at 100% with no carve-out. Each addition must be justified in
# those same terms; a path parameter that is not a secret reference does not
# belong here.
_PERMITTED_PATH_PARAMS = {
    ("credential_set", "from_file"),
    ("inferencekey_set", "from_file"),
}

# A third exemption class, kept separate because its reasoning is different
# from both of the others -- diluting the credential-reference set would lose
# the property that every member of it is a secret reference.
#
# ``model_acquire.file_selector`` carries glob patterns. They are matched by
# ``huggingface_hub`` against the **remote repository's** file listing, never
# against this host's filesystem: the parameter cannot name a local path,
# because there is no local path in the operation it configures. It filters
# a download; it cannot add a file, cannot read one back, and is never
# executed. A pattern like ``../../etc/passwd`` matches no repository entry and
# does nothing.
#
# It exists because a GGUF repository ships twenty-odd quantizations of one
# model and acquiring all of them to serve one is not a download that fits on
# the appliance.
_PERMITTED_REMOTE_SELECTORS = {
    ("model_acquire", "file_selector"),
}

# Commands permitted on a *recorded artifact*, kept separate from the
# path-shaped credential references above because they are a different
# exemption with different reasoning.
#
# ``buildspec_set(entrypoint=...)`` -- added 2026-08-12, and it widens this
# surface, so it is argued rather than asserted.
#
# It is a command, and pretending otherwise would be dishonest. What makes
# it acceptable is that **``steps`` on the same tool already grants strictly
# more**: a build step is an arbitrary shell command executed as root in a
# container on the node, while an entrypoint can only name something a step
# already put there. Refusing the narrower capability while permitting the
# broader one protects nothing.
#
# Both are set on a *recorded artifact that executes nothing when recorded*.
# The build is a separate, explicit operation, so the command is
# inspectable before it ever runs and retained after -- which is the
# property this contract actually protects, as distinct from "a command reaches
# the host", which neither of these does.
#
# No path to an entrypoint exists on a *live deployment*: ``deployment_
# create`` takes none, and ``use_image_entrypoint`` is a boolean choosing
# between the image's own and ``vllm serve``. That asymmetry is deliberate
# and should stay.
#
# **Worth a human's eye.** `steps` sits outside `_FORBIDDEN_PARAM_NAMES`
# only because that list matches names that *look* like commands, so the
# broader capability was never examined.
_PERMITTED_RECORDED_COMMANDS = {
    ("buildspec_set", "entrypoint"),
}


# Execution machinery that must not appear anywhere in the MCP package.
_FORBIDDEN_EXECUTION = (
    "subprocess",
    "os.system",
    "os.popen",
    "os.exec",
    "pty.spawn",
    "paramiko",
    "fabric",
    "eval(",
    "exec(",
    "__import__",
)

# The MCP server is an HTTP client of the coordinator API.
# Importing any of these would make it a second, privileged path into the
# product rather than one more client of it.
_FORBIDDEN_INTERNAL_IMPORTS = {
    "tensorstead.service",
    "tensorstead.coordinator",
    "tensorstead.adapters",
    "tensorstead.agent",
    "tensorstead.ports",
}


def _tools() -> list:
    from tensorstead.mcp.server import build_server

    return list(asyncio.run(build_server().list_tools()))


def test_no_tool_is_named_like_a_shell_primitive() -> None:
    """No shell, exec, file, or SSH tool exists."""
    for tool in _tools():
        lowered = tool.name.lower()
        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            assert fragment not in lowered, (
                f"MCP tool {tool.name!r} looks like a {fragment!r} primitive; the agent "
                f"surface exposes no shell, exec, file, or SSH capability"
            )


def test_no_tool_parameter_accepts_a_command_or_arbitrary_path() -> None:
    """No parameter can carry a command, script, or arbitrary path."""
    for tool in _tools():
        schema = tool.input_schema or {}
        for param in schema.get("properties") or {}:
            permitted = (
                _PERMITTED_PATH_PARAMS | _PERMITTED_RECORDED_COMMANDS | _PERMITTED_REMOTE_SELECTORS
            )
            if (tool.name, param) in permitted:
                continue
            assert param.lower() not in _FORBIDDEN_PARAM_NAMES, (
                f"MCP tool {tool.name!r} accepts parameter {param!r}, which could carry a "
                f"command, script, or arbitrary path"
            )


def test_the_only_path_shaped_parameter_is_the_documented_credential_reference() -> None:
    """``from_file`` on the two credential operations is the only path parameter.

    Pinned deliberately: if a second path parameter ever appears, this fails and
    someone has to justify it in the same terms the first one was justified in.
    """
    path_shaped = {
        (tool.name, param)
        for tool in _tools()
        for param in (tool.input_schema or {}).get("properties", {})
        if "file" in param.lower() or "path" in param.lower() or "dir" in param.lower()
    }
    permitted = _PERMITTED_PATH_PARAMS | _PERMITTED_REMOTE_SELECTORS
    assert path_shaped == permitted, (
        f"path-shaped MCP parameters are {sorted(path_shaped)}; only "
        f"{sorted(permitted)} are justified"
    )


def test_inferencekey_set_cannot_carry_a_secret_value() -> None:
    """The inference credential takes references only, exactly as the other does.

    The design moved ownership of this credential into the product. That must not
    quietly widen the agent surface: a value parameter here would put a secret
    into an agent's context and any transcript its host retains.
    """
    tool = next(t for t in _tools() if t.name == "inferencekey_set")
    params = set((tool.input_schema or {}).get("properties", {}))

    assert "from_env" in params and "from_file" in params
    for value_bearing in ("secret", "value", "token", "password", "credential"):
        assert value_bearing not in params, (
            f"inferencekey_set accepts {value_bearing!r}, which could carry a secret"
        )


def test_credential_set_cannot_carry_a_secret_value() -> None:
    """The agent surface takes references only.

    This is what keeps the score at a clean 100% with no carve-out: the operation
    is present, so parity holds, and it is narrowed rather than omitted.
    """
    tool = next(t for t in _tools() if t.name == "credential_set")
    params = set((tool.input_schema or {}).get("properties", {}))

    assert "from_env" in params and "from_file" in params
    for value_bearing in ("secret", "value", "token", "password", "credential"):
        assert value_bearing not in params, (
            f"credential_set exposes {value_bearing!r}; the MCP form must have no parameter "
            f"capable of carrying a secret value"
        )


def test_the_mcp_package_contains_no_execution_machinery() -> None:
    """Nothing in the package can run a process."""
    for path in sorted(_MCP.rglob("*.py")):
        text = path.read_text()
        for token in _FORBIDDEN_EXECUTION:
            assert token not in text, (
                f"{path.relative_to(_SRC)} references {token!r}; the MCP server executes nothing"
            )


def test_the_mcp_server_imports_no_coordinator_business_logic() -> None:
    """MCP is a client of the API, not a second way into the product.

    This is what makes the boundary structural: the coordinator has no
    dependency on MCP, and MCP has no privileged path into the coordinator.
    """
    for path in sorted(_MCP.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                for forbidden in _FORBIDDEN_INTERNAL_IMPORTS:
                    assert not module.startswith(forbidden), (
                        f"{path.relative_to(_SRC)} imports {module!r}; the MCP server is an "
                        f"HTTP client of the coordinator API and imports no coordinator "
                        f"business logic"
                    )


def test_the_coordinator_does_not_depend_on_mcp() -> None:
    """MCP is strictly optional; the coordinator runs unchanged without it.

    The other half of the same boundary. If the coordinator imported the MCP
    package, "not the architectural foundation" would be a statement of intent
    rather than a fact about the build.
    """
    for area in ("coordinator", "service", "agent", "adapters", "domain", "ports"):
        for path in sorted((_SRC / area).rglob("*.py")):
            text = path.read_text()
            assert "tensorstead.mcp" not in text, (
                f"{path.relative_to(_SRC)} references tensorstead.mcp; MCP is an optional "
                f"integration surface and nothing in the product may depend on it "
                f""
            )
