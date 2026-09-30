"""GUARDRAIL: the product must describe its own inputs.

A UX study of the 0.1.8 estate found 0 of 28 CLI options carried help text and
all 8 `deployment_create` MCP parameters were undocumented. A user could not
learn from the tool what the tool required, and neither could an agent.

Nothing failed when that was true, which is why it stayed true. These tests
make it fail.
"""

from __future__ import annotations

import asyncio

import pytest
import typer.testing

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.vllm import VLLMAdapter
from tensorstead.cli.main import app
from tensorstead.mcp.server import build_server

pytestmark = pytest.mark.contract

runner = typer.testing.CliRunner()

# Every command group and the commands beneath it that take options.
_COMMANDS = [
    ["node", "register"],
    ["node", "list"],
    ["node", "show"],
    ["model", "acquire"],
    ["model", "list"],
    ["runtime", "list"],
    ["deployment", "create"],
    ["deployment", "start"],
    ["deployment", "stop"],
    ["deployment", "list"],
    ["deployment", "export"],
    ["image", "list"],
    ["operation", "list"],
    ["credential", "set"],
    ["credential", "list"],
]


_SCANNED: list[str] = []


def _undocumented_options(argv: list[str]) -> list[str]:
    """Option flags whose help column is empty in `--help` output."""
    import re

    result = runner.invoke(app, [*argv, "--help"])
    if result.exit_code != 0:
        return []
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    bare: list[str] = []
    for line in plain.splitlines():
        m = re.match(r"^\s*[│|]\s*(--[a-z0-9-]+)\s+(.*?)\s*[│|]\s*$", line)
        if not m:
            continue
        flag, rest = m.group(1), m.group(2)
        # Strip a type placeholder such as <str> before judging emptiness.
        rest = re.sub(r"^<[a-z]+>", "", rest).strip()
        if flag == "--help":
            continue
        _SCANNED.append(f"{' '.join(argv)} {flag}")
        if len(rest) < 5:
            bare.append(f"{' '.join(argv)} {flag}")
    return bare


def test_every_cli_option_carries_help_text() -> None:
    """A user must be able to learn the tool from the tool."""
    offenders: list[str] = []
    for argv in _COMMANDS:
        offenders.extend(_undocumented_options(argv))

    assert offenders == [], f"CLI options with no help text: {offenders}"
    # A parser that silently stops matching would make this test pass forever
    # while checking nothing -- the exact failure this suite exists to catch.
    assert len(_SCANNED) > 25, (
        f"the help parser matched only {len(_SCANNED)} options; the output "
        "format probably changed and this guardrail has stopped checking"
    )


def test_every_mcp_tool_parameter_carries_a_description() -> None:
    """The agent surface has the same obligation as the human one."""
    server = build_server()
    tools = asyncio.run(server.list_tools())

    undocumented = [
        f"{tool.name}.{param}"
        for tool in tools
        for param, schema in (tool.input_schema or {}).get("properties", {}).items()
        if not (schema.get("description") or "").strip()
    ]

    assert undocumented == [], f"MCP parameters with no description: {undocumented}"


def test_every_mcp_tool_carries_a_description() -> None:
    server = build_server()
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools if not (t.description or "").strip()] == []


@pytest.mark.parametrize(
    ("adapter", "bad_key", "expected_key"),
    [
        (VLLMAdapter(), "gpu_memory", "gpu_memory_utilization"),
        (LlamaCppAdapter(), "n_gpu_layer", "n_gpu_layers"),
    ],
)
def test_a_rejected_config_key_names_the_accepted_ones(
    adapter: object, bad_key: str, expected_key: str
) -> None:
    """An error that does not teach sends the user back to guessing."""
    with pytest.raises(ValueError) as caught:
        adapter.validate_config({bad_key: 1})  # type: ignore[attr-defined]

    message = str(caught.value)
    assert "accepted" in message, "the error must list what is accepted"
    assert expected_key in message, (
        f"the accepted-key list must contain {expected_key!r}, so a user who "
        f"guessed {bad_key!r} can find the real name"
    )


def test_the_cli_reports_its_own_version_without_a_coordinator() -> None:
    """The first question, answerable with nothing configured."""
    from tensorstead.version import VERSION

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert VERSION in result.output


def test_status_names_the_target_its_source_and_the_fix_when_unconfigured() -> None:
    """Say what was tried, where it came from, and the remedy.

    "Where did this value come from" is the question an operator could not
    answer: the CLI silently used a built-in default and gave no hint that a
    config file was even a concept, let alone where it lived.
    """
    result = runner.invoke(app, ["status"], env={"TENSORSTEAD_MGMT_TOKEN": ""})

    assert "coordinator" in result.output
    assert "built-in default" in result.output, "an unconfigured value must say so"
    assert "config file" in result.output, "the config path must be named even when absent"
    assert "stead login" in result.output, "the remedy must be named"


def test_starting_a_deployment_records_the_image_it_pulled() -> None:
    """`image list` was permanently empty because nothing wrote to it.

    `save_image` existed on the port and in the SQLite adapter and was called
    from nowhere, so `image_records` was never populated: image discovery
    returned nothing and `image delete` could never find an
    artifact to remove or to refuse. An always-empty inventory is
    indistinguishable from a node holding no images.
    """
    from datetime import datetime

    from tensorstead.domain.models import ImageRecord

    saved: list[ImageRecord] = []

    class _Repo:
        def save_image(self, image: ImageRecord) -> None:
            saved.append(image)

    from tensorstead.service.lifecycle import LifecycleService

    service = object.__new__(LifecycleService)
    object.__setattr__(service, "_repo", _Repo())

    class _Revision:
        image_reference = "registry.example/vllm:1.2.3"

    class _Node:
        id = "node-a"

    LifecycleService._record_image(
        service,
        _Node(),  # type: ignore[arg-type]
        _Revision(),
        {"image_digest": "sha256:abcdef"},
    )

    assert len(saved) == 1, "the pulled image must be recorded"
    assert saved[0].digest == "sha256:abcdef"
    assert saved[0].reference == "registry.example/vllm:1.2.3"
    assert saved[0].node_id == "node-a"
    assert isinstance(saved[0].pulled_at, datetime)


def test_a_response_without_a_digest_records_nothing() -> None:
    """An agent that reports no digest must not create a bogus inventory row."""
    from tensorstead.service.lifecycle import LifecycleService

    saved: list[object] = []

    class _Repo:
        def save_image(self, image: object) -> None:
            saved.append(image)

    service = object.__new__(LifecycleService)
    object.__setattr__(service, "_repo", _Repo())

    for response in ({}, {"image_digest": ""}, None, "not-a-dict"):
        LifecycleService._record_image(
            service,
            type("N", (), {"id": "n"})(),
            type("R", (), {"image_reference": "r"})(),
            response,
        )

    assert saved == []
