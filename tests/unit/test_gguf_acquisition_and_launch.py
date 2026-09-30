"""Serving a GGUF model: what the two runtimes' assumptions had quietly baked in.

Nothing on this estate had ever run llama.cpp, and four things assumed a
transformers-shaped model would be the only thing acquired:

* the source verified a tree by requiring ``config.json``, which GGUF never has
  because the format embeds that metadata itself;
* the llama.cpp adapter passed ``--model`` the deployment's *directory*, while
  ``llama-server`` wants a file — its own ``validate_model_path`` said so and
  nothing acted on it;
* a repository holding many quantizations had no way to say which one was meant;
* and the coordinator would have sent a selection to an agent whose request
  model forbids unknown keys.

These pin each one. They are unit tests against real temporary trees rather
than mocks, because every defect here was an assumption about what is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.sources.huggingface import HuggingFaceSource

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------- verifying
def test_a_gguf_tree_verifies_without_config_json(tmp_path: Path) -> None:
    """GGUF is self-describing; demanding config.json would fail every one.

    The cost of getting this wrong is paid at the worst moment: after the whole
    download, at the last step before promotion.
    """
    (tmp_path / "model-UD-Q8_K_XL.gguf").write_bytes(b"GGUF\x00")
    HuggingFaceSource().verify(tmp_path)  # does not raise


def test_a_selected_gguf_tree_in_a_repository_subdirectory_verifies(tmp_path: Path) -> None:
    """Hub selections retain repository directories below the staged root."""
    selected = tmp_path / "UD-IQ1_S"
    selected.mkdir()
    (selected / "model-00001-of-00003.gguf").write_bytes(b"GGUF\x00")
    (selected / "model-00002-of-00003.gguf").write_bytes(b"GGUF\x00")
    HuggingFaceSource().verify(tmp_path)  # does not raise


def test_an_empty_gguf_is_still_refused(tmp_path: Path) -> None:
    """An interrupted download must never be promoted."""
    (tmp_path / "model.gguf").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        HuggingFaceSource().verify(tmp_path)


def test_a_tree_with_neither_is_still_refused(tmp_path: Path) -> None:
    """The GGUF path must not become a way for any bad tree to pass."""
    (tmp_path / "README.md").write_text("nothing usable here")
    with pytest.raises(ValueError, match=r"config\.json"):
        HuggingFaceSource().verify(tmp_path)


def test_a_transformers_tree_is_verified_exactly_as_before(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3"]}))
    HuggingFaceSource().verify(tmp_path)


# ------------------------------------------------------------------ selecting
class _RecordingHub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def resolve_revision(self, model_id: str, revision: str | None) -> str:
        return revision or "sha1"

    def snapshot_download(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"resolved_revision": "sha1"}


def test_a_selection_reaches_the_hub_as_allow_patterns(tmp_path: Path) -> None:
    hub = _RecordingHub()
    HuggingFaceSource(hub=hub).acquire(
        "unsloth/repo-GGUF",
        "sha1",
        str(tmp_path),
        None,
        file_selector=("m-UD-Q8_K_XL.gguf", "mmproj-BF16.gguf"),
    )
    assert hub.calls[0]["allow_patterns"] == ["m-UD-Q8_K_XL.gguf", "mmproj-BF16.gguf"]


def test_a_whole_repository_sends_no_filter_at_all(tmp_path: Path) -> None:
    """Not ``allow_patterns=None`` — absent.

    A hub client that predates the argument keeps working for every model
    acquired before selection existed, which is all of them.
    """
    hub = _RecordingHub()
    HuggingFaceSource(hub=hub).acquire("nvidia/m", "sha1", str(tmp_path), None)
    assert "allow_patterns" not in hub.calls[0]


def test_the_coordinator_omits_an_empty_selection_on_the_wire() -> None:
    """The agent's request model forbids unknown keys.

    An agent predating file selection would refuse a request carrying the field
    even to say "all files", so a whole-repository acquire must send the body it
    always sent. This is why the contract bump to 1.14 is MINOR.
    """
    from datetime import datetime

    from tensorstead.coordinator.node_http import NodeHTTPClient
    from tensorstead.domain.models import Node

    node = Node(
        id="01M0000000000000000000000A",
        name="n1",
        agent_endpoint="https://n1:8443",
        agent_contract_version="1.14",
        agent_cert_fingerprint="aa",
        platform_facts={},
        registered_at=datetime.now().astimezone(),
    )
    sent: dict[str, Any] = {}

    class _Client(NodeHTTPClient):
        def __init__(self) -> None:
            pass

        def _call(self, node: Node, method: str, path: str, **kw: Any) -> dict[str, Any]:
            sent.update(kw["json"])
            return {}

    _Client().acquire_model(
        node, source_id="hf", source_model_id="m", revision="sha1", credential=None
    )
    assert "file_selector" not in sent

    sent.clear()
    _Client().acquire_model(
        node,
        source_id="hf",
        source_model_id="m",
        revision="sha1",
        credential=None,
        file_selector=("q8.gguf",),
    )
    assert sent["file_selector"] == ["q8.gguf"]


# -------------------------------------------------------------------- loading
def _gguf(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"GGUF\x00")
    return path


def test_a_lone_gguf_is_resolved_to_the_file(tmp_path: Path) -> None:
    """``--model`` named the directory; llama-server cannot load a directory."""
    _gguf(tmp_path, "model-UD-Q8_K_XL.gguf")
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[:2] == ["--model", str(tmp_path / "model-UD-Q8_K_XL.gguf")]


def test_many_quantizations_are_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A GGUF repo ships twenty of these; picking one for the operator is the
    silent mismatch this product exists to prevent."""
    for name in ("m-UD-Q8_K_XL.gguf", "m-UD-Q4_K_M.gguf", "m-UD-IQ2_M.gguf"):
        _gguf(tmp_path, name)
    with pytest.raises(ValueError, match="model_file"):
        LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))


def test_model_file_names_the_one_this_deployment_serves(tmp_path: Path) -> None:
    for name in ("m-UD-Q8_K_XL.gguf", "m-UD-Q4_K_M.gguf"):
        _gguf(tmp_path, name)
    args = LlamaCppAdapter().build_launch_args(
        {"model_file": "m-UD-Q8_K_XL.gguf"}, model_path=str(tmp_path)
    )
    assert args[:2] == ["--model", str(tmp_path / "m-UD-Q8_K_XL.gguf")]


def test_a_sharded_model_resolves_to_its_first_shard(tmp_path: Path) -> None:
    """One model split across files is one model, not an ambiguous choice."""
    _gguf(tmp_path, "m-00002-of-00002.gguf")
    _gguf(tmp_path, "m-00001-of-00002.gguf")
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[:2] == ["--model", str(tmp_path / "m-00001-of-00002.gguf")]


def test_a_selected_gguf_in_a_subdirectory_is_resolved_to_the_file(tmp_path: Path) -> None:
    """The other half of a nested selection: verified, then actually launchable.

    ``huggingface.verify`` was taught that a selection such as ``UD-IQ1_S/*.gguf``
    keeps the repository's directory below the model root. The adapter was not,
    so the tree downloaded, verified, and then failed at launch with "model
    directory has no .gguf file" -- the failure moved rather than went away.
    """
    selected = tmp_path / "UD-IQ1_S"
    selected.mkdir()
    _gguf(selected, "m-00001-of-00002.gguf")
    _gguf(selected, "m-00002-of-00002.gguf")

    LlamaCppAdapter().validate_model_path(tmp_path)  # does not raise
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[:2] == ["--model", str(selected / "m-00001-of-00002.gguf")]


def test_two_quantization_directories_are_refused_rather_than_guessed(tmp_path: Path) -> None:
    """The hazard the recursion introduces, and the reason grouping keeps the parent.

    Two quantizations of the same weights routinely carry byte-identical
    filenames. Grouping shard sets by stem alone would see one stem across both
    directories, conclude "one sharded model", and silently serve whichever
    sorts first -- picking a quantization for the operator, which is exactly
    what ``model_file`` exists to prevent.
    """
    for quant, shards in (("UD-IQ1_S", 3), ("UD-Q8_K_XL", 2)):
        directory = tmp_path / quant
        directory.mkdir()
        for shard in range(1, shards + 1):
            _gguf(directory, f"model-{shard:05d}-of-{shards:05d}.gguf")

    with pytest.raises(ValueError, match="model_file") as raised:
        LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    # Named relative to the model root, which is the form model_file takes.
    assert "UD-IQ1_S/model-00001-of-00003.gguf" in str(raised.value)
    assert "UD-Q8_K_XL/model-00001-of-00002.gguf" in str(raised.value)


def test_model_file_may_name_a_path_inside_a_subdirectory(tmp_path: Path) -> None:
    """The operator's way out of the ambiguity above."""
    for quant in ("UD-IQ1_S", "UD-Q8_K_XL"):
        directory = tmp_path / quant
        directory.mkdir()
        _gguf(directory, "model.gguf")

    args = LlamaCppAdapter().build_launch_args(
        {"model_file": "UD-Q8_K_XL/model.gguf"}, model_path=str(tmp_path)
    )
    assert args[:2] == ["--model", str(tmp_path / "UD-Q8_K_XL" / "model.gguf")]


def test_a_gguf_less_directory_is_still_refused(tmp_path: Path) -> None:
    """Recursion must not turn "no weights here" into a silent pass."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "README.md").write_text("not weights")

    with pytest.raises(ValueError, match=r"no \.gguf file"):
        LlamaCppAdapter().validate_model_path(tmp_path)


def test_a_projector_is_resolved_against_the_same_directory(tmp_path: Path) -> None:
    """Vision is the one capability the GGUF build has that the NVFP4 pair does
    not, and it needs a second file the operator should not hand-path."""
    _gguf(tmp_path, "m-UD-Q8_K_XL.gguf")
    _gguf(tmp_path, "mmproj-BF16.gguf")
    args = LlamaCppAdapter().build_launch_args(
        {"model_file": "m-UD-Q8_K_XL.gguf", "mmproj_file": "mmproj-BF16.gguf"},
        model_path=str(tmp_path),
    )
    assert "--mmproj" in args
    assert args[args.index("--mmproj") + 1] == str(tmp_path / "mmproj-BF16.gguf")


def test_it_binds_all_interfaces_inside_the_container(tmp_path: Path) -> None:
    """llama-server defaults to loopback; the port map then reaches nothing.

    The container starts, logs "model loaded", reports healthy, and serves no
    one -- the mirror of a runtime that bound a port nobody
    published. vLLM defaults to 0.0.0.0, which is why this never surfaced until
    llama.cpp ran here for the first time.
    """
    _gguf(tmp_path, "m.gguf")
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[args.index("--host") + 1] == "0.0.0.0"


def test_the_container_interface_is_not_operator_configurable(tmp_path: Path) -> None:
    """An operator who could set it could only break the port mapping."""
    with pytest.raises(ValueError):
        LlamaCppAdapter().validate_config({"extra_args": {"host": "127.0.0.1"}})


def test_flash_attention_is_emitted_with_a_value(tmp_path: Path) -> None:
    """A bare --flash-attn swallows the next argument.

    Current llama.cpp spells it ``--flash-attn on|off|auto``. Emitted bare it
    consumed the following flag and the server exited with "unknown value for
    --flash-attn: '--jinja'" -- an error naming neither the broken option nor
    the one that vanished.
    """
    _gguf(tmp_path, "m.gguf")
    on = LlamaCppAdapter().build_launch_args({"flash_attn": True}, model_path=str(tmp_path))
    assert on[on.index("--flash-attn") + 1] == "on"
    off = LlamaCppAdapter().build_launch_args({"flash_attn": False}, model_path=str(tmp_path))
    assert off[off.index("--flash-attn") + 1] == "off"
    # Unset means llama.cpp's own default ("auto"), which is not the same as off.
    assert "--flash-attn" not in LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))


def test_the_recipe_flags_pass_through_extra_args(tmp_path: Path) -> None:
    """The published recipe's speculative and sampling flags are not modelled
    here and do not need to be — extra_args forwards them verbatim."""
    _gguf(tmp_path, "m.gguf")
    args = LlamaCppAdapter().build_launch_args(
        {
            "ctx_size": 262144,
            "extra_args": {"spec-type": "draft-mtp", "spec-draft-n-max": 6, "temperature": 0.6},
        },
        model_path=str(tmp_path),
    )
    spec = args.index("--spec-type")
    assert args[spec : spec + 2] == ["--spec-type", "draft-mtp"]
    assert "--ctx-size" in args and "262144" in args


def test_the_acquired_three_shard_selection_resolves_to_its_first_shard(tmp_path: Path) -> None:
    """The tree this estate actually acquired, in the layout it actually has.

    ``unsloth/Qwen3.8-Flash-Next-GGUF`` at ``UD-IQ1_S/*.gguf`` stages three
    shards below the model root. llama.cpp is handed shard one and opens the
    other two by name, so the nested path -- not the root, and not the
    directory -- is what must reach ``--model``.
    """
    selected = tmp_path / "UD-IQ1_S"
    selected.mkdir()
    for shard in range(1, 4):
        _gguf(selected, f"Qwen3.8-Flash-Next-UD-IQ1_S-{shard:05d}-of-00003.gguf")

    LlamaCppAdapter().validate_model_path(tmp_path)  # does not raise
    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[:2] == [
        "--model",
        str(selected / "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"),
    ]


def test_an_empty_nested_gguf_is_not_weights(tmp_path: Path) -> None:
    """A zero-byte .gguf is an interrupted download, not a model.

    ``huggingface.verify`` refuses one at acquisition, but validation is the
    separate defence that covers a tree which arrived any other way -- and
    counting one as a candidate is worse than missing it: it would be resolved
    to and handed to llama-server, which fails with no classified cause.
    """
    selected = tmp_path / "UD-IQ1_S"
    selected.mkdir()
    (selected / "model.gguf").write_bytes(b"")

    with pytest.raises(ValueError, match=r"no \.gguf file"):
        LlamaCppAdapter().validate_model_path(tmp_path)


def test_an_empty_gguf_does_not_make_a_lone_model_ambiguous(tmp_path: Path) -> None:
    """The same rule on the resolve side, which is where it changes an answer.

    A truncated leftover beside real weights would otherwise turn a directory
    holding one model into a refusal naming two files, one of which is empty.
    """
    _gguf(tmp_path, "model-UD-Q8_K_XL.gguf")
    (tmp_path / "model-UD-Q4_K_M.gguf").write_bytes(b"")

    args = LlamaCppAdapter().build_launch_args({}, model_path=str(tmp_path))
    assert args[:2] == ["--model", str(tmp_path / "model-UD-Q8_K_XL.gguf")]
