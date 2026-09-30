"""An image is asked what it accepts, rather than remembered.

`VLLMConfig` models twenty-five vLLM flags in a hand-maintained schema — a
private copy of something vLLM publishes, going stale on their release schedule
rather than ours. The estate makes it worse: the DSpark image is a community
fork built on the nodes, and it accepts values no released vLLM has.

Established by running it by hand on the appliance before writing any of
this:

- `make_arg_parser` builds the *same* parser `vllm serve` builds, so the CLI is
  a projection of a structure we can read directly;
- that structure knew things no upstream documentation does — `spec_method`
  includes `dspark`, `kv_cache_dtype` includes `nvfp4_ds_mla`, and `--headless`
  exists, which turned out to be the entire cause of every failed TP=2 start;
- the probe **needs a device**, because building the parser constructs config
  defaults that require device detection. That was not obvious and cost an
  attempt;
- it must run **on the node**. The controller is x86 and these images are
  arm64; an emulated probe reported a platform failure that had nothing to do
  with the image.

The first version of this slice shipped four boundaries open, all found by
review of the source rather than by these tests. Each has a case below, because
the shape of the mistake is what makes them worth keeping:

- it **blessed what it had not reviewed**, marking every unrecognised option
  authorized — including `--trust-remote-code`;
- it **attributed the answer to an image it had not run**, resolving a digest
  and then executing the mutable tag;
- it **accepted any trailing JSON**, so a payload that parsed but meant nothing
  became a *known* surface with zero options — unknown-as-empty, reintroduced
  through the back door;
- it **hid a failed cleanup**, so a probe could answer `known` while leaving a
  container on the node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter
from tensorstead.adapters.runtimes.option_policy import POLICY_VERSION
from tensorstead.adapters.runtimes.vllm import _SCHEMA_GRAMMAR, VLLMAdapter
from tensorstead.agent.app import build_agent_app
from tests.fakes.container_engine import FakeContainerEngine
from tests.fakes.service_manager import FakeServiceManager

pytestmark = pytest.mark.contract

_AUTH = {"Authorization": "Bearer mgmt"}
_IMAGE = "local/dspark-deepseek-v4-flash:0.1.1"
_IMAGE_ID = "sha256:b1c50db67ef0d26fd6ac2a0e839b45f1bac401192f9ed2920b0a6379fcb2fc66"


def _payload(options: list[dict[str, Any]]) -> str:
    return json.dumps({"schema": _SCHEMA_GRAMMAR, "runtime": "vllm", "options": options})


_OPTIONS: list[dict[str, Any]] = [
    {
        "name": "kv_cache_dtype",
        "flags": ["--kv-cache-dtype"],
        "kind": "str",
        "action": "_StoreAction",
        "nargs": None,
        "choices": ["auto", "fp8", "nvfp4_ds_mla", "turboquant_4bit_nc"],
        "default": {"state": "value", "value": "auto"},
        "help": "Data type for kv cache storage.",
    },
    {
        "name": "headless",
        "flags": ["--headless"],
        "kind": None,
        "action": "_StoreTrueAction",
        "nargs": None,
        "choices": [],
        "default": {"state": "value", "value": False},
        "help": "Run in headless mode.",
    },
    {
        "name": "model",
        "flags": ["--model"],
        "kind": "str",
        "action": "_StoreAction",
        "nargs": None,
        "choices": [],
        "default": {"state": "absent"},
        "help": "Name or path of the model.",
    },
    {
        "name": "trust_remote_code",
        "flags": ["--trust-remote-code"],
        "kind": None,
        "action": "_StoreTrueAction",
        "nargs": None,
        "choices": [],
        "default": {"state": "value", "value": False},
        "help": "Trust remote code from HuggingFace.",
    },
    {
        "name": "a_flag_nobody_has_reviewed",
        "flags": ["--a-flag-nobody-has-reviewed"],
        "kind": "str",
        "action": "_AppendAction",
        "nargs": "+",
        "choices": [],
        "default": {"state": "unrepresentable", "text": "<vllm.config.SomeObject object>"},
        "help": "Something a later vLLM shipped.",
    },
]

# What the appliance actually returned, in shape. The leading lines matter: vLLM
# writes startup notices to stdout before the payload, so a parser that assumed
# the whole stream was JSON would fail on every real image.
_REAL_OUTPUT = (
    "INFO 08-14 03:25:57 [importing.py:53] Triton is installed but 0 active driver(s)\n"
    "W0814 03:25:58.434000 1 torch/utils/cpp_extension.py:140] No CUDA runtime found\n"
) + _payload(_OPTIONS)


class _ProbingEngine(FakeContainerEngine):
    """Records how a probe was asked for, and answers with real output."""

    def __init__(self, output: str = _REAL_OUTPUT, fail: bool = False) -> None:
        super().__init__()
        self.output = output
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        # The node holds this image. A probe of an image the node does not have
        # is now a refusal, so a test that wants to probe has to say so.
        self.local_images[_IMAGE] = _IMAGE_ID

    def run_once(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.fail:
            from tensorstead.agent.container_engine.base import ImageBuildError

            raise ImageBuildError("Failed to infer device type", reference=kwargs["image"])
        return self.output


def _agent(engine: Any, tmp_path: Path) -> TestClient:
    return TestClient(
        build_agent_app(
            management_token="mgmt",
            container_engine=engine,
            service_manager=FakeServiceManager(),
            store_dir=tmp_path / "store",
            marker_dir=tmp_path / "markers",
        )
    )


def _probe(client: TestClient, runtime: str = "vllm") -> dict[str, Any]:
    resp = client.post(
        "/agent/v1/images:probe",
        json={"reference": _IMAGE, "runtime_type": runtime},
        headers=_AUTH,
    )
    assert resp.status_code == 200, resp.text
    return dict(resp.json())


def test_the_image_reports_its_own_option_surface(tmp_path: Path) -> None:
    """The point of the mechanism: values this build accepts, not remembered ones."""
    body = _probe(_agent(_ProbingEngine(), tmp_path))

    assert body["state"] == "known"
    kv = next(o for o in body["options"] if o["name"] == "kv_cache_dtype")
    assert "nvfp4_ds_mla" in kv["choices"], (
        "a fork-specific value the product could not have known was dropped"
    )
    assert kv["default"] == {"state": "value", "value": "auto", "text": None}


def test_the_flag_that_cost_a_week_is_discoverable(tmp_path: Path) -> None:
    """`--headless` was the whole cause, and no upstream documentation has it."""
    body = _probe(_agent(_ProbingEngine(), tmp_path))

    headless = next(o for o in body["options"] if "--headless" in o["flags"])
    # Discoverable, and marked as ours: which rank serves is derived from
    # position, so it is visible without being settable.
    assert headless["authorized"] is False
    assert "derived from position" in headless["unauthorized_reason"]


def test_product_owned_flags_are_marked_not_hidden(tmp_path: Path) -> None:
    """The runtime accepts it; this product does not accept it from you.

    Marked rather than dropped. Concealing an option the runtime genuinely has
    sends an operator back to guessing why a documented flag does nothing --
    and `--headless` is the case in point, invisible for days at real cost.
    """
    body = _probe(_agent(_ProbingEngine(), tmp_path))
    by_name = {o["name"]: o for o in body["options"]}

    assert "model" in by_name, "a product-owned option was hidden rather than marked"
    assert by_name["model"]["authorized"] is False
    assert "acquired model" in by_name["model"]["unauthorized_reason"]

    assert by_name["kv_cache_dtype"]["authorized"] is True
    assert by_name["kv_cache_dtype"]["unauthorized_reason"] is None


def test_discovery_does_not_authorize_what_nobody_reviewed(tmp_path: Path) -> None:
    """The first draft's inversion, now a test.

    `parse_schema` marked an option authorized whenever it was absent from the
    product-owned name map. That map is a handful of argv names, not a review,
    so every option vLLM ships that nobody has classified arrived permitted --
    and `--trust-remote-code` was the case that made it obvious.

    The default is now refusal, and the refusal names *why*: an effect class for
    something reviewed, an explicit "nobody has classified this" otherwise.
    Those are different statements and an operator needs to tell them apart.
    """
    body = _probe(_agent(_ProbingEngine(), tmp_path))
    by_name = {o["name"]: o for o in body["options"]}

    trust = by_name["trust_remote_code"]
    assert trust["authorized"] is False
    assert "loads code" in trust["unauthorized_reason"]

    unreviewed = by_name["a_flag_nobody_has_reviewed"]
    assert unreviewed["authorized"] is False
    assert "no reviewed classification" in unreviewed["unauthorized_reason"]
    # Visible, though. Refusing is not the same as hiding.
    assert unreviewed["flags"] == ["--a-flag-nobody-has-reviewed"]

    # Every decision carries the rules that made it, or it cannot be rechecked.
    assert {o["policy_version"] for o in body["options"]} == {POLICY_VERSION}


def test_the_recorded_shape_survives_being_written_down(tmp_path: Path) -> None:
    """Enough to validate against, not merely enough to display.

    A type name alone renders three different contracts identically -- one
    value, a repeatable list, and a bare switch. Recording the action, the
    collection shape and the boolean polarity is what makes a stored schema
    something a later configuration can actually be checked against.
    """
    body = _probe(_agent(_ProbingEngine(), tmp_path))
    by_name = {o["name"]: o for o in body["options"]}

    assert by_name["kv_cache_dtype"]["action"] == "store"
    assert by_name["kv_cache_dtype"]["repeatable"] is False
    assert by_name["kv_cache_dtype"]["polarity"] is None

    assert by_name["headless"]["action"] == "store_true"
    assert by_name["headless"]["polarity"] is True

    listy = by_name["a_flag_nobody_has_reviewed"]
    assert listy["action"] == "append"
    assert listy["repeatable"] is True
    assert listy["nargs"] == "+"


def test_an_unserialisable_default_stays_distinguishable(tmp_path: Path) -> None:
    """Three kinds of "no ordinary default" must not collapse into one.

    The first draft ran every non-null default through `str()`, so a default of
    `None`, the string `"None"`, and a Python object with no serialisable form
    all came out as text that reads the same. A schema that cannot tell them
    apart cannot validate a configuration against them.
    """
    body = _probe(_agent(_ProbingEngine(), tmp_path))
    by_name = {o["name"]: o for o in body["options"]}

    assert by_name["model"]["default"]["state"] == "absent"

    opaque = by_name["a_flag_nobody_has_reviewed"]["default"]
    assert opaque["state"] == "unrepresentable"
    assert "SomeObject" in opaque["text"]
    # The rendering is never offered as the value: a reader must not be able to
    # mistake `repr` output for something they can send back.
    assert opaque["value"] is None


def test_startup_noise_before_the_payload_is_ignored(tmp_path: Path) -> None:
    """Real images log before they answer; the payload is the tagged envelope."""
    body = _probe(_agent(_ProbingEngine(), tmp_path))

    assert body["state"] == "known"
    assert body["options"], "the payload was lost behind the runtime's own logging"


def test_a_failed_probe_is_unknown_and_never_empty(tmp_path: Path) -> None:
    """The defect here is rendering "we did not find out" as "it accepts nothing".

    The second is never true, and confusing the two is the defect this product
    exists to prevent.
    """
    body = _probe(_agent(_ProbingEngine(fail=True), tmp_path))

    assert body["state"] == "unknown"
    assert body["options"] == []
    assert "infer device type" in body["detail"], "the reason was discarded"


@pytest.mark.parametrize(
    ("output", "why"),
    [
        ('{"other": true}', "JSON that is not our schema"),
        ("not json at all", "no JSON whatsoever"),
        (json.dumps({"schema": _SCHEMA_GRAMMAR, "options": []}), "our schema, but empty"),
        (json.dumps({"schema": "someone.else/1", "options": [{"name": "x"}]}), "another grammar"),
        (_payload([{"flags": ["--x"]}]), "an option with no name"),
        (_payload([{"name": "x", "flags": []}]), "an option with no flags"),
        (_payload([{"name": "x", "flags": ["--x"]}]), "an option with no encoded default"),
    ],
)
def test_a_payload_that_is_not_our_schema_is_unknown(tmp_path: Path, output: str, why: str) -> None:
    """The back door into unknown-as-empty, closed.

    The first parser took the last line that happened to be JSON and read
    `options` off it with a default of `[]`. So `{"other": true}` produced zero
    options and the route reported the image's surface as **known and empty** --
    which is the precise inversion this contract forbids, arrived at through a
    payload that was merely valid JSON rather than through a failure anyone
    would notice.

    An empty option list is included deliberately: a vLLM parser with no options
    is not a thing that exists, so reading one means the probe ran somewhere it
    could not build a parser, and that is unknown rather than empty.
    """
    body = _probe(_agent(_ProbingEngine(output=output), tmp_path))

    assert body["state"] == "unknown", f"{why} was reported as a known surface"
    assert body["options"] == []
    assert body["detail"], "unknown without a reason is barely better than empty"


def test_the_answer_is_bound_to_the_image_that_gave_it(tmp_path: Path) -> None:
    """A tag is not an identity.

    The route resolved a digest and then executed the mutable reference, so a
    retag between the two would return image B's schema labelled with image A's
    id. Recording it by digest afterwards cannot repair an attribution that was
    already wrong -- so the resolved identity is what runs.
    """
    engine = _ProbingEngine()
    body = _probe(_agent(engine, tmp_path))

    assert body["image_id"] == _IMAGE_ID
    assert engine.calls[0]["image"] == _IMAGE_ID, (
        "the probe executed the mutable reference, so its answer describes whatever "
        "that tag pointed at by then rather than the image the response names"
    )


def test_an_image_the_node_does_not_hold_is_refused_before_running(tmp_path: Path) -> None:
    """Fail closed on an unresolvable image rather than answering about nothing."""
    engine = _ProbingEngine()
    engine.local_images.clear()
    body = _probe(_agent(engine, tmp_path))

    assert body["state"] == "unknown"
    assert body["image_id"] is None
    assert "resolves to no image" in body["detail"]
    assert engine.calls == [], "a container was started for an image that is not here"


def test_a_runtime_without_a_probe_is_unknown_not_broken(tmp_path: Path) -> None:
    """llama.cpp declares none, and that is an answer rather than a failure."""
    body = _probe(_agent(_ProbingEngine(), tmp_path), runtime="llamacpp")

    assert body["state"] == "unknown"
    assert "declares no schema probe" in body["detail"]


def test_the_probe_container_can_reach_nothing_and_keeps_nothing(tmp_path: Path) -> None:
    """A question about an image must not be a way to run one.

    No model, no network, no published port, and removed either way. Asserted
    because this is a new reason for the agent to start a container and the
    reason must stay narrow.

    This checks what the *route* asks for. What the engine actually does with
    the request is a separate question, asked at the engine seam in
    ``tests/unit/test_probe_execution_bounds.py`` -- asserting the arguments
    handed to a fake was the whole of the earlier claim, and it proved nothing
    about network denial, timeouts, output bounds, or cleanup.
    """
    engine = _ProbingEngine()
    _probe(_agent(engine, tmp_path))

    call = engine.calls[0]
    assert call["with_accelerator"] is True, "vLLM cannot build its parser without a device"
    assert "model_path" not in call and "endpoint" not in call
    assert call["script"], "the probe is supplied as a file, not as a shell argument"


def test_the_adapters_disagree_about_probing_on_purpose() -> None:
    """The second runtime is what keeps the seam honest, here as elsewhere."""
    assert VLLMAdapter().schema_probe() is not None
    assert LlamaCppAdapter().schema_probe() is None


def test_the_probe_emits_exactly_what_the_parser_reads() -> None:
    """The two halves of the grammar, checked against each other.

    Everything above feeds `parse_schema` a payload written by hand, which
    tests the reader against my idea of the writer. The writer is a string that
    runs inside a container built on a machine this suite never touches, so
    "my idea of it" is the whole risk: if the probe emitted `default` as a bare
    string, every test above would still pass and every real probe would fail.

    So this runs the **actual probe source**, with vLLM's parser swapped for an
    ordinary `argparse` one, and feeds its real stdout to the real reader. No
    vLLM needed — `make_arg_parser` returns an `ArgumentParser`, and the probe
    only ever touches `_actions`.
    """
    import argparse
    import contextlib
    import io

    from tensorstead.adapters.runtimes.vllm import _VLLM_SCHEMA_PROBE

    class _Opaque:
        def __repr__(self) -> str:
            return "<vllm.config.SomeObject object>"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "nvfp4_ds_mla"],
        default="auto",
        help="Data type for kv cache storage.",
    )
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    parser.add_argument("--model", type=str, default=None, help="Model path.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code.")
    parser.add_argument("--middleware", action="append", nargs="+", default=_Opaque())
    parser.add_argument("--max-num-seqs", type=int, default=256)

    # The probe's own source, with only its two vLLM imports and the parser
    # construction replaced. Everything that encodes an option is the shipped
    # code, character for character.
    source = _VLLM_SCHEMA_PROBE
    source = (
        source.replace("from vllm.entrypoints.openai.cli_args import make_arg_parser\n", "")
        .replace("from vllm.utils.argparse_utils import FlexibleArgumentParser\n", "")
        .replace("parser = make_arg_parser(FlexibleArgumentParser())", "")
    )
    assert "make_arg_parser" not in source, "the substitution missed; this would import vLLM"

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(source, "<probe>", "exec"), {"parser": parser})  # noqa: S102

    options = VLLMAdapter().parse_schema(captured.getvalue())
    by_name = {o.name: o for o in options}

    assert by_name["kv_cache_dtype"].choices == ["auto", "nvfp4_ds_mla"]
    assert by_name["kv_cache_dtype"].default.state == "value"
    assert by_name["kv_cache_dtype"].default.value == "auto"
    assert by_name["kv_cache_dtype"].kind == "str"

    assert by_name["headless"].action == "store_true"
    assert by_name["headless"].polarity is True
    assert by_name["headless"].authorized is False

    assert by_name["model"].default.state == "absent"
    assert by_name["trust_remote_code"].authorized is False

    # The case that only a real run produces: argparse holding a default no
    # JSON encoder will take.
    middleware = by_name["middleware"]
    assert middleware.default.state == "unrepresentable"
    assert "SomeObject" in (middleware.default.text or "")
    assert middleware.repeatable is True
    assert middleware.nargs == "+"

    assert by_name["max_num_seqs"].authorized is True, (
        "a reviewed tuning option should survive the round trip authorized"
    )
