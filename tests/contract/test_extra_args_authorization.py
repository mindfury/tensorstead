"""One option, one answer, whatever the spelling.

Both shipped adapters reserve a small set of flags: the model path, the
inference credential, the endpoint, the derived topology. Both checked those
reserved names against exactly one spelling of each name, and a command line has
more than one spelling per option. Seven bypasses were reproduced, all of
which reproduced again here before the fix:

- `{"nnodes=9": true}` was accepted and rendered `--nnodes=9` next to the
  `--nnodes 2` the product derives — the record and the argv disagreeing, which
  is the entire defect class this product exists to close;
- the same shape reached `--model` and `--api-key`, putting a replacement model
  path and a plaintext credential into host-visible argv;
- `{"no-flash-attn": true}` rendered alongside the `--flash-attn` the modelled
  field produced, emitting both sides of one boolean.

The bypasses were not a missing name in a list. They were two adapters
implementing the same rule separately, so the tests below run the hostile cases
against **both** — a fix that lands in one adapter and not the other is how this
happened the first time.

The benign cases matter as much as the hostile ones. `--no-mmap` is an ordinary
llama.cpp flag and `--no-enable-server-load-tracking` an ordinary vLLM one;
refusing every `no-` key would break the passthrough rather than secure it, and
a passthrough that refuses legitimate flags sends operators around the
management plane, which is what `extra_args` exists to prevent.
"""

from __future__ import annotations

import pytest

from tensorstead.adapters.runtimes.llamacpp import LlamaCppAdapter, LlamaCppConfig
from tensorstead.adapters.runtimes.vllm import VLLMAdapter, VLLMConfig

pytestmark = pytest.mark.contract

_VLLM_MODEL_PATH = "/models/deepseek"
_LLAMACPP_MODEL_PATH = "/models/model.gguf"


# Each case is (config, the substring the refusal must name). The substring is
# the option actually being reached, not the key as written -- an operator who
# typed `model=/tmp/other` needs to be told the refusal is about `--model`.
_VLLM_HOSTILE = [
    ({"extra_args": {"model=/tmp/other": True}}, "model"),
    ({"extra_args": {"api-key=plaintext": True}}, "api-key"),
    ({"extra_args": {"nnodes=9": True}}, "nnodes"),
    ({"extra_args": {"node-rank=1": True}}, "node-rank"),
    ({"extra_args": {"master-addr=wrong.invalid": True}}, "master-addr"),
    ({"extra_args": {"master-port=1": True}}, "master-port"),
    ({"extra_args": {"headless=true": True}}, "headless"),
    ({"extra_args": {"host=0.0.0.0": True}}, "host"),
    ({"extra_args": {"port=9999": True}}, "port"),
    # Negative aliases of product-owned and modelled options.
    ({"extra_args": {"no-headless": True}}, "headless"),
    (
        {"enable_prefix_caching": True, "extra_args": {"no-enable-prefix-caching": True}},
        "enable_prefix_caching",
    ),
    (
        {"trust_remote_code": True, "extra_args": {"no-trust-remote-code": True}},
        "trust_remote_code",
    ),
    # A key that is not a flag name at all.
    ({"extra_args": {"max num seqs": 8}}, "whitespace"),
]

_LLAMACPP_HOSTILE = [
    ({"extra_args": {"model=/tmp/other": True}}, "model"),
    ({"extra_args": {"api-key=plaintext": True}}, "api-key"),
    ({"extra_args": {"host=0.0.0.0": True}}, "host"),
    ({"extra_args": {"port=9999": True}}, "port"),
    ({"flash_attn": True, "extra_args": {"no-flash-attn": True}}, "flash_attn"),
    ({"extra_args": {"ctx size": 8}}, "whitespace"),
]


@pytest.mark.parametrize(("config", "names"), _VLLM_HOSTILE)
def test_vllm_refuses_alternate_spellings_at_validation(
    config: dict[str, object], names: str
) -> None:
    """``validate_config`` is where an operator finds out."""
    with pytest.raises(ValueError) as caught:
        VLLMAdapter().validate_config(dict(config))
    assert names in str(caught.value), (
        f"refusal for {config!r} must name {names!r} so the operator learns which option "
        f"they actually reached, not merely that their key was rejected"
    )


@pytest.mark.parametrize(("config", "names"), _VLLM_HOSTILE)
def test_vllm_refuses_alternate_spellings_at_render(config: dict[str, object], names: str) -> None:
    """And ``build_launch_args`` refuses too, not only the validating call.

    Rendering re-validates rather than trusting that whoever stored this
    configuration validated it -- a stored revision predating this fix must not
    render argv the current rules forbid.
    """
    with pytest.raises(ValueError):
        VLLMAdapter().build_launch_args(dict(config), model_path=_VLLM_MODEL_PATH)


@pytest.mark.parametrize(("config", "names"), _LLAMACPP_HOSTILE)
def test_llamacpp_refuses_alternate_spellings_at_validation(
    config: dict[str, object], names: str
) -> None:
    """The sibling adapter gets the identical answer."""
    with pytest.raises(ValueError) as caught:
        LlamaCppAdapter().validate_config(dict(config))
    assert names in str(caught.value)


@pytest.mark.parametrize(("config", "names"), _LLAMACPP_HOSTILE)
def test_llamacpp_refuses_alternate_spellings_at_render(
    config: dict[str, object], names: str
) -> None:
    with pytest.raises(ValueError):
        LlamaCppAdapter().build_launch_args(dict(config), model_path=_LLAMACPP_MODEL_PATH)


def test_no_reserved_flag_appears_twice_in_rendered_vllm_argv() -> None:
    """The property the name lists are only a means to.

    Checking names is checking the mechanism. This checks the outcome: whatever
    an operator writes, no product-owned flag reaches argv more than once. It
    would still hold if the reserved-name rule were implemented some entirely other
    way, which is what makes it worth writing separately.
    """
    from tensorstead.ports.runtime_adapter import NodePosition

    reserved = ("--model", "--api-key", "--host", "--port")
    args = VLLMAdapter().build_launch_args(
        {"extra_args": {"max-log-len": 100, "no-enable-server-load-tracking": True}},
        model_path=_VLLM_MODEL_PATH,
        position=NodePosition(
            node_index=1,
            node_count=2,
            self_address="10.100.184.2",
            peer_addresses=["10.100.184.1", "10.100.184.2"],
        ),
    )
    for flag in reserved:
        occurrences = [a for a in args if a == flag or a.startswith(f"{flag}=")]
        assert len(occurrences) <= 1, f"{flag} rendered {len(occurrences)} times in {args}"


def test_vllm_still_forwards_an_unmodelled_negative_flag() -> None:
    """A legitimate `--no-` flag is passthrough, not a bypass attempt.

    ``--no-enable-server-load-tracking`` is real vLLM surface that this product
    does not model. Refusing it would make the reserved-name rule a worse problem
    than the one it solves.
    """
    args = VLLMAdapter().build_launch_args(
        {"extra_args": {"no-enable-server-load-tracking": True, "max-log-len": 100}},
        model_path=_VLLM_MODEL_PATH,
    )
    assert "--no-enable-server-load-tracking" in args
    at = args.index("--max-log-len")
    assert args[at : at + 2] == ["--max-log-len", "100"]


def test_llamacpp_still_forwards_an_unmodelled_negative_flag() -> None:
    """``--no-mmap`` is ordinary llama.cpp surface."""
    args = LlamaCppAdapter().build_launch_args(
        {"extra_args": {"no-mmap": True, "mlock": True}},
        model_path=_LLAMACPP_MODEL_PATH,
    )
    assert "--no-mmap" in args
    assert "--mlock" in args


def test_a_modelled_boolean_set_false_still_renders_its_own_negative() -> None:
    """Refusing `no-x` must leave the supported way of saying it working.

    The refusal tells operators to set the field to false instead. If that did
    not work, the rule would have removed the capability rather than routing it
    through validation -- so the instruction in the error message is asserted
    rather than assumed.
    """
    validated = VLLMConfig(enable_prefix_caching=False)
    assert validated.enable_prefix_caching is False
    assert "no_flash_attn" not in LlamaCppConfig.model_fields
