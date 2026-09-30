"""Authorization policy for learned runtime options.

A probe reports what a runtime *accepts*. This decides what this product will
accept *from an operator*, and the two are different questions. Conflating them
would make Tensorstead's safety surface a function of vLLM's release notes: every
new flag they ship would arrive pre-authorized, and the ones that redirect what
is served or carry a credential arrive by the same door as the ones that tune a
batch size.

The first probe slice conflated them anyway. `parse_schema` marked an option
authorized whenever it was absent from `_FORBIDDEN_EXTRA_ARGS` — a list of the
handful of argv names the product derives, not a classification of anything. A
direct exercise reported `--trust-remote-code` as `authorized=True` with no
reason attached. That is the inverse of the rule: absence
from a small name map is not a review.

**So the default here is refusal.** An option nobody has classified is visible
and unauthorized — visible because concealing it puts an operator back to
guessing why a documented flag does nothing, which is exactly how `--headless`
stayed invisible for days; unauthorized because the product cannot yet say what
it does.

## Effects, not flags

The design is emphatic that this must not become a second allowlist chasing vLLM's
surface — that is the failure the whole spec exists to end. The question asked
of an option is *what class of effect does it have*: does it redirect what is
served, carry a credential, load code, move the endpoint, or reach the network?
Those classes are stable across releases in a way that names are not.

The honest description of what is written below: the **classes** are the policy
and they are stable; the **mapping** from a name into a class is a reviewed
judgement, recorded here, that grows when someone reviews an option. What makes
that different from an allowlist is the direction of the default. An allowlist
that misses a flag permits it. This refuses it, and says which review is
missing.

## Versioned

`POLICY_VERSION` changes whenever a classification changes, and a recorded
authorization decision carries the version that made it. Otherwise a revision
validated last month cannot be distinguished from one validated under different
rules, and "this was checked" stops being a statement anyone can act on.
"""

from __future__ import annotations

from typing import Any, Final

# Bumped on any change to the classifications below, or to what authorizes
# them. Date-ordered rather than sequential so a stale decision is legible as
# stale on sight.
#
# 2026-09-05.1 added the approval path for ``loads-code``. No classification
# moved: every option sits in the class it sat in before, and every default
# answer is the same. What changed is that one class can now be opened by a
# reviewed approval naming an exact tuple, so a decision recorded under
# 2026-08-15.1 was made when no such opening existed.
POLICY_VERSION: Final = "2026-09-05.1"

# The effect classes. These are the policy; the map beneath is the reviewed
# application of it.
REDIRECTS_WHAT_IS_SERVED: Final = "redirects-what-is-served"
CARRIES_A_CREDENTIAL: Final = "carries-a-credential"
LOADS_CODE: Final = "loads-code"
MOVES_THE_ENDPOINT: Final = "moves-the-endpoint"
REACHES_THE_NETWORK: Final = "reaches-the-network"
PRODUCT_OWNED: Final = "product-owned"
# The one authorized class: an option that changes how the runtime executes
# what it was already told to serve, without touching any fact the product
# owns. Batch sizes, memory fractions, cache dtypes, parallelism.
TUNES_EXECUTION: Final = "tunes-execution"

# Every class except one is refused, and the reason names the class rather than
# the flag, so an operator learns the rule and not just the verdict.
_REFUSAL_BY_EFFECT: Final[dict[str, str]] = {
    REDIRECTS_WHAT_IS_SERVED: (
        "it redirects what is served; the model, tokenizer and config come from what "
        "the node acquired, and a deployment that could point elsewhere would make the "
        "recorded model id describe something that is not running"
    ),
    CARRIES_A_CREDENTIAL: (
        "it names credential or TLS material; secrets reach the runtime as container "
        "environment precisely so they never appear in host-visible argv"
    ),
    LOADS_CODE: (
        "it loads code into the runtime; what executes on an appliance is not "
        "something a deployment record may decide on its own"
    ),
    MOVES_THE_ENDPOINT: (
        "it moves the served endpoint, which the deployment declares and the container "
        "publishes; a runtime listening elsewhere is unreachable and the product would "
        "report the endpoint it declared rather than the one in use"
    ),
    REACHES_THE_NETWORK: (
        "it reaches an external resource at launch, which turns starting a deployment "
        "into an unrecorded outbound call from the node"
    ),
    PRODUCT_OWNED: (
        "the product derives this from the deployment, and a fact that is both derived "
        "and supplied is a fact the record and the argv can disagree about"
    ),
}

# Reviewed classifications, keyed by the option's ``dest``. Underscores, because
# that is what argparse reports; callers normalise before looking up.
#
# Grows by review, never by pattern. An entry here means someone decided what
# the option does; its absence means nobody has, which is a different statement
# from "it is dangerous" and is reported as such.
_CLASSIFIED: Final[dict[str, str]] = {
    # Redirect what is served.
    "model": REDIRECTS_WHAT_IS_SERVED,
    "tokenizer": REDIRECTS_WHAT_IS_SERVED,
    "hf_config_path": REDIRECTS_WHAT_IS_SERVED,
    "config": REDIRECTS_WHAT_IS_SERVED,
    "served_model_name": REDIRECTS_WHAT_IS_SERVED,
    "lora_modules": REDIRECTS_WHAT_IS_SERVED,
    "prompt_adapters": REDIRECTS_WHAT_IS_SERVED,
    "speculative_config": REDIRECTS_WHAT_IS_SERVED,
    "tokenizer_mode": REDIRECTS_WHAT_IS_SERVED,
    "download_dir": REDIRECTS_WHAT_IS_SERVED,
    "load_format": REDIRECTS_WHAT_IS_SERVED,
    # Credentials and TLS.
    "api_key": CARRIES_A_CREDENTIAL,
    "ssl_keyfile": CARRIES_A_CREDENTIAL,
    "ssl_certfile": CARRIES_A_CREDENTIAL,
    "ssl_ca_certs": CARRIES_A_CREDENTIAL,
    "ssl_cert_reqs": CARRIES_A_CREDENTIAL,
    "hf_token": CARRIES_A_CREDENTIAL,
    # Load code.
    "trust_remote_code": LOADS_CODE,
    "chat_template": LOADS_CODE,
    "tool_parser_plugin": LOADS_CODE,
    "logits_processors": LOADS_CODE,
    "worker_cls": LOADS_CODE,
    "worker_extension_cls": LOADS_CODE,
    # Move the endpoint.
    "host": MOVES_THE_ENDPOINT,
    "port": MOVES_THE_ENDPOINT,
    "uds": MOVES_THE_ENDPOINT,
    "root_path": MOVES_THE_ENDPOINT,
    "api_server_count": MOVES_THE_ENDPOINT,
    # Reach the network at launch.
    "allowed_origins": REACHES_THE_NETWORK,
    "middleware": REACHES_THE_NETWORK,
    # Derived by the product from the deployment's declared nodes.
    "nnodes": PRODUCT_OWNED,
    "node_rank": PRODUCT_OWNED,
    "master_addr": PRODUCT_OWNED,
    "master_port": PRODUCT_OWNED,
    "headless": PRODUCT_OWNED,
    "data_parallel_address": PRODUCT_OWNED,
    "data_parallel_rpc_port": PRODUCT_OWNED,
    # Tune execution. Reviewed as touching no fact the product owns: each
    # changes how the runtime executes what it was already told to serve.
    "tensor_parallel_size": TUNES_EXECUTION,
    "pipeline_parallel_size": TUNES_EXECUTION,
    "max_model_len": TUNES_EXECUTION,
    "max_num_seqs": TUNES_EXECUTION,
    "max_num_batched_tokens": TUNES_EXECUTION,
    "gpu_memory_utilization": TUNES_EXECUTION,
    "swap_space": TUNES_EXECUTION,
    "block_size": TUNES_EXECUTION,
    "kv_cache_dtype": TUNES_EXECUTION,
    "dtype": TUNES_EXECUTION,
    "quantization": TUNES_EXECUTION,
    "seed": TUNES_EXECUTION,
    "enforce_eager": TUNES_EXECUTION,
    "enable_prefix_caching": TUNES_EXECUTION,
    "enable_chunked_prefill": TUNES_EXECUTION,
    "cuda_graph_sizes": TUNES_EXECUTION,
    "max_logprobs": TUNES_EXECUTION,
    "disable_log_stats": TUNES_EXECUTION,
    "async_scheduling": TUNES_EXECUTION,
    "distributed_executor_backend": TUNES_EXECUTION,
}


# Context key marking config that has *already* passed ``validate_config``.
#
# The authorization gate belongs at the boundary where operator-supplied config
# enters -- ``validate_config`` -- and nowhere else. An adapter reading a field
# back out of config that is already on a revision is not an authorization
# question: that config only reached the revision by passing the gate, and
# re-asking there makes an approved deployment unreadable by its own adapter.
#
# Named rather than implicit so the invariant is stated where it is relied on:
# **only pass this for config that has already been through the gate.**
_GATE_PASSED: Final = "already_validated"


def already_validated(info: Any) -> bool:
    """Whether this validation is a re-parse of config that already passed the gate."""
    context = getattr(info, "context", None)
    return isinstance(context, dict) and bool(context.get(_GATE_PASSED))


def validated_context() -> dict[str, Any]:
    """The context an adapter uses to re-parse config it has already gated."""
    return {_GATE_PASSED: True}


def approved_from_context(info: Any) -> frozenset[str]:
    """Options a reviewed approval authorized, read from a validation context.

    Takes Pydantic's ``ValidationInfo`` structurally rather than by type, so
    this module stays free of a pydantic import and both adapters read the
    context through one implementation. Absent or malformed context yields the
    empty set, which is the refusal — this is the fail-closed edge, so it must
    never raise its way into looking like an approval.
    """
    context = getattr(info, "context", None)
    if not isinstance(context, dict):
        return frozenset()
    approved = context.get("approved_options")
    if not approved:
        return frozenset()
    return frozenset(normalize_dest(str(option)) for option in approved)


def normalize_dest(dest: str) -> str:
    """One spelling of an option name. Mirrors ``approvals.normalize_option``."""
    return dest.strip().lower().replace("-", "_")


def classify(dest: str) -> str | None:
    """The reviewed effect class for an option, or ``None`` if unclassified.

    ``None`` is the honest answer for most of vLLM's surface and stays that way
    until someone reviews it. It is not a synonym for dangerous.
    """
    return _CLASSIFIED.get(normalize_dest(dest))


# The one class an approval can open, and the reason it is the only one.
#
# ``loads-code`` is refused for *insufficient authority* -- "not something a
# deployment record may decide on its own" -- and authority is precisely what a
# reviewed approval supplies. Every other refusal is a statement that the thing
# itself is wrong, which no signature repairs: a credential in argv is still
# host-visible, a redirected model still makes the record describe something
# that is not running, and a moved endpoint is still unreachable. Widening this
# set would turn one narrow, reviewed opening into a general bypass.
APPROVABLE_EFFECTS: Final[frozenset[str]] = frozenset({LOADS_CODE})


def is_approvable(dest: str) -> bool:
    """Whether an approval could ever authorize this option (spec: approvals)."""
    return classify(dest) in APPROVABLE_EFFECTS


def authorize(dest: str, *, approved: frozenset[str] | None = None) -> tuple[bool, str | None]:
    """Whether this product accepts ``dest`` from an operator, and why not.

    Returns ``(authorized, reason)``. The reason is ``None`` exactly when the
    option is authorized.

    ``approved`` names options an out-of-band, reviewed approval has authorized
    for this exact deployment (``tensorstead.domain.approvals``). It defaults to
    empty, so every caller that does not deliberately supply one gets the
    unchanged refusal — the guard fails closed at each of the three places
    ``validate_config`` is reached, rather than depending on each of them
    remembering to ask.
    """
    effect = classify(dest)
    if effect is None:
        return False, (
            f"no reviewed classification exists for this option under authorization "
            f"policy {POLICY_VERSION}; it is reported because the runtime accepts it, "
            f"and refused because the product cannot yet say what it does"
        )
    if effect == TUNES_EXECUTION:
        return True, None
    if approved and normalize_dest(dest) in approved and effect in APPROVABLE_EFFECTS:
        return True, None
    reason = _REFUSAL_BY_EFFECT[effect]
    if effect in APPROVABLE_EFFECTS:
        # Naming the remedy, not just the rule. Without this an operator holding
        # NVIDIA's own documented command line is told only that the product
        # disagrees with it, which is how a safety feature turns into a
        # hand-started container the coordinator knows nothing about.
        reason = (
            f"{reason}. A reviewed approval naming the exact model source, immutable "
            f"revision, and image digest can authorize it: see `code_approval_create`"
        )
    return False, reason
