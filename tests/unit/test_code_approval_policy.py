"""The narrow opening stays narrow.

``trust_remote_code`` is now approvable. These hold the boundary of that: which
options an approval can ever reach, what an approval must name to be one at all,
and that the fingerprint is a key rather than a label.

The class-level tests matter more than they look. The approval mechanism's
whole risk is that it becomes a general bypass one option at a time, and the
thing standing between here and there is that only ``loads-code`` is approvable.
A future reviewer adding an entry to ``_CLASSIFIED`` should have to make these
fail on purpose.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tensorstead.adapters.runtimes.option_policy import (
    APPROVABLE_EFFECTS,
    CARRIES_A_CREDENTIAL,
    LOADS_CODE,
    MOVES_THE_ENDPOINT,
    POLICY_VERSION,
    PRODUCT_OWNED,
    REDIRECTS_WHAT_IS_SERVED,
    authorize,
    is_approvable,
)
from tensorstead.domain.approvals import (
    APPROVABLE_EFFECT,
    CodeExecutionApproval,
    fingerprint,
    normalize_option,
)
from tensorstead.domain.identity import new_ulid

pytestmark = pytest.mark.unit

_REVISION = "fab0aecb760cec45227f6656abcaafa11abca87a"
_DIGEST = "sha256:02366b8f87b8490c137b49f91beb7046904b16c70b7af2408753854d94e69170"


def _approval(**overrides: object) -> CodeExecutionApproval:
    fields: dict = {
        "id": new_ulid(),
        "option": "trust_remote_code",
        "runtime_type": "vllm",
        "model_source_id": "huggingface",
        "source_model_id": "nvidia/Qwen3.8-Flash-Next-NVFP4",
        "model_revision": _REVISION,
        "image_digest": _DIGEST,
        "reason": "vendor guidance requires it",
        "approved_by": "operator",
        "policy_version": POLICY_VERSION,
        "created_at": datetime.now().astimezone(),
    }
    fields.update(overrides)
    return CodeExecutionApproval(**fields)


# ------------------------------------------------------------- the boundary


def test_only_loads_code_is_approvable() -> None:
    """One class, and the domain and the policy must name the same one."""
    assert {LOADS_CODE} == APPROVABLE_EFFECTS
    assert APPROVABLE_EFFECT == LOADS_CODE


@pytest.mark.parametrize(
    "option",
    [
        "api_key",
        "hf_token",
        "ssl_keyfile",
        "model",
        "served_model_name",
        "port",
        "host",
        "nnodes",
        "headless",
        "master_addr",
        "download_dir",
        "load_format",
    ],
)
def test_an_approval_can_never_reach_these(option: str) -> None:
    """Every other refusal says the thing itself is wrong, which no review repairs."""
    assert not is_approvable(option)
    approved_anyway = frozenset({option})
    authorized, reason = authorize(option, approved=approved_anyway)
    assert authorized is False, f"{option} was authorized by an approval"
    assert reason


@pytest.mark.parametrize(
    "effect", [CARRIES_A_CREDENTIAL, REDIRECTS_WHAT_IS_SERVED, MOVES_THE_ENDPOINT, PRODUCT_OWNED]
)
def test_no_other_effect_class_became_approvable(effect: str) -> None:
    assert effect not in APPROVABLE_EFFECTS


@pytest.mark.parametrize("option", ["trust_remote_code", "chat_template", "worker_cls"])
def test_loads_code_options_are_approvable_and_still_refused_by_default(option: str) -> None:
    """Approvable is not approved: the default answer is unchanged."""
    assert is_approvable(option)
    assert authorize(option)[0] is False
    assert authorize(option, approved=frozenset({option}))[0] is True


def test_an_unclassified_option_is_not_approvable() -> None:
    """Absence from the map means nobody reviewed it, which is not permission."""
    assert not is_approvable("some_flag_nobody_has_reviewed")
    assert (
        authorize(
            "some_flag_nobody_has_reviewed", approved=frozenset({"some_flag_nobody_has_reviewed"})
        )[0]
        is False
    )


def test_the_refusal_names_the_remedy_only_where_one_exists() -> None:
    """An operator should not be pointed at a route that cannot help them."""
    assert "code_approval_create" in (authorize("trust_remote_code")[1] or "")
    assert "code_approval_create" not in (authorize("api_key")[1] or "")


# ------------------------------------------------------- the key, not a label


def test_the_fingerprint_is_stable_across_spelling_and_case() -> None:
    """The coordinator and the agent must never disagree over a hyphen."""
    assert fingerprint(
        option="trust-remote-code",
        runtime_type="vLLM",
        model_source_id="huggingface",
        source_model_id="nvidia/M",
        model_revision=_REVISION.upper(),
        image_digest=_DIGEST.upper(),
    ) == fingerprint(
        option="trust_remote_code",
        runtime_type="vllm",
        model_source_id="huggingface",
        source_model_id="nvidia/M",
        model_revision=_REVISION,
        image_digest=_DIGEST,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("option", "chat_template"),
        ("runtime_type", "sglang"),
        ("model_source_id", "local"),
        ("source_model_id", "nvidia/Other"),
        ("model_revision", "a" * 40),
        ("image_digest", "sha256:" + "b" * 64),
    ],
)
def test_every_field_changes_the_fingerprint(field: str, value: str) -> None:
    """Nothing in the tuple is decorative; each one is part of the key."""
    base = dict(
        option="trust_remote_code",
        runtime_type="vllm",
        model_source_id="huggingface",
        source_model_id="nvidia/M",
        model_revision=_REVISION,
        image_digest=_DIGEST,
    )
    assert fingerprint(**base) != fingerprint(**{**base, field: value})


def test_normalize_option_matches_the_policy_spelling() -> None:
    assert normalize_option("Trust-Remote-Code") == "trust_remote_code"


# ---------------------------------------------------- what an approval must be


@pytest.mark.parametrize("revision", ["main", "v1.0", "", "fab0aec", "z" * 40])
def test_an_approval_must_name_an_immutable_revision(revision: str) -> None:
    """A tag can move; an approval that outlives its bytes is not one."""
    with pytest.raises(ValueError, match="immutable"):
        _approval(model_revision=revision)


@pytest.mark.parametrize("digest", ["latest", "sha256:short", "", "02366b8f"])
def test_an_approval_must_name_an_image_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="digest"):
        _approval(image_digest=digest)


@pytest.mark.parametrize("field", ["reason", "approved_by"])
def test_an_approval_without_a_reviewer_or_a_reason_is_refused(field: str) -> None:
    """An approval nobody signed is indistinguishable later from one nobody thought about."""
    with pytest.raises(ValueError, match=field):
        _approval(**{field: "   "})


def test_a_64_character_revision_is_accepted() -> None:
    """Not every source spells an immutable revision as a 40-char git sha."""
    assert _approval(model_revision="c" * 64).fingerprint.startswith("sha256:")


def test_the_grant_carries_the_whole_tuple_not_just_the_key() -> None:
    """So the agent can say which field disagreed, not merely that one did."""
    grant = _approval().as_grant()

    for field in (
        "option",
        "runtime_type",
        "model_source_id",
        "source_model_id",
        "model_revision",
        "image_digest",
        "fingerprint",
        "approval_id",
    ):
        assert grant[field], f"grant omitted {field}"
    assert grant["fingerprint"] == fingerprint(
        option=grant["option"],
        runtime_type=grant["runtime_type"],
        model_source_id=grant["model_source_id"],
        source_model_id=grant["source_model_id"],
        model_revision=grant["model_revision"],
        image_digest=grant["image_digest"],
    )


def test_a_grant_carries_no_secret() -> None:
    """It travels in a request body and may be logged by either end."""
    serialized = str(_approval().as_grant()).lower()
    for token in ("secret", "password", "token", "credential", "api_key"):
        assert token not in serialized
