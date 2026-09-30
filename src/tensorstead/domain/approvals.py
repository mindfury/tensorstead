"""Approval for a runtime option that loads code — the second decider.

The authorization policy refuses ``trust_remote_code`` with a sentence that
names the rule rather than the flag:

    it loads code into the runtime; what executes on an appliance is not
    something a deployment record may decide **on its own**

Those last four words are the whole design. The refusal was never "this must
never run"; it was "a deployment record is not sufficient authority for it."
NVIDIA's published guidance for `Qwen3.8-Flash-Next-NVFP4` requires the flag, so
a product that can only refuse forces the operator around the management plane
to a hand-started container — the drift this product exists to prevent, arrived
at by way of a safety feature.

So this is the other authority: a separate, immutable record, created through
its own route, that names **one exact tuple** and authorizes one option for it.

## What is in the tuple, and why each part

``option``            one classified option, never a wildcard
``runtime_type``      vLLM's ``--trust-remote-code`` and SGLang's are different
                      reviews of different code paths
``model_source_id``   ``huggingface`` is not ``local``
``source_model_id``   the repository whose code was read
``model_revision``    an **immutable** revision. The reviewer read the code at
                      one commit; a moving tag would let the bytes change under
                      an approval that still looks valid, which is this
                      repository's defining failure mode with a signature on it
``image_digest``      the runtime that will execute that code. A different vLLM
                      build is a different execution environment

Change any one and the fingerprint changes and the approval no longer matches.
That is the whole of acceptance criterion "cannot retain authorization if its
model revision, image digest, or approved runtime identity changes" — not a
revocation mechanism, which would need someone to notice, but a key that simply
stops fitting.

## Only ``loads-code`` is approvable

Deliberately the narrowest possible opening. The other refused classes stay
refused with no approval path at all, because an approval cannot make them safe:

- ``carries-a-credential`` — the objection is that argv is host-visible. An
  approval does not make a secret in argv invisible.
- ``redirects-what-is-served`` / ``product-owned`` — the objection is that the
  record and the argv would disagree. An approval that permitted it would be
  authorizing the product to lie in its own records.
- ``moves-the-endpoint`` — the deployment publishes the endpoint it declares; a
  runtime listening elsewhere is unreachable no matter who signed for it.

``loads-code`` is different in kind: the objection is *insufficient authority*,
and authority is exactly what an approval supplies.

## Where each half is enforced

The coordinator does not know the image digest — it records ``image_digest=""``
and the agent resolves the real one when it materializes the container. So the
binding is checked in two places, and neither is redundant:

- **coordinator**, at create/modify: the model half (source, model, revision,
  and that the revision is pinned), so an unauthorized record cannot be written
  at all;
- **agent**, at materialization: the whole tuple, including the digest it
  resolved itself and the revision from its *own* replica marker rather than
  from the payload it was handed.

The agent re-deriving both facts locally is what makes this more than an
assertion travelling in a request body. An approval for one model cannot be
replayed against another, because the agent computes the fingerprint from what
is actually on its disk and compares.

**The honest limit:** a caller holding the agent's management token can still
construct any deployment it likes, this one included. That token is already
total authority over the node — it can name any image and any model path — so
this adds no new exposure. What the two-sided check buys is that a *mismatched*
or *stale* approval is refused rather than honoured.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from tensorstead.domain.identity import assert_valid_ulid

# The one approvable effect class. Kept as a literal rather than imported from
# the adapters package because the domain does not depend on adapters; the
# policy module asserts the two agree, so a rename cannot silently split them.
APPROVABLE_EFFECT: Final = "loads-code"

# A revision that identifies content for all time. Approvals bind to one, and
# a tag like ``main`` is refused: the reviewer read the code at a commit, and an
# approval that survives the bytes changing underneath it is not an approval.
_IMMUTABLE_REVISION: Final = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")
_DIGEST: Final = re.compile(r"^sha256:[0-9a-f]{64}$")


def normalize_option(option: str) -> str:
    """One spelling of an option name, matching what the policy map is keyed by."""
    return option.strip().lower().replace("-", "_")


def fingerprint(
    *,
    option: str,
    runtime_type: str,
    model_source_id: str,
    source_model_id: str,
    model_revision: str,
    image_digest: str,
) -> str:
    """The tuple's content address, computed identically on both sides.

    Canonical JSON with sorted keys, so the coordinator and the agent cannot
    disagree because of dict ordering. Returned with a ``sha256:`` prefix so a
    fingerprint is never mistaken for one of the digests that compose it.
    """
    payload = json.dumps(
        {
            "option": normalize_option(option),
            "runtime_type": runtime_type.strip().lower(),
            "model_source_id": model_source_id.strip(),
            "source_model_id": source_model_id.strip(),
            "model_revision": model_revision.strip().lower(),
            "image_digest": image_digest.strip().lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CodeExecutionApproval:
    """One reviewed authorization for one option on one exact tuple.

    Immutable by construction and by storage: there is no update route. A
    changed mind is a delete plus a create, which leaves both events in the
    record instead of silently rewriting what was approved.
    """

    id: str
    option: str
    runtime_type: str
    model_source_id: str
    source_model_id: str
    model_revision: str
    image_digest: str
    # Why this was approved, and by whom. Required, and never defaulted: an
    # approval with no stated reason is indistinguishable later from one nobody
    # thought about. This is the only field a human must actually write.
    reason: str
    approved_by: str
    # The authorization policy in force when the review happened. A
    # classification can change; a decision recorded under the old one must be
    # legible as such rather than silently inheriting new rules.
    policy_version: str
    created_at: datetime

    def __post_init__(self) -> None:
        assert_valid_ulid(self.id, what="approval id")
        for name in (
            "option",
            "runtime_type",
            "model_source_id",
            "source_model_id",
            "reason",
            "approved_by",
            "policy_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required on an approval")
        if not _IMMUTABLE_REVISION.match(self.model_revision.strip().lower()):
            raise ValueError(
                f"model_revision {self.model_revision!r} is not an immutable revision: an "
                f"approval binds to a 40- or 64-character hex commit, never a tag. A tag can "
                f"move, and an approval that outlives the bytes it was granted for is not one"
            )
        if not _DIGEST.match(self.image_digest.strip().lower()):
            raise ValueError(
                f"image_digest {self.image_digest!r} is not a sha256 digest; an approval names "
                f"the exact runtime that will execute the code, not a mutable tag"
            )

    @property
    def fingerprint(self) -> str:
        """This approval's content address — the key a deployment must match."""
        return fingerprint(
            option=self.option,
            runtime_type=self.runtime_type,
            model_source_id=self.model_source_id,
            source_model_id=self.source_model_id,
            model_revision=self.model_revision,
            image_digest=self.image_digest,
        )

    def as_grant(self) -> dict[str, Any]:
        """The travelling form: what the coordinator hands the agent.

        Carries the tuple in full rather than only the fingerprint, so the agent
        can say *which* field failed to match rather than only that something
        did. A fingerprint alone would make a stale approval and a wrong model
        produce the same unhelpful refusal.
        """
        return {
            "approval_id": self.id,
            "option": normalize_option(self.option),
            "runtime_type": self.runtime_type,
            "model_source_id": self.model_source_id,
            "source_model_id": self.source_model_id,
            "model_revision": self.model_revision,
            "image_digest": self.image_digest,
            "fingerprint": self.fingerprint,
            "policy_version": self.policy_version,
        }
