"""Code-execution approvals: the authority a deployment record does not have.

See ``tensorstead.domain.approvals`` for the design and for why ``loads-code`` is
the only class an approval can open. This module is the coordinator's half: it
creates and lists approvals, and it answers the one question the deployment path
asks — *is this exact deployment authorized to load code, and by which
approval?*

The answer is recomputed from the approvals table at every create and every
modify. Nothing is cached on the deployment and nothing is inherited from the
previous revision. That is what makes deleting an approval a real revocation
rather than a note in a table, and it is why a revision whose model revision or
image digest changed cannot carry its predecessor's authorization forward: the
tuple it hashes to is simply a different tuple.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tensorstead.adapters.runtimes.option_policy import (
    POLICY_VERSION,
    is_approvable,
    normalize_dest,
)
from tensorstead.domain.approvals import CodeExecutionApproval, fingerprint
from tensorstead.domain.errors import InvalidDeploymentError, NotFoundError
from tensorstead.domain.identity import new_ulid
from tensorstead.domain.models import Model
from tensorstead.ports.repository import Repository


def requested_code_options(runtime_config: dict[str, Any]) -> frozenset[str]:
    """Approvable options this config declares, as modelled first-class fields.

    Only top-level modelled fields are considered, and that is deliberate.
    ``extra_args`` is **not** searched, because an approvable option arriving
    through the passthrough is not an approval question at all -- it is the
    bypass ``authorize_extra_arg`` already refuses by reserved name, and it must
    keep being refused there whether or not an approval exists. Treating it as
    approvable here would turn one reviewed opening into two doors, one of which
    nobody reviewed.
    """
    requested = set()
    for key, value in runtime_config.items():
        if key == "extra_args":
            continue
        if value and is_approvable(normalize_dest(str(key))):
            requested.add(normalize_dest(str(key)))
    return frozenset(requested)


class ApprovalService:
    """Create, list, and revoke reviewed code-execution approvals."""

    def __init__(self, repository: Repository) -> None:
        self._repo = repository

    def create(
        self,
        *,
        option: str,
        runtime_type: str,
        model_source_id: str,
        source_model_id: str,
        model_revision: str,
        image_digest: str,
        reason: str,
        approved_by: str,
    ) -> CodeExecutionApproval:
        """Record one approval. Alters no deployment and starts nothing.

        Refuses an option no approval could ever authorize, rather than storing
        a row that would never match anything: an approval for ``api_key`` that
        sits in the table looking effective is worse than a refusal, because it
        reads to a later operator as a permission that exists.
        """
        normalized = normalize_dest(option)
        if not is_approvable(normalized):
            raise InvalidDeploymentError(
                f"option {option!r} cannot be authorized by an approval. Only options "
                f"classified 'loads-code' can: every other refusal says the thing itself "
                f"is wrong -- a credential in argv stays host-visible, a redirected model "
                f"still makes the record describe something that is not running -- and no "
                f"review repairs that",
                detail={"option": normalized, "policy_version": POLICY_VERSION},
            )
        # The domain's own invariants -- immutable revision, real digest --
        # raise ``ValueError``. Classified here rather than allowed to escape:
        # as an unhandled error it reaches FastAPI as a bare "Internal Server
        # Error", which names no field and no remedy and is indistinguishable
        # from a sick coordinator, so the operator investigates the wrong half
        # of the system. This classification follows the same fix a recorded
        # finding called for.
        try:
            approval = self._build(
                normalized=normalized,
                runtime_type=runtime_type,
                model_source_id=model_source_id,
                source_model_id=source_model_id,
                model_revision=model_revision,
                image_digest=image_digest,
                reason=reason,
                approved_by=approved_by,
            )
        except ValueError as exc:
            raise InvalidDeploymentError(str(exc), detail={"option": normalized}) from exc
        existing = self._repo.find_code_approval(approval.fingerprint)
        if existing is not None:
            raise InvalidDeploymentError(
                f"this exact tuple is already approved as {existing.id} by "
                f"{existing.approved_by!r}; delete that approval to replace it, so the "
                f"record shows the replacement rather than hiding it",
                detail={"existing_approval_id": existing.id},
            )
        self._repo.save_code_approval(approval)
        return approval

    @staticmethod
    def _build(
        *,
        normalized: str,
        runtime_type: str,
        model_source_id: str,
        source_model_id: str,
        model_revision: str,
        image_digest: str,
        reason: str,
        approved_by: str,
    ) -> CodeExecutionApproval:
        return CodeExecutionApproval(
            id=new_ulid(),
            option=normalized,
            runtime_type=runtime_type.strip().lower(),
            model_source_id=model_source_id.strip(),
            source_model_id=source_model_id.strip(),
            model_revision=model_revision.strip().lower(),
            image_digest=image_digest.strip().lower(),
            reason=reason,
            approved_by=approved_by,
            policy_version=POLICY_VERSION,
            created_at=datetime.now().astimezone(),
        )

    def list(self) -> list[CodeExecutionApproval]:
        return self._repo.list_code_approvals()

    def get(self, approval_id: str) -> CodeExecutionApproval:
        approval = self._repo.get_code_approval(approval_id)
        if approval is None:
            raise NotFoundError(f"no code approval with id {approval_id!r}")
        return approval

    def delete(self, approval_id: str) -> None:
        """Revoke an approval. Running deployments are not disturbed.

        Deliberately: stopping inference is a bigger action than withdrawing an
        authorization, and the product does not reverse work on a host without
        being asked (this product's rule). What revocation guarantees is
        that the next create, modify, or start finds nothing to match -- which
        is the moment the decision is actually made.
        """
        self.get(approval_id)
        self._repo.delete_code_approval(approval_id)


def authorize_code_options(
    repository: Repository,
    *,
    runtime_type: str,
    runtime_config: dict[str, Any],
    model: Model,
    image_reference: str,
    participating_nodes: tuple[str, ...] | list[str],
) -> tuple[frozenset[str], CodeExecutionApproval | None]:
    """Which code-loading options this exact deployment may set, and by whose approval.

    Returns ``(approved_options, approval)``. An empty set with ``None`` means
    the config asked for nothing approvable, which is the ordinary case and
    costs one dict scan.

    Every refusal below names what to fix, because each one is a thing the
    operator can actually do something about: pin the model, acquire the image,
    or create the approval.
    """
    requested = requested_code_options(runtime_config)
    if not requested:
        return frozenset(), None

    # An approval binds to an immutable revision, so a deployment that cannot
    # state one cannot be matched to an approval at all. Refused here, with the
    # reason, rather than failing later as a mysterious fingerprint miss.
    if not model.revision_pinned or not model.resolved_revision:
        raise InvalidDeploymentError(
            f"a deployment that loads code must name a model pinned to an immutable "
            f"revision; model {model.source_model_id!r} is "
            f"{'unpinned' if not model.revision_pinned else 'missing a resolved revision'}. "
            f"An approval is granted for code read at one commit, and cannot follow a "
            f"revision that moves",
            detail={"options": sorted(requested), "model_id": model.id},
        )

    digest = _recorded_image_digest(repository, image_reference, participating_nodes)
    if digest is None:
        raise InvalidDeploymentError(
            f"a deployment that loads code must name an image already present on its "
            f"nodes, so the approval can bind to the exact runtime that will execute the "
            f"code; no digest is recorded for {image_reference!r} on the declared nodes. "
            f"Acquire or import the image first, then create the deployment",
            detail={"options": sorted(requested), "image_reference": image_reference},
        )

    approvals = []
    for option in sorted(requested):
        candidate = fingerprint(
            option=option,
            runtime_type=runtime_type,
            model_source_id=model.source_id,
            source_model_id=model.source_model_id,
            model_revision=model.resolved_revision,
            image_digest=digest,
        )
        approval = repository.find_code_approval(candidate)
        if approval is None:
            raise InvalidDeploymentError(
                f"{option} is not approved for this deployment. It loads code into the "
                f"runtime, which a deployment record may not decide on its own. A reviewed "
                f"approval for this exact tuple authorizes it:\n"
                f"  runtime_type    {runtime_type}\n"
                f"  model_source_id {model.source_id}\n"
                f"  source_model_id {model.source_model_id}\n"
                f"  model_revision  {model.resolved_revision}\n"
                f"  image_digest    {digest}\n"
                f"Create one with `code_approval_create`; it is refused for any other "
                f"model, revision, or image",
                detail={
                    "option": option,
                    "required_fingerprint": candidate,
                    "runtime_type": runtime_type,
                    "model_source_id": model.source_id,
                    "source_model_id": model.source_model_id,
                    "model_revision": model.resolved_revision,
                    "image_digest": digest,
                },
            )
        approvals.append(approval)

    # Several approvable options would mean several approvals, and a revision
    # records one. Refused rather than recording the first and implying it
    # covered both -- a record that names one authorization for two decisions
    # is the kind of near-truth this product exists to avoid. No shipped
    # deployment needs two; if one ever does, the revision grows a list.
    if len(approvals) > 1:
        raise InvalidDeploymentError(
            f"a deployment may declare one code-loading option per revision; this one "
            f"declares {len(approvals)}: {', '.join(sorted(requested))}",
            detail={"options": sorted(requested)},
        )
    return requested, approvals[0]


def _recorded_image_digest(
    repository: Repository,
    image_reference: str,
    participating_nodes: tuple[str, ...] | list[str],
) -> str | None:
    """The digest recorded for this reference on every declared node, if agreed.

    ``None`` when the image is absent from any declared node, or when the nodes
    disagree about what the reference resolves to. Disagreement is treated as
    absence on purpose: a tag pointing at different bytes on two nodes is
    exactly the case where authorizing "the image" would authorize two different
    runtimes, and this estate's own records show that mattering.
    """
    nodes = set(participating_nodes)
    digests = {
        record.digest
        for record in repository.list_images()
        if record.reference == image_reference and record.node_id in nodes
    }
    covered = {
        record.node_id
        for record in repository.list_images()
        if record.reference == image_reference and record.node_id in nodes
    }
    if len(digests) != 1 or covered != nodes:
        return None
    return digests.pop()
