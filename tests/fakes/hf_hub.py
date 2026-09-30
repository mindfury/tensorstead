"""Fake ``huggingface_hub``.

Models the model-source retrieval the real ``huggingface_hub`` provides,
so acquisition tests need no network and no Hugging Face
token (tier 1). It delegates retrieval and returns the revision the source
resolved or an explicit unpinned marker.

The fake models the behaviours the product depends on: a revision resolves, a
gated model is refused without a credential, and progress is reported
as it downloads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

_DEFAULT_RESOLVED_REVISION = "e1f2a3b4c5d6e7f8090a1b2c3d4e5f60718293ab"

# Progress callback signature: (fraction, message) — see ports/model_source.py.
ProgressFn = Callable[[float, str], None]


class FakeHuggingFaceHub:
    """A stateful in-memory stand-in for the Hugging Face Hub client.

    ``gated_repos`` are refused with ``AuthorizationRefused`` until a credential
    is supplied, modelling the gated-repo refusal without a live HF token.
    """

    def __init__(self) -> None:
        self.gated_repos: set[str] = set()
        # Repos whose *access terms* are unaccepted. A valid token does
        # not clear these — only the operator can, with the provider.
        self.terms_repos: set[str] = set()
        # Repos that model an interrupted download that left a
        # tree *missing* config.json. ``snapshot_download`` writes no config.json
        # for these, so the structural verify must refuse to promote them. The
        # default (a complete tree) is what every other test gets.
        self.incomplete_repos: set[str] = set()
        self.resolved_revisions: dict[str, str] = {}
        # What each snapshot_download was asked to filter by, so a test can
        # assert the selection reached the hub rather than only that it was
        # accepted.
        self.allow_patterns_seen: list[list[str] | None] = []

    def set_revision(self, repo_id: str, revision: str) -> None:
        self.resolved_revisions[repo_id] = revision

    def resolve_revision(self, repo_id: str, revision: str | None) -> str | None:
        """Resolve ``revision`` (or the source default) to a concrete revision.

        Returns ``None`` when the revision cannot be pinned, so the
        caller records ``revision_pinned = False`` rather than fabricating one.
        """
        if revision and revision != "main":
            return revision
        return self.resolved_revisions.get(repo_id, _DEFAULT_RESOLVED_REVISION)

    def gate(self, repo_id: str) -> None:
        self.gated_repos.add(repo_id)

    def require_access_terms(self, repo_id: str) -> None:
        """Mark a repo whose access terms the operator has not accepted.

        Modelled separately from ``gate`` because the two are different
        refusals: a token fixes one and cannot fix the other. The product must
        report the second, never route around it.
        """
        self.gated_repos.add(repo_id)
        self.terms_repos.add(repo_id)

    def mark_incomplete(self, repo_id: str) -> None:
        """Model an interrupted download: the tree lands *without* config.json.

        The structural verify the source runs between ``acquire`` and
        ``promote`` must refuse to promote this tree, so an interrupted transfer
        can never be stamped ``available``. Every other repo gets a
        complete, verifiable tree.
        """
        self.incomplete_repos.add(repo_id)

    def snapshot_download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        local_dir: str | None = None,
        progress: ProgressFn | None = None,
        allow_patterns: list[str] | None = None,
    ) -> str:
        """Model ``huggingface_hub.snapshot_download``.

        ``allow_patterns`` is the hub's file filter. It is accepted here because
        the real adapter accepts it: this fake is the conformance partner of
        ``_RealHubAdapter``, and a fake with a *narrower* surface
        than the real thing hides exactly the defect it exists to prevent --
        which is what happened when file selection was added and only the real
        adapter was missed.

        Returns the resolved revision. Raises ``AuthorizationRefused`` for a
        gated repo without a token, and ``AccessTermsNotAccepted`` for a repo
        whose terms are unaccepted **whether or not a token was supplied** —
        that is the whole point.

        When ``local_dir`` is given, a complete repo writes the files a usable
        tree has (a ``config.json``), so the source's structural verify has a
        real tree to accept -- modelling a real download rather than an
        unrealistic empty dir. A repo in ``incomplete_repos`` writes nothing,
        which is the interrupted-download shape verify exists to reject.
        """
        if repo_id in self.terms_repos:
            raise AccessTermsNotAccepted(repo_id)
        if repo_id in self.gated_repos and not token:
            raise AuthorizationRefused(repo_id)
        resolved = self.resolve_revision(repo_id, revision)
        assert resolved is not None
        self.allow_patterns_seen.append(allow_patterns)
        if local_dir is not None and repo_id not in self.incomplete_repos:
            os.makedirs(local_dir, exist_ok=True)
            (Path(local_dir) / "config.json").write_text(json.dumps({"arch": "test"}))
        if progress is not None:
            progress(1.0, "downloaded")
        return resolved


class AuthorizationRefused(Exception):
    """Upstream refused authorization for a gated model."""

    status_code = 403
    access_terms_required = False

    def __init__(self, repo_id: str, message: str | None = None) -> None:
        super().__init__(message or f"authorization refused for {repo_id!r}: no valid credential")
        self.repo_id = repo_id


class AccessTermsNotAccepted(AuthorizationRefused):
    """The model's access terms are unaccepted.

    A distinct refusal from a missing credential: supplying a token does not
    clear it. Only the operator can, with the provider — which is exactly why
    the product reports it rather than trying to act on it.
    """

    access_terms_required = True

    def __init__(self, repo_id: str) -> None:
        super().__init__(
            repo_id,
            message=(
                f"access to {repo_id!r} requires accepting the model's terms and "
                f"conditions with the provider"
            ),
        )
