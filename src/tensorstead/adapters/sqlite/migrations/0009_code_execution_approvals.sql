-- Reviewed approvals for runtime options that load code.
--
-- The authorization policy refuses `trust_remote_code` because "what executes
-- on an appliance is not something a deployment record may decide on its own".
-- This table is the other decider: a separate, reviewed record naming one exact
-- tuple, created through its own route rather than as part of a deployment
-- mutation. NVIDIA's published guidance for Qwen3.8-Flash-Next-NVFP4 requires
-- the flag, and a product that can only refuse sends the operator to a
-- hand-started container the coordinator knows nothing about.
--
-- The tuple is the primary key, not the id. Two approvals for the same tuple
-- are the same approval, and letting a second one exist would mean deleting one
-- leaves an authorization that still works -- a revocation that silently does
-- nothing. `INSERT` (never `INSERT OR REPLACE`) so re-approving an existing
-- tuple is a visible conflict rather than a silent overwrite of who approved
-- what and why.
--
-- No `updated_at`, deliberately: there is no update path. A changed mind is a
-- delete plus a create, which leaves both events in the record.
--
-- `fingerprint` is stored although it is derived from the five tuple columns.
-- It is what the agent compares against, and storing it means a mismatch
-- between the stored tuple and the stored fingerprint is detectable rather
-- than impossible-by-construction-until-it-isn't.
CREATE TABLE IF NOT EXISTS code_execution_approvals (
    id                TEXT PRIMARY KEY,
    option            TEXT NOT NULL,
    runtime_type      TEXT NOT NULL,
    model_source_id   TEXT NOT NULL,
    source_model_id   TEXT NOT NULL,
    model_revision    TEXT NOT NULL,
    image_digest      TEXT NOT NULL,
    fingerprint       TEXT NOT NULL,
    reason            TEXT NOT NULL,
    approved_by       TEXT NOT NULL,
    policy_version    TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    UNIQUE (option, runtime_type, model_source_id, source_model_id,
            model_revision, image_digest)
);

-- The lookup the deployment path actually performs is by fingerprint.
CREATE INDEX IF NOT EXISTS idx_code_execution_approvals_fingerprint
    ON code_execution_approvals (fingerprint);

-- Which approval authorized this revision, if any.
--
-- Recorded on the revision rather than looked up at read time, because the
-- revision is immutable and the approval is not: an approval can be deleted,
-- and a revision that then reported "authorized by nothing" would be
-- retroactively rewriting what was true when it was created. This column says
-- what authorized it *then*, which is what a history is for.
--
-- It is provenance, never permission. Nothing reads this column to decide
-- whether a deployment may start -- that decision is made fresh against the
-- approvals table at create and again at the agent. A revision carrying a
-- fingerprint whose approval has since been deleted will simply fail to start,
-- and the record will still say what it was authorized by.
--
-- DEFAULT '' so every revision that predates this column reads as
-- unauthorized, which is what they all were.
ALTER TABLE deployment_revisions
    ADD COLUMN code_approval_fingerprint TEXT NOT NULL DEFAULT '';
ALTER TABLE deployment_revisions
    ADD COLUMN code_approval_id TEXT NOT NULL DEFAULT '';
