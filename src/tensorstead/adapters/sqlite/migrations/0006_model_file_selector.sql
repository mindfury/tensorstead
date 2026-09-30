-- A model may be a *subset* of the repository it came from.
--
-- Multi-file repositories are the norm for GGUF publishers: one repo carries
-- twenty-odd quantizations of the same weights, and an operator wants exactly
-- one of them. Acquiring the whole repo is not an option (hundreds of GB for a
-- ~39 GB model), so acquisition has to be able to select.
--
-- The selection has to live *here*, on the model's identity, rather than being
-- a parameter of the acquire request. Two reasons, both about the record
-- matching reality:
--
--   1. UNIQUE (source_id, source_model_id, resolved_revision) says one repo at
--      one revision is one model. Two quants of the same repo at the same
--      revision collide on it -- the second acquisition either fails, or
--      silently reuses the first's record while different bytes sit on disk.
--   2. A record naming `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` when the node holds
--      one of its twenty-three files describes something that does not exist.
--      That is this estate's defining failure mode, and the acquire path is
--      where it would be introduced.
--
-- Stored as a canonical string (sorted, deduped, newline-joined) rather than
-- JSON so the UNIQUE constraint compares selections rather than spellings.
-- NOT NULL DEFAULT '' because SQLite treats NULLs as distinct in a UNIQUE
-- index: a nullable column would stop deduplicating whole-repo acquisitions,
-- which is the behaviour every existing model relies on.
--
-- SQLite cannot add a column to a UNIQUE constraint, so this is the documented
-- table rebuild. Foreign keys are disabled around it because model_replicas
-- references models(id); the pragma is a no-op inside a transaction, and the
-- runner's executescript commits before running, so it takes effect here.

PRAGMA foreign_keys=OFF;

CREATE TABLE models_rebuilt (
    id                TEXT PRIMARY KEY,                 -- ULID
    source_id         TEXT NOT NULL REFERENCES model_sources(id),
    source_model_id   TEXT NOT NULL,                    -- e.g. HF repo id
    resolved_revision TEXT,                             -- null => explicitly unpinned
    revision_pinned   INTEGER NOT NULL,
    size_bytes        INTEGER,
    content_digest    TEXT,
    file_selector     TEXT NOT NULL DEFAULT '',         -- '' => the whole repository
    UNIQUE (source_id, source_model_id, resolved_revision, file_selector)
);

-- Every existing model was acquired whole, so '' is the truthful backfill.
INSERT INTO models_rebuilt (
    id, source_id, source_model_id, resolved_revision,
    revision_pinned, size_bytes, content_digest, file_selector
)
SELECT id, source_id, source_model_id, resolved_revision,
       revision_pinned, size_bytes, content_digest, ''
  FROM models;

DROP TABLE models;
ALTER TABLE models_rebuilt RENAME TO models;

PRAGMA foreign_keys=ON;
