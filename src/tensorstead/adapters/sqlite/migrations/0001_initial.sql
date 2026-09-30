-- 0001_initial.sql — create all persisted tables.
-- Hand-written SQL confined to the SQLite adapter. Declared state only;
-- observed state is never persisted as current.
--
-- Key uniqueness decisions:
--   - Model: (source_id, source_model_id, resolved_revision) is unique.
--   - ModelReplica: (model_id, node_id) is the primary key.
--   - DeploymentRevision is IMMUTABLE — there is intentionally no UPDATE path.

CREATE TABLE nodes (
    id                      TEXT PRIMARY KEY,           -- ULID
    name                    TEXT NOT NULL UNIQUE,
    agent_endpoint          TEXT NOT NULL,
    agent_contract_version  TEXT NOT NULL,
    agent_cert_fingerprint  TEXT NOT NULL,              -- pinned at registration
    platform_facts          TEXT NOT NULL,              -- JSON; open key/value record
    registered_at           TEXT NOT NULL               -- ISO-8601 UTC
);

CREATE TABLE model_sources (
    id                        TEXT PRIMARY KEY,         -- "huggingface"
    supports_revision_pinning INTEGER NOT NULL,         -- bool
    requires_credential       INTEGER NOT NULL          -- bool
);

CREATE TABLE models (
    id                TEXT PRIMARY KEY,                 -- ULID
    source_id         TEXT NOT NULL REFERENCES model_sources(id),
    source_model_id   TEXT NOT NULL,                    -- e.g. HF repo id
    resolved_revision TEXT,                             -- null => explicitly unpinned
    revision_pinned   INTEGER NOT NULL,
    size_bytes        INTEGER,
    content_digest    TEXT,
    UNIQUE (source_id, source_model_id, resolved_revision)
);

CREATE TABLE model_replicas (
    model_id     TEXT NOT NULL REFERENCES models(id),
    node_id      TEXT NOT NULL REFERENCES nodes(id),
    local_path   TEXT NOT NULL,
    state        TEXT NOT NULL,                          -- staging|available|failed
    verified_at  TEXT,
    PRIMARY KEY (model_id, node_id)
);

CREATE TABLE credentials (
    source_id  TEXT NOT NULL REFERENCES model_sources(id),
    name       TEXT NOT NULL,                             -- a source may hold several
    secret_ref TEXT NOT NULL,                             -- a reference, never the value
    is_default INTEGER NOT NULL,
    set_at     TEXT NOT NULL,
    PRIMARY KEY (source_id, name)
);

CREATE TABLE deployments (
    id                TEXT PRIMARY KEY,                   -- ULID, stable across revisions
    name              TEXT NOT NULL UNIQUE,
    desired_state     TEXT NOT NULL,                      -- stopped|running
    current_revision  INTEGER NOT NULL,
    running_revision  INTEGER,                            -- null => never started
    created_at        TEXT NOT NULL
);

-- Immutable. No UPDATE is ever issued against this table; every
-- accepted modification inserts revision n+1. Denormalized model identity
-- keeps a revision sufficient to recreate itself.
CREATE TABLE deployment_revisions (
    deployment_id       TEXT NOT NULL REFERENCES deployments(id),
    revision            INTEGER NOT NULL,
    model_id            TEXT NOT NULL REFERENCES models(id),
    -- denormalized model identity, copied at creation
    model_source_id     TEXT NOT NULL,
    source_model_id     TEXT NOT NULL,
    resolved_revision   TEXT,
    revision_pinned     INTEGER NOT NULL,
    runtime_type        TEXT NOT NULL,                    -- vllm|llamacpp
    runtime_version     TEXT NOT NULL,
    image_reference     TEXT NOT NULL,
    image_digest        TEXT NOT NULL,                    -- platform-specific, not manifest-list
    runtime_config      TEXT NOT NULL,                    -- JSON; per-runtime schema
    participating_nodes TEXT NOT NULL,                    -- JSON ordered list of node ids
    endpoint            TEXT NOT NULL,                    -- host:port
    origin_platform_facts TEXT NOT NULL,                  -- JSON
    created_at          TEXT NOT NULL,
    PRIMARY KEY (deployment_id, revision)
);

-- A unit of management work with progress and a terminal outcome.
CREATE TABLE operations (
    id                  TEXT PRIMARY KEY,                 -- ULID
    kind                TEXT NOT NULL,
    target_type         TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    deployment_revision INTEGER,
    state               TEXT NOT NULL,                    -- pending|running|succeeded|failed
    failure_reason      TEXT,                             -- JSON; structured
    per_node_outcomes   TEXT,                             -- JSON; per-node + overall
    progress            TEXT,                             -- JSON; latest snapshot
    started_at          TEXT,
    finished_at         TEXT
);

CREATE TABLE image_records (
    node_id    TEXT NOT NULL REFERENCES nodes(id),
    reference  TEXT NOT NULL,
    digest     TEXT NOT NULL,                             -- platform-specific
    pulled_at  TEXT NOT NULL,
    PRIMARY KEY (node_id, digest)
);

CREATE INDEX idx_models_source ON models (source_id);
CREATE INDEX idx_replicas_node ON model_replicas (node_id);
CREATE INDEX idx_revisions_deployment ON deployment_revisions (deployment_id);
CREATE INDEX idx_operations_target ON operations (target_type, target_id);
