-- Inference credentials owned by the product.
--
-- The value never lands here: this table holds a *reference* resolved by the
-- credential provider at request time, exactly as `credentials` does for model
-- sources. Nothing in this schema can hold a secret.
--
-- Kept out of `deployment_revisions` on purpose. A revision is what `export`
-- projects, and putting the reference there would mean export had to *remember
-- to exclude* it. A separate binding leaves exports structurally incapable of
-- carrying it, which is what makes the guarantee hold without a redaction step.

CREATE TABLE inference_credentials (
    name       TEXT PRIMARY KEY,                  -- operator-chosen
    secret_ref TEXT NOT NULL,                     -- a reference, never the value
    set_at     TEXT NOT NULL
);

-- One credential per deployment. Absent means the deployment's runtime gets
-- whatever the node was provisioned with, which is the pre-existing behaviour
-- and the migration path for deployments created before this table existed.
CREATE TABLE deployment_inference_credentials (
    deployment_id TEXT PRIMARY KEY REFERENCES deployments(id),
    name          TEXT NOT NULL REFERENCES inference_credentials(name),
    bound_at      TEXT NOT NULL
);
