-- What image each revision actually ran on each node (CodeGuard CG-10).
--
-- A revision records the image *reference* it was asked to run; the digest is
-- only a fact once an agent has resolved it, and service-created revisions
-- store image_digest = ''. The image-deletion guard compared that empty field,
-- so nothing a deployment really used was protected. This records the digest
-- the agent reported when it created the container, per revision and node.
--
-- Append-only history: a later re-point of the tag does not rewrite what a
-- revision ran, so a retained revision stays protected after the tag moves.
CREATE TABLE revision_images (
    deployment_id TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    node_id       TEXT NOT NULL,
    image_digest  TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (deployment_id, revision, node_id, image_digest)
);

CREATE INDEX idx_revision_images_digest ON revision_images (node_id, image_digest);
