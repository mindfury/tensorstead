-- Managed runtime images.
--
-- A build spec is data, not a command: recording one executes nothing. It is
-- retained after the image it produced is replaced, so an operator can still
-- read what an image contained and why it existed.

CREATE TABLE image_build_specs (
    name        TEXT PRIMARY KEY,
    base_image  TEXT NOT NULL,              -- pinned by digest, or reported unreproducible
    steps       TEXT NOT NULL,              -- JSON array, ordered
    created_at  TEXT NOT NULL
);

-- Provenance for images already tracked. `origin` distinguishes a registry
-- digest from a locally produced image identifier: the product forbids presenting
-- one as the other, because that makes an export look portable when it is not.
ALTER TABLE image_records ADD COLUMN origin TEXT NOT NULL DEFAULT 'pulled';
ALTER TABLE image_records ADD COLUMN produced_by TEXT;
