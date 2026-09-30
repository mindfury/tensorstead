-- A per-node override for the coordinator-to-agent management credential
-- Empty means "no override; use the
-- fleet-wide TENSORSTEAD_MGMT_TOKEN", matching the existing migration path a
-- node predating this feature keeps working under.
--
-- Stores a *reference* into the credential provider, never a value -- the
-- same rule that already applies to every other secret this product
-- tracks (InferenceCredential.secret_ref, Credential.secret_ref).
ALTER TABLE nodes ADD COLUMN agent_management_token_ref TEXT NOT NULL DEFAULT '';
