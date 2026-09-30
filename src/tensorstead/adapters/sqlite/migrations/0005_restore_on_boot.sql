-- Boot restoration becomes a declared, default-off property.
--
-- Boot restoration installed and enabled a systemd unit the moment a deployment
-- started, and desired lifecycle state was named as the sole control. So a
-- deployment earned the right to run on every future boot *before anyone knew
-- whether it could run at all*.
--
-- On 2026-08-15 a deployment deadlocked the NVIDIA kernel driver on its first
-- start. It had never completed a successful run, and boot restoration brought
-- it back on every reboot, which is what turned a recoverable incident into a
-- reimaged node. The operator's escape hatch from a wedged appliance is a
-- reboot; boot restoration is what took the escape hatch away.
--
-- DEFAULT 0 is the point. Every revision that predates this column loses boot
-- persistence, which is a real behaviour change and the safe direction: a
-- deployment nobody has asked to be persistent should not be. Operators who
-- want it back declare it, and declaring it is now visible in the record.
ALTER TABLE deployment_revisions
    ADD COLUMN restore_on_boot INTEGER NOT NULL DEFAULT 0;
