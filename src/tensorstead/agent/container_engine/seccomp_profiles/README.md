# Seccomp profiles

`docker-default-v29.9.1.json` is Docker's default seccomp profile, copied unchanged
from `vendor/github.com/moby/profiles/seccomp/default.json` at moby tag
`docker-v29.9.1` (commit b26fb77d3e7b01f9d21b89539c1ab428f5ca431f,
sha256 785b2429264afba4d594320337cb17f144f3c7d51585f9805eef72e28f4f9334).
Licensed Apache-2.0 by the Moby project.

The agent never applies it as-is: Docker already does that for every container.
`tensorstead.agent.container_engine.seccomp` adds approved syscall groups to it,
for deployments whose adapter declared `seccomp_allow`.
