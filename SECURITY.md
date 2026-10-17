# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/mindfury/tensorstead/security/advisories/new)
("Report a vulnerability" on the repository's Security tab). Do not open a public
issue for a suspected vulnerability.

Include what you found, how to reproduce it, and the Tensorstead version
(`stead --version`) or commit. You should get an acknowledgement within a week.

## Supported versions

Fixes land on `main` and ship in the next release. Only the latest release is
supported.

## Scope

Tensorstead manages inference runtimes on hosts it is given management
credentials for. Of particular interest:

- anything that lets a deployment record, `extra_args` or `host_config` load
  code, relax container isolation, or reach host resources without the reviewed
  approval the authorization policy requires (`option_policy.py`);
- credential exposure: management tokens, inference keys, or source credentials
  appearing in argv, logs, exports or API responses;
- the coordinator↔agent and agent↔agent channels (TLS, authentication).
