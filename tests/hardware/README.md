# Hardware tests — safety rules

These tests run against **real, stateful, shared** inference hosts. Unlike the unit, contract, and
integration tiers, there is no fresh in-memory store here and no fake to absorb a mistake: a careless test
deletes someone's model weights or stops a deployment another person is using.

Read this before adding one.

## The rules

1. **Create only uniquely identifiable, test-owned resources.**
   Every resource a test creates carries a name that could not plausibly belong to anything else — prefix
   `tensorstead-test-`, then a run-unique suffix. Never `test-model` or `llama`.

2. **Detect collisions before mutating.**
   If a resource with the intended name already exists, the test **fails** rather than adopting or
   overwriting it. A name collision means either a previous run leaked or something else is using that
   name; both deserve a human, not a `--force`.

3. **Never touch what you did not create.**
   No test enumerates deployments and stops them. No test clears the model store. No test removes an image
   because disk was low. If a test needs a clean host, it says so and skips.

4. **Clean up what you created, even on failure.**
   Use a fixture with teardown, not a trailing call that a failed assertion skips past. Teardown removes
   only the run's own resources, matched by that run's unique prefix.

5. **Leave the host as you found it.**
   Reboots, service restarts, and driver changes belong to the `hardware_disruptive` tier, which is behind
   a *second* opt-in for exactly this reason.

## Tiers and opt-in

Both flags are off by default, so neither tier can run by accident:

| Tier | Marker | Flag | What it may do |
|---|---|---|---|
| Hardware | `@pytest.mark.hardware` | `--hardware` | Read state, create and remove its own resources |
| Disruptive | `@pytest.mark.hardware_disruptive` | `--hardware-disruptive` | Reboot, restart services, tear down multi-node state |

```bash
make test-hardware      # non-disruptive only
uv run pytest -m hardware_disruptive tests/hardware --hardware-disruptive
```

Tests additionally require `TENSORSTEAD_HARDWARE_ENABLED=1`, so a stray `--hardware` on a laptop still does
nothing.

## Environment

| Variable | Meaning |
|---|---|
| `TENSORSTEAD_HARDWARE_ENABLED` | Must be `1` for any hardware test to run |
| `TENSORSTEAD_API` | Coordinator base URL |
| `TENSORSTEAD_MGMT_TOKEN` | Management token for that coordinator |
| `TENSORSTEAD_TEST_NODE` | Name of the registered node to use |
| `TENSORSTEAD_TEST_NODE_B` | Second node, for multi-node tests |

A multi-node test that finds only one node configured **skips**. It does not quietly degrade to
single-node and report success — that would turn a test of distribution into a test of nothing while still
showing green.

## Why this file exists

Nothing fails when these rules are absent. That is precisely the problem: the cost of breaking them lands
on whoever is using the appliance, not on the test run that broke them.
