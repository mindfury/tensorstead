"""Test fakes at the six boundaries.

Hand-written fakes (never ``unittest.mock``) so that *behaviour* — stage-then-
promote, idempotency, divergence — is modelled, not just call assertions. The
agent conformance suite (``test_agent_conformance.py``) runs against both the
fake and the real agent so the fake is provably faithful, not merely convenient.

The six boundaries: node agent, peer agent, container engine, huggingface_hub,
NVML, and service manager.
"""
