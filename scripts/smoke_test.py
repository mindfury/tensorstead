#!/usr/bin/env python3
"""Run configured, read-only Tensorstead release smoke checks."""

from __future__ import annotations

import sys
from pathlib import Path

# ``scripts/`` is not a package. Add the repository root so this controller-
# side test runner can import its test-only implementation without installing
# test code into the Tensorstead wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.smoke.runner import SmokeCheckError, SmokeSettings, run


def main() -> int:
    try:
        run(SmokeSettings.from_environment())
    except SmokeCheckError as exc:
        print(f"Smoke test failed: {exc}")
        return 1
    print("Smoke test passed: API, MCP, inference authentication, and Qwen response are healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
