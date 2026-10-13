"""The model name a deployment declares it serves under.

Runtimes spell it differently -- vLLM, TensorRT-LLM and ExLlama model a
``served_model_name`` setting; SGLang deployments on the estate pass
``served-model-name`` through ``extra_args``; llama.cpp calls it ``--alias``
-- and the collision guard needs one answer. ``None`` means the deployment
declared no name and the runtime derives one from the model path, which this
module cannot know and does not guess.
"""

from __future__ import annotations

from typing import Any

# extra_args keys that name the served model, in the spellings runtimes accept.
_EXTRA_ARG_KEYS = ("served-model-name", "served_model_name", "alias")


def declared_served_name(runtime_config: dict[str, Any] | None) -> str | None:
    """The served model name a runtime config declares, or None."""
    config = runtime_config or {}
    modelled = config.get("served_model_name")
    if isinstance(modelled, str) and modelled.strip():
        return modelled.strip()
    extra = config.get("extra_args") or {}
    if isinstance(extra, dict):
        for key in _EXTRA_ARG_KEYS:
            for spelling in (key, f"--{key}"):
                value = extra.get(spelling)
                if isinstance(value, str) and value.strip():
                    # vLLM accepts several names separated by spaces; the
                    # first is the one reported in responses.
                    return value.split()[0]
    return None
