"""Shared authorization for runtime ``extra_args`` keys.

Every adapter offers an unvalidated passthrough, because a schema that cannot
express what an operator needs does not stop them: it sends them around the
management plane, after which the coordinator's records describe a container
nobody is managing. The passthrough is deliberate.

What is *not* deliberate is the passthrough reaching options the product owns.
Each adapter reserved a small set of names — the model path, the inference
credential, the endpoint, the derived topology — and each adapter checked those
names in one spelling only. A command line has more than one spelling per
option, so the reserved names were bypassable in two ways that both reproduced
in practice:

- **embedded values.** ``{"model=/tmp/other": true}`` normalises to the literal
  key ``model=/tmp/other``, which matches no reserved name, and then renders as
  the single token ``--model=/tmp/other``. Ordinary long-option syntax, and it
  reached ``--model``, ``--api-key``, ``--nnodes`` and the rest.
- **negative aliases.** ``{"no-flash-attn": true}`` looks for a field named
  ``no_flash_attn``, finds none, and renders alongside the ``--flash-attn`` the
  modelled field already produced. Both sides of one boolean, in one argv.

The fix lives here rather than in each adapter because the defect was *two*
adapters implementing the same rule separately and both getting it wrong the
same way. A third adapter inheriting the correct rule for free is the point;
patching a second name list twice is exactly the mistake review caught.

The rule this enforces: an ``extra_args`` key must be a bare flag name, and
every spelling that reaches a product-owned or modelled option gets the same
answer as its canonical spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args


def normalize_flag(key: str) -> str:
    """Reduce an ``extra_args`` key to its canonical flag name.

    Leading dashes optional, underscores or dashes accepted: ``--flash_attn``,
    ``flash-attn`` and ``flash_attn`` are one option and must receive one
    answer.
    """
    return key.strip().lstrip("-").replace("_", "-")


def _is_boolean_field(field: Any) -> bool:
    """Whether a Pydantic field models a boolean, optional or not.

    ``bool`` and ``bool | None`` both count: a negative alias contradicts the
    modelled field either way.
    """
    annotation = getattr(field, "annotation", None)
    return annotation is bool or bool in get_args(annotation)


def authorize_extra_arg(
    key: str,
    *,
    forbidden: Mapping[str, str],
    model_fields: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` when ``key`` reaches something config does not own.

    ``forbidden`` maps a canonical flag name to the reason the product owns it.
    ``model_fields`` is the config model's own fields — a modelled option must
    be set through its field, where it is validated, rather than forwarded
    unchecked.

    Silent on success: an unmodelled, unreserved flag is exactly what the
    passthrough is for, and this function's job is to stay out of its way.
    """
    if "=" in key:
        # The dictionary value is already the value channel. Permitting a
        # second one means the same option has two spellings and only one of
        # them is checked -- which is how `--model=/tmp/other` got through.
        raise ValueError(
            f"extra_args key {key!r} may not contain '=': write the flag as the key and "
            f"its value as the value, so the flag can be checked against the options "
            f"this product owns"
        )
    if any(character.isspace() for character in key.strip()):
        # Not one of the recorded cases, but the same class: a key that is not a flag name
        # renders as one argv token that no parser will read as the operator
        # intended, and it is never checked against anything.
        raise ValueError(
            f"extra_args key {key!r} may not contain whitespace: a key is one flag name, "
            f"and its value belongs in the value"
        )

    flag = normalize_flag(key)
    if not flag:
        raise ValueError(f"extra_args key {key!r} names no flag")

    if flag in forbidden:
        raise ValueError(f"extra_args may not set {key!r}: {forbidden[flag]}")

    modelled = flag.replace("-", "_")
    if modelled in model_fields and modelled != "extra_args":
        raise ValueError(
            f"extra_args may not set {key!r}: it is a validated field, "
            f"set {modelled!r} directly so it is checked rather than forwarded"
        )

    # A negative alias is the same option approached from the other side. Only
    # resolved against names this product actually owns or models: an unmodelled
    # negative flag is ordinary runtime surface and stays a legitimate
    # passthrough, which is most of what `--no-` flags are.
    if flag.startswith("no-"):
        positive = flag[3:]
        if positive in forbidden:
            raise ValueError(
                f"extra_args may not set {key!r}: it negates {positive!r}, which "
                f"{forbidden[positive]}"
            )
        positive_field = positive.replace("-", "_")
        field = model_fields.get(positive_field)
        if field is not None and positive_field != "extra_args" and _is_boolean_field(field):
            raise ValueError(
                f"extra_args may not set {key!r}: it contradicts the validated field "
                f"{positive_field!r}, which would render both sides of one option. "
                f"Set {positive_field!r} to false instead"
            )
