from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from euler_dataset_contract import (
    register_addon_validator,
    validate_addon_version,
    validate_slot,
    validate_string_list,
    validate_token,
)


_ALLOWED_USED_AS = {"condition", "input", "output", "target"}
_ALLOWED_KEYS = {
    "version",
    "used_as",
    "slot",
    "hierarchy_scope",
    "applies_to",
    "task",
}


def validate_euler_train_addon(value: Any, context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")

    unknown = sorted(set(value.keys()) - _ALLOWED_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown {context} key(s): {joined}")

    validate_addon_version(value.get("version"), f"{context}.version")
    validate_role_fields(value, context)


def validate_role_fields(value: dict[str, Any], context: str) -> None:
    """Validate ``used_as``/``slot``/``task``/``hierarchy_scope``/``applies_to``.

    Shared by the ``addons.euler_train`` head validator and by the
    ``modality_roles`` a trainer passes to :func:`euler_train.init`.
    """
    used_as = value.get("used_as")
    if used_as is not None:
        if not isinstance(used_as, str) or not used_as:
            raise ValueError(f"{context}.used_as must be a non-empty string")
        if used_as not in _ALLOWED_USED_AS:
            allowed = ", ".join(sorted(_ALLOWED_USED_AS))
            raise ValueError(
                f"{context}.used_as must be one of {{{allowed}}}, got {used_as!r}"
            )

    slot = value.get("slot")
    if slot is not None:
        validate_slot(slot, f"{context}.slot")

    task = value.get("task")
    if task is not None:
        validate_token(task, f"{context}.task")

    hierarchy_scope = value.get("hierarchy_scope")
    applies_to = value.get("applies_to")
    if hierarchy_scope is not None or applies_to is not None:
        if used_as != "condition":
            raise ValueError(
                f"{context}.hierarchy_scope and {context}.applies_to are only "
                "allowed when used_as is 'condition'"
            )
        if hierarchy_scope is not None:
            validate_token(hierarchy_scope, f"{context}.hierarchy_scope")
        if applies_to is not None:
            validate_string_list(
                applies_to,
                f"{context}.applies_to",
                allow_wildcard=True,
            )


def normalize_modality_roles(value: Any) -> dict[str, dict[str, Any]]:
    """Validate a trainer's ``modality_roles`` and return ``{name: fields}``.

    Each value is either a ``used_as`` string (``"input"``) or a mapping with
    the role keys of the ``addons.euler_train`` head addon (``used_as``,
    ``slot``, ``task``, ``hierarchy_scope``, ``applies_to``).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "modality_roles must be a mapping of modality name to a used_as "
            f"string or a role mapping, got {type(value).__name__}"
        )
    allowed = _ALLOWED_KEYS - {"version"}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_role in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("modality_roles keys must be non-empty modality names")
        context = f"modality_roles[{name!r}]"
        if isinstance(raw_role, str):
            role: dict[str, Any] = {"used_as": raw_role}
        elif isinstance(raw_role, Mapping):
            role = {str(k): v for k, v in raw_role.items() if v is not None}
            unknown = sorted(set(role) - allowed)
            if unknown:
                raise ValueError(f"Unknown {context} key(s): {', '.join(unknown)}")
        else:
            raise TypeError(
                f"{context} must be a used_as string or a mapping, "
                f"got {type(raw_role).__name__}"
            )
        validate_role_fields(role, context)
        normalized[name] = role
    return normalized


register_addon_validator("euler_train", validate_euler_train_addon, overwrite=True)
