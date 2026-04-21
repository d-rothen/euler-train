"""Batch-aware convenience layer for :meth:`Run.save_outputs`.

The helpers here bridge a collated training batch (for example, the output
of a ``torch.utils.data.DataLoader`` wrapping an
``euler_loading.MultiModalDataset``) and the slot structure that
:meth:`Run.save_outputs` expects:

- sample ids are pulled from the batch dict and sanitized into
  filesystem-safe basenames,
- per-slot tensors are wrapped into ``{sample_id: item}`` mappings so
  ``save_outputs`` uses named filenames instead of sequential indices,
- an optional ``id_schema`` is injected into the manifest metadata so
  downstream consumers can split or match ids back to their source
  dataset.

There is no import-time dependency on euler-loading: the *dataset*
argument is duck-typed through a ``describe_id_schema()`` method and the
batch is duck-typed through its ``full_id`` / ``id`` list entries.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable, Iterable

__all__ = [
    "SAMPLE_ID_UNSAFE_PATTERN",
    "sanitize_sample_id",
    "extract_sample_ids",
    "make_named_slot",
    "build_save_kwargs",
    "save_outputs_from_batch",
]


SAMPLE_ID_UNSAFE_PATTERN = r"[^A-Za-z0-9._-]+"
_SAMPLE_ID_UNSAFE = re.compile(SAMPLE_ID_UNSAFE_PATTERN)


def sanitize_sample_id(name: Any) -> str:
    """Return a filesystem-safe basename derived from *name*.

    Runs of characters outside ``[A-Za-z0-9._-]`` collapse into a single
    underscore, and leading/trailing underscores are stripped. Empty
    results fall back to ``"sample"``.
    """
    text = str(name)
    cleaned = _SAMPLE_ID_UNSAFE.sub("_", text).strip("_")
    return cleaned or "sample"


def extract_sample_ids(
    batch: Mapping[str, Any],
    *,
    sanitize: bool = True,
    keys: Iterable[str] | None = None,
) -> list[str] | None:
    """Pull per-sample identifiers from a collated batch dict.

    PyTorch's default ``collate_fn`` turns a per-sample string field into a
    ``list[str]``. This helper returns the first such list found,
    preferring ``full_id`` (which includes the hierarchy path) over the
    bare ``id``.

    Args:
        batch: A batched sample dict.
        sanitize: When *True*, each id is passed through
                  :func:`sanitize_sample_id` to make it filesystem-safe.
        keys: Optional custom search order. Defaults to
              ``("full_id", "id")``.

    Returns:
        A list of (optionally sanitized) ids, or *None* when no suitable
        field is present.
    """
    lookup = tuple(keys) if keys is not None else ("full_id", "id")
    for key in lookup:
        candidate = batch.get(key)
        if isinstance(candidate, list) and candidate:
            if all(isinstance(v, (str, bytes)) for v in candidate):
                values = [
                    v.decode() if isinstance(v, bytes) else v for v in candidate
                ]
                return [sanitize_sample_id(v) for v in values] if sanitize else values
    return None


def _slice_tensor(value: Any, n: int | None) -> Any:
    """Return ``value[:n]`` — moving torch tensors off-GPU when possible."""
    if n is not None:
        value = value[:n]
    detach = getattr(value, "detach", None)
    if callable(detach):
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        return cpu() if callable(cpu) else detached
    return value


def make_named_slot(
    tensor: Any,
    sample_ids: list[str] | None,
    n: int | None = None,
    *,
    transform: Callable[[Any], Any] | None = None,
) -> Any:
    """Return an indexed batch slice or a ``{sample_id: item}`` dict.

    When *sample_ids* is provided, the tensor's leading dimension is split
    and paired with the ids, producing a dict that
    :meth:`Run.save_outputs` treats as *named* outputs. When *sample_ids*
    is *None*, the tensor is returned unchanged so ``save_outputs`` falls
    back to sequential-index filenames.

    Args:
        tensor: Tensor or ``None`` to wrap.
        sample_ids: Optional list of sanitized ids (one per batch item).
        n: Optional prefix length; defaults to using the whole tensor.
        transform: Optional callable applied to the sliced tensor before
                   wrapping (e.g. a denormalize step).

    Returns:
        ``None`` when *tensor* is ``None``; the sliced tensor when
        *sample_ids* is ``None``; otherwise a ``{id: item}`` mapping.
    """
    if tensor is None:
        return None
    sliced = _slice_tensor(tensor, n)
    if transform is not None:
        sliced = transform(sliced)
    if sample_ids is None:
        return sliced
    count = min(
        n if n is not None else len(sample_ids),
        len(sample_ids),
        len(sliced),
    )
    return {sample_ids[i]: sliced[i] for i in range(count)}


def _map_slot_value(value: Any, sample_ids: list[str] | None, n: int | None) -> Any:
    """Shape a single slot value for :meth:`Run.save_outputs`.

    Dict values (already named, or aux sub-dicts) are copied through
    unchanged; ``None`` skips the slot; everything else is assumed to be a
    tensor/array and is wrapped via :func:`make_named_slot`.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return make_named_slot(value, sample_ids, n)


def build_save_kwargs(
    slots: Mapping[str, Mapping[str, Any] | None],
    sample_ids: list[str] | None,
    n: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply :func:`make_named_slot` across a full save_outputs kwargs dict.

    *slots* has the same ``{output_type: {slot_name: value}}`` shape that
    :meth:`Run.save_outputs` accepts as keyword arguments. Non-dict,
    non-``None`` leaf values are wrapped as named-sample dicts; ``aux``
    sub-dicts are recursed into so each contained tensor is wrapped too.
    """
    result: dict[str, dict[str, Any]] = {}
    for output_type, slot_map in slots.items():
        if slot_map is None:
            continue
        transformed: dict[str, Any] = {}
        for slot_name, value in slot_map.items():
            if slot_name == "aux" and isinstance(value, Mapping):
                aux_transformed: dict[str, Any] = {}
                for aux_name, aux_value in value.items():
                    aux_transformed[aux_name] = _map_slot_value(
                        aux_value, sample_ids, n
                    )
                transformed[slot_name] = aux_transformed
            else:
                transformed[slot_name] = _map_slot_value(value, sample_ids, n)
        result[output_type] = transformed
    return result


def _describe_dataset_id_schema(dataset: Any) -> dict[str, Any] | None:
    """Read an id-construction schema from a duck-typed dataset object."""
    describer = getattr(dataset, "describe_id_schema", None)
    if not callable(describer):
        return None
    try:
        schema = describer()
    except Exception:
        return None
    if isinstance(schema, Mapping) and schema:
        return dict(schema)
    return None


def save_outputs_from_batch(
    run: Any,
    *,
    batch: Mapping[str, Any],
    epoch: int | None = None,
    step: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    dataset: Any = None,
    id_schema: Mapping[str, Any] | None = None,
    n: int | None = None,
    sample_ids: list[str] | None = None,
    include_id_schema: bool = True,
    **output_types: Mapping[str, Any] | None,
) -> Any:
    """Forward a batch of predictions to :meth:`Run.save_outputs`.

    This is the free-function form; :meth:`Run.save_outputs_from_batch` is
    the preferred entry point on a live run.

    Extracts per-sample ids from *batch*, converts each output slot from a
    tensor into the ``{id: item}`` form ``save_outputs`` expects, and —
    when *dataset* (or an explicit *id_schema*) is supplied — injects the
    id-construction schema into *metadata* so downstream tooling can match
    ids back to their ds-crawler origin.

    Args:
        run: Object exposing ``save_outputs(epoch, step, metadata, **kwargs)``.
             Duck-typed; typically a :class:`Run`.
        batch: Collated batch dict; ``full_id`` (preferred) or ``id`` is
               expected to be a ``list[str]``.
        epoch: Forwarded to ``run.save_outputs``.
        step: Forwarded to ``run.save_outputs``.
        metadata: Base metadata to forward. When *include_id_schema* is
                  *True*, *metadata* is not *None*, and either *id_schema*
                  or ``dataset.describe_id_schema()`` is available, the
                  schema is merged under ``metadata["id_schema"]``
                  (existing values take precedence). When *metadata* is
                  *None*, no id-schema is injected — injecting would
                  create a metadata dict missing the required ``dataset``
                  key that :meth:`Run.save_outputs` validates.
        dataset: Optional dataset object exposing ``describe_id_schema()``.
                 Used only for metadata enrichment — no hard dependency.
        id_schema: Optional explicit id-schema dict. Overrides the
                   dataset-derived schema when both are supplied.
        n: Optional batch-prefix length; defaults to the full batch.
        sample_ids: Optional explicit id list overriding batch extraction.
        include_id_schema: When *False*, skip id-schema injection entirely.
        **output_types: Per-output-type slot dicts forwarded to
                        ``run.save_outputs`` after id-wrapping.

    Returns:
        Whatever ``run.save_outputs`` returns (typically the output
        directory).
    """
    if sample_ids is None:
        sample_ids = extract_sample_ids(batch)

    kwargs = build_save_kwargs(output_types, sample_ids, n)

    merged_metadata: dict[str, Any] | None = (
        dict(metadata) if metadata is not None else None
    )

    if include_id_schema and merged_metadata is not None:
        schema = dict(id_schema) if id_schema is not None else None
        if schema is None and dataset is not None:
            schema = _describe_dataset_id_schema(dataset)
        if schema:
            merged_metadata.setdefault("id_schema", schema)

    return run.save_outputs(
        epoch=epoch,
        step=step,
        metadata=merged_metadata,
        **kwargs,
    )
