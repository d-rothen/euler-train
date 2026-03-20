"""runlog — lightweight file-based experiment logging."""
from __future__ import annotations

from .run import Run
from .architecture import export_architecture

__all__ = ["init", "Run", "export_architecture"]
__version__ = "0.1.0"


def init(
    dir: str | None = None,
    config=None,
    meta: dict | None = None,
    output_formats: dict[str, str] | None = None,
    run_id: str | None = None,
    datasets: dict | None = None,
    run_name: str | None = None,
    evaluations: dict[str, dict] | None = None,
    mode: str | None = None,
) -> Run:
    """Create a new run — or resume an existing one — and return the handle.

    Parameters
    ----------
    dir:
        Project / output directory.  Each call creates a unique run
        under ``{dir}/runs/{timestamp_id}/``.  When *None* (the
        default), the directory is resolved as:

        1. ``$ET_HOME/<project>`` (if ``$ET_HOME`` is set),
        2. ``~/euler_train/<project>``,

        where ``<project>`` is the git repository name, or the current
        working directory name when not inside a git repo.

        **Run-directory shorthand:** if *dir* points to an existing run
        directory (i.e. it contains a ``meta.json`` file) and *run_id*
        is not provided, the run is automatically resumed — the
        ``run_id`` is read from ``meta.json`` and the project directory
        is derived from the path (``dir/../..``).
    config:
        Hyperparameters — accepts a *dict*, a path to a JSON / YAML file,
        an ``argparse.Namespace``, or a dataclass instance.
    meta:
        Extra user-defined fields merged into ``meta.json``
        (e.g. ``{"description": "baseline", "tags": ["v2"]}``).
    output_formats:
        Override auto-inferred save formats.  Keys can be an output type
        (``"depth"``), a slot / aux name (``"transmission"``), or a
        dotted combination (``"depth.pred"``).  Values are ``"png"``,
        ``"npy"``, or ``"npz"``.
    run_id:
        If given, resume an existing run instead of creating a new one.
        The run directory ``{dir}/runs/{run_id}/`` must already exist.
        The existing ``config.json`` is loaded automatically (unless
        *config* is explicitly provided to override it).
    datasets:
        Optional mapping of split name to ``euler_loading.MultiModalDataset``
        instance (e.g. ``{"train": train_ds, "val": val_ds}``).  When
        provided, each split is logged into ``meta.json`` under
        ``datasets[split]`` with per-modality records:
        ``path`` and inferred metadata (``used_as``, ``slot``,
        ``modality_type``).  Hierarchical modalities also include
        ``hierarchy_scope`` and ``applies_to``.  If a dataset implements
        ``describe_for_runlog()``, that contract is used directly.
        Otherwise inference prefers
        ``ds-crawler`` config properties when available, then falls back to
        naming-based heuristics.
    run_name:
        Optional human-readable name for the run.  Stored in ``meta.json``.
    evaluations:
        Optional mapping of evaluation key to evaluation entry.  Each
        entry may contain ``datasets`` (same dataset objects accepted by
        *datasets*), ``name``, ``status``, ``checkpoint``, and
        ``metadata``.  Typically used when resuming a run (via *run_id*)
        for evaluation.  See also :meth:`Run.add_evaluation`.
    mode:
        Optional label for the current process context (for example
        ``"train"``, ``"val"``, or ``"eval"``).  When provided,
        lifecycle fields and crash details are mirrored into
        ``meta.json`` under ``modes[mode]``.
    """
    return Run(
        dir=dir, config=config, meta=meta,
        output_formats=output_formats, run_id=run_id,
        datasets=datasets, run_name=run_name,
        evaluations=evaluations, mode=mode,
    )
