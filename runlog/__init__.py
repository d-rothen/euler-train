"""runlog — lightweight file-based experiment logging."""

from .run import Run

__all__ = ["init", "Run"]
__version__ = "0.1.0"


def init(
    dir: str,
    config=None,
    meta: dict | None = None,
    output_formats: dict[str, str] | None = None,
) -> Run:
    """Create a new run and return the :class:`Run` handle.

    Parameters
    ----------
    dir:
        Project / output directory.  Each call creates a unique run
        under ``{dir}/runs/{timestamp_id}/``.
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
    """
    return Run(dir=dir, config=config, meta=meta, output_formats=output_formats)
