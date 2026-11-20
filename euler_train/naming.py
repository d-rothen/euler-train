"""Naming checks of training naming 2.0 runs and validation sets.

A run whose ``metric_naming`` declares ``schema_version`` 2.x is held to it for
every key it logs (``docs/training.md`` of euler-metric-naming). The checks run
through euler-metric-naming (the ``naming`` extra, 0.4 or newer) once per
distinct key; what fails is warned about once and counted in ``meta.json``
under ``naming_report``. Nothing is dropped. With ``strict_naming=True`` the
first problem raises :class:`NamingError` instead, for CI and smoke runs.

Runs without a declaration, or with a schema-1 declaration, are not checked.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

log = logging.getLogger("euler_train")

#: Fields of a train.jsonl / val.jsonl record that are not metrics.
RESERVED_RECORD_FIELDS = frozenset({"step", "epoch", "wall_time", "elapsed_sec"})
#: At most this many problems are listed in the report; the rest are counted.
MAX_REPORTED_PROBLEMS = 200

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)$")


class NamingError(ValueError):
    """A key, declaration or validation record breaks the run's naming (strict mode)."""


def schema_major(naming: Any) -> int | None:
    """The major of a ``metric_naming`` declaration's ``schema_version``, or None."""
    if not isinstance(naming, Mapping):
        return None
    match = _VERSION_RE.match(str(naming.get("schema_version", "")))
    return int(match.group(1)) if match else None


def _validator() -> tuple[Any, str | None]:
    """euler-metric-naming's contract module and version, or ``(None, why not)``."""
    try:
        import euler_metric_naming
        from euler_metric_naming import contract
    except ImportError:
        return None, "euler-metric-naming is not installed (pip install 'euler-train[naming]')"
    if not hasattr(contract, "training_metric_problem"):
        version = getattr(euler_metric_naming, "__version__", "?")
        return None, f"euler-metric-naming {version} predates training naming 2.0 (needs 0.4)"
    return contract, None


def require_strict(naming: Any) -> None:
    """Raise :class:`NamingError` unless *naming* can be held to strictly."""
    if schema_major(naming) != 2:
        raise NamingError(
            "strict_naming needs a training naming 2.0 declaration "
            "(metric_naming with schema_version '2.x')"
        )
    contract, why_not = _validator()
    if contract is None:
        raise NamingError(f"strict_naming cannot check the run: {why_not}")


class NamingGuard:
    """Checks the keys of a run and keeps its ``naming_report``."""

    def __init__(
        self,
        naming: Any,
        *,
        strict: bool = False,
        report: Mapping[str, Any] | None = None,
    ) -> None:
        self.strict = bool(strict)
        self.active = schema_major(naming) == 2
        self._naming = naming
        self._known: dict[tuple[str, str], str | None] = {}
        self._problems: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._truncated = 0
        self._sets_used = False
        self._contract, self._why_not = (None, None)
        if strict:
            require_strict(naming)
        if not self.active:
            return
        self._contract, self._why_not = _validator()
        if self._contract is None:
            if strict:
                raise NamingError(f"strict_naming cannot check the run: {self._why_not}")
            log.warning("Training naming 2.0 keys are not checked: %s", self._why_not)
        if isinstance(report, Mapping):
            self._load(report)
        if self._contract is not None:
            for problem in self._contract.training_naming_problems(naming):
                self._record("metric_naming", problem.path, problem.code, problem.message, None)

    @property
    def checked(self) -> bool:
        return self.active and self._contract is not None

    # -- checks -------------------------------------------------------------------------------

    def check_record(self, source: str, record: Mapping[str, Any], step: int | None) -> None:
        """Check every key and value of one ``train.jsonl`` / ``val.jsonl`` record."""
        if not self.checked:
            return
        contract = self._contract
        for key, value in record.items():
            if key in RESERVED_RECORD_FIELDS:
                continue
            cache = (source, key)
            if cache not in self._known:
                problem = contract.training_metric_problem(self._naming, key)
                self._known[cache] = None if problem is None else problem.code
                if problem is not None:
                    self._record(source, key, problem.code, problem.message, step)
            elif self._known[cache] is not None:
                self._count(source, key, self._known[cache])
            value_problem = contract.metric_value_problem(value)
            if value_problem is not None:
                self._record(source, key, value_problem.code, value_problem.message, step)

    def check_set(self, key: str, document: Mapping[str, Any]) -> None:
        """Check a validation set document (its own rules and its protocol block)."""
        self._sets_used = True
        if self._contract is None:
            contract, why_not = _validator()
            if contract is None:
                if self.strict:
                    raise NamingError(f"cannot check validation set {key!r}: {why_not}")
                return
        else:
            contract = self._contract
        source = f"validation/{key}"
        for problem in contract.validation_set_problems(document):
            self._record(source, problem.path, problem.code, problem.message, None)
        protocol = document.get("protocol") if isinstance(document, Mapping) else None
        if isinstance(protocol, Mapping) and str(protocol.get("schemaVersion", "")).startswith("2."):
            for message in contract.protocol_problems(protocol):
                self._record(source, "protocol", "protocol", message, None)

    def check_pass(
        self, key: str, document: Mapping[str, Any], record: Mapping[str, Any], step: int | None
    ) -> None:
        """Check one validation pass record against its set document."""
        self._sets_used = True
        contract = self._contract
        if contract is None:
            contract, _ = _validator()
            if contract is None:
                return
        source = f"validation/{key}"
        for problem in contract.validation_pass_problems(document, record):
            self._record(source, problem.path, problem.code, problem.message, step)

    # -- the report ----------------------------------------------------------------------------

    def _record(self, source, key, code, message, step) -> None:
        entry_key = (source, key, code)
        entry = self._problems.get(entry_key)
        if entry is not None:
            entry["count"] += 1
            return
        if self.strict:
            raise NamingError(f"{source}: {key or '(document)'}: {code}: {message}")
        if len(self._problems) >= MAX_REPORTED_PROBLEMS:
            self._truncated += 1
            return
        log.warning("Naming: %s: %s: %s (%s)", source, key or "(document)", code, message)
        entry = {"source": source, "key": key, "code": code, "count": 1}
        if step is not None:
            entry["first_step"] = step
        self._problems[entry_key] = entry

    def _count(self, source: str, key: str, code: str) -> None:
        entry = self._problems.get((source, key, code))
        if entry is not None:
            entry["count"] += 1
        else:
            self._truncated += 1

    def _load(self, report: Mapping[str, Any]) -> None:
        problems = report.get("problems")
        if isinstance(problems, list):
            for entry in problems:
                if not isinstance(entry, Mapping):
                    continue
                source, key, code = entry.get("source"), entry.get("key"), entry.get("code")
                if isinstance(source, str) and isinstance(key, str) and isinstance(code, str):
                    self._problems[(source, key, code)] = dict(entry)
                    if not source.startswith("validation/") and source != "metric_naming":
                        self._known[(source, key)] = code
        truncated = report.get("truncated")
        if isinstance(truncated, int):
            self._truncated = truncated

    def report(self) -> dict[str, Any] | None:
        """The ``naming_report`` of ``meta.json`` (None for runs that are not checked)."""
        if not (self.active or self._sets_used or self._problems):
            return None
        contract, why_not = (self._contract, self._why_not)
        if contract is None:
            contract, why_not = _validator()
        report: dict[str, Any] = {
            "schema_version": (
                str(self._naming.get("schema_version")) if self.active else None
            ),
            "checked": contract is not None,
        }
        if contract is not None:
            import euler_metric_naming

            report["validator"] = f"euler-metric-naming {euler_metric_naming.__version__}"
        else:
            report["message"] = why_not
        report["problems"] = [dict(entry) for entry in self._problems.values()]
        report["truncated"] = self._truncated
        return report
