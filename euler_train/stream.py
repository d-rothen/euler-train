"""Optional model-run stream consumers for Euler View."""
from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .serialization import _json_default

log = logging.getLogger("euler_train.stream")


class OutputStreamConsumer(Protocol):
    """Minimal interface for best-effort event consumers."""

    def bind(self, context: "StreamContext") -> None:
        ...

    def emit(self, event: dict[str, Any]) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class StreamContext:
    run_id: str
    project_dir: Path
    runs_dir: Path
    run_dir: Path
    meta: dict[str, Any]
    config: dict[str, Any]
    code_ref: dict[str, Any] | None
    run_environment: dict[str, Any] | None


@dataclass
class EulerViewStreamConfig:
    """Connection settings for the built-in Euler View HTTP consumer."""

    base_url: str
    model_id: int | None = None
    api_token: str | None = None
    stream_token: str | None = None
    stream_attach_token: str | None = None
    datasource_id: int | None = None
    euler_train_dir: str | None = None
    run_dir: str | None = None
    timeout_sec: float = 10.0
    batch_size: int = 20
    flush_interval_sec: float = 2.0
    max_pending_events: int = 1000
    session_expires_in_sec: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str):
            raise ValueError("stream.base_url must be a string")
        self.base_url = self.base_url.strip().rstrip("/")
        if not self.base_url:
            raise ValueError("stream.base_url is required")

        self.api_token = _normalize_optional_text(
            self.api_token,
            field_name="stream.api_token",
        )
        self.stream_token = _normalize_optional_text(
            self.stream_token,
            field_name="stream.stream_token",
        )
        self.stream_attach_token = _normalize_optional_text(
            self.stream_attach_token,
            field_name="stream.stream_attach_token",
        )
        self.euler_train_dir = _normalize_optional_text(
            self.euler_train_dir,
            field_name="stream.euler_train_dir",
        )
        self.run_dir = _normalize_optional_text(
            self.run_dir,
            field_name="stream.run_dir",
        )

        self.model_id = _require_positive_int(
            self.model_id,
            field_name="stream.model_id",
            allow_none=True,
        )
        self.datasource_id = _require_positive_int(
            self.datasource_id,
            field_name="stream.datasource_id",
            allow_none=True,
        )
        self.session_expires_in_sec = _require_positive_int(
            self.session_expires_in_sec,
            field_name="stream.session_expires_in_sec",
            allow_none=True,
        )
        self.batch_size = _require_positive_int(
            self.batch_size,
            field_name="stream.batch_size",
        )
        self.max_pending_events = _require_positive_int(
            self.max_pending_events,
            field_name="stream.max_pending_events",
        )
        self.timeout_sec = _require_positive_number(
            self.timeout_sec,
            field_name="stream.timeout_sec",
        )
        self.flush_interval_sec = _require_non_negative_number(
            self.flush_interval_sec,
            field_name="stream.flush_interval_sec",
        )

        if self.stream_token is None:
            if self.model_id is None and self.stream_attach_token is None:
                raise ValueError(
                    "stream.model_id is required when neither stream.stream_token nor stream.stream_attach_token is provided",
                )
            if self.api_token is None:
                raise ValueError(
                    "stream.api_token is required when stream.stream_token is not provided",
                )


class OutputStream:
    """Dispatch events to one or more best-effort consumers."""

    def __init__(self, consumers: Sequence[OutputStreamConsumer]) -> None:
        self._consumers = list(consumers)
        self._closed = False

    @property
    def consumers(self) -> tuple[OutputStreamConsumer, ...]:
        return tuple(self._consumers)

    def bind(self, context: StreamContext) -> None:
        self._dispatch("bind", context)

    def emit(self, event: dict[str, Any]) -> None:
        self._dispatch("emit", event)

    def flush(self) -> None:
        self._dispatch("flush")

    def close(self) -> None:
        if self._closed:
            return
        self._dispatch("flush")
        self._dispatch("close")
        self._closed = True

    def _dispatch(self, method_name: str, *args: Any) -> None:
        if self._closed and method_name != "close":
            return

        active: list[OutputStreamConsumer] = []
        for consumer in self._consumers:
            method = getattr(consumer, method_name)
            payload = copy.deepcopy(args)
            try:
                method(*payload)
                active.append(consumer)
            except Exception as exc:  # pragma: no cover - defensive safeguard
                log.warning(
                    "Disabling stream consumer after %s failure: %s",
                    method_name,
                    exc,
                )
        self._consumers = active


class EulerViewStreamConsumer:
    """Best-effort HTTP consumer for the model-run streaming API."""

    def __init__(self, config: EulerViewStreamConfig) -> None:
        self.config = config
        self._context: StreamContext | None = None
        self._pending: list[dict[str, Any]] = []
        self._stream_token: str | None = config.stream_token
        self._resolved_stream_attach_token: str | None = config.stream_attach_token
        self._session_expires_at = float("inf") if config.stream_token else 0.0
        self._ingest_url = f"{self.config.base_url}/api/model-run-stream/ingest"
        self._last_flush_at = 0.0
        self._next_retry_at = 0.0
        self._last_warning: tuple[str, float] | None = None
        self._closed = False

    def bind(self, context: StreamContext) -> None:
        self._context = context

    def emit(self, event: dict[str, Any]) -> None:
        if self._closed:
            return

        try:
            payload = _to_jsonable(event)
        except Exception as exc:
            self._warn(
                "Dropping stream event that could not be serialized",
                exc,
            )
            return

        self._pending.append(payload)
        self._trim_pending()

        force = payload.get("type") in {
            "init",
            "checkpoint",
            "output_snapshot",
            "finish",
        }
        self._maybe_flush(force=force)

    def flush(self) -> None:
        self._flush_pending(force=True)

    def close(self) -> None:
        if self._closed:
            return
        self._flush_pending(force=True)
        self._closed = True

    def _maybe_flush(self, *, force: bool = False) -> None:
        if not self._pending:
            return

        now = time.time()
        should_flush = force
        should_flush = should_flush or len(self._pending) >= self.config.batch_size
        should_flush = should_flush or (
            now - self._last_flush_at >= self.config.flush_interval_sec
        )
        if should_flush:
            self._flush_pending(force=force)

    def _flush_pending(self, *, force: bool = False) -> None:
        if not self._pending:
            return

        now = time.time()
        if not force and now < self._next_retry_at:
            return

        if not self._ensure_session(force=force):
            return

        while self._pending:
            batch = self._pending[: self.config.batch_size]
            try:
                self._post_json(
                    self._ingest_url,
                    {"events": batch},
                    token=self._stream_token,
                )
            except Exception as exc:
                self._schedule_retry()
                self._warn("Model-run stream ingest failed", exc)
                return

            del self._pending[: len(batch)]
            self._last_flush_at = time.time()
            self._next_retry_at = 0.0

    def _ensure_session(self, *, force: bool = False) -> bool:
        if self._stream_token and time.time() + 5 < self._session_expires_at:
            return True

        if self.config.stream_token:
            self._stream_token = self.config.stream_token
            self._session_expires_at = float("inf")
            return True

        if self._context is None:
            return False

        now = time.time()
        if not force and now < self._next_retry_at:
            return False

        session_url = f"{self.config.base_url}/api/model-run-stream/session"
        session_body = _build_stream_session_body(
            run_id=self._context.run_id,
            model_id=self.config.model_id,
            run_dir=self.config.run_dir or str(self._context.run_dir),
            euler_train_dir=self.config.euler_train_dir or str(self._context.runs_dir),
            stream_attach_token=(
                self.config.stream_attach_token
                or self._resolved_stream_attach_token
            ),
            slurm_job_id=_extract_slurm_job_id(self._context.meta),
            datasource_id=self.config.datasource_id,
            session_expires_in_sec=self.config.session_expires_in_sec,
        )

        try:
            payload = self._post_json(
                session_url,
                session_body,
                token=self.config.api_token,
            )
        except Exception as exc:
            self._schedule_retry()
            self._warn("Model-run stream session setup failed", exc)
            return False

        token = payload.get("token")
        if not isinstance(token, str) or not token.strip():
            self._schedule_retry()
            self._warn("Model-run stream session response did not contain a token")
            return False

        self._stream_token = token.strip()
        ingest_url = payload.get("ingestUrl")
        if isinstance(ingest_url, str) and ingest_url.strip():
            self._ingest_url = ingest_url.strip()
        else:
            self._ingest_url = f"{self.config.base_url}/api/model-run-stream/ingest"

        resolved_stream_attach_token = _extract_session_stream_attach_token(payload)
        if resolved_stream_attach_token is not None:
            self._resolved_stream_attach_token = resolved_stream_attach_token

        self._session_expires_at = _parse_timestamp(payload.get("expiresAt"))
        self._next_retry_at = 0.0
        return True

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        token: str | None,
    ) -> dict[str, Any]:
        return _post_json_request(
            url,
            payload,
            token=token,
            timeout_sec=self.config.timeout_sec,
        )

    def _trim_pending(self) -> None:
        overflow = len(self._pending) - self.config.max_pending_events
        if overflow <= 0:
            return
        del self._pending[:overflow]
        self._warn(
            f"Dropped {overflow} pending stream event(s) after queue overflow",
        )

    def _schedule_retry(self) -> None:
        backoff = max(self.config.flush_interval_sec, 2.0)
        self._next_retry_at = time.time() + backoff

    def _warn(self, message: str, exc: Exception | None = None) -> None:
        detail = f"{message}: {exc}" if exc is not None else message
        now = time.time()
        if self._last_warning is not None:
            previous_detail, previous_at = self._last_warning
            if previous_detail == detail and now - previous_at < 30:
                return

        log.warning(detail)
        self._last_warning = (detail, now)


def coerce_output_stream(spec: Any) -> OutputStream | None:
    if spec is None:
        return None
    if isinstance(spec, OutputStream):
        return spec
    if isinstance(spec, EulerViewStreamConfig):
        return OutputStream([EulerViewStreamConsumer(spec)])
    if isinstance(spec, Mapping):
        return OutputStream([EulerViewStreamConsumer(_config_from_mapping(spec))])
    if _looks_like_consumer(spec):
        return OutputStream([spec])
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
        consumers: list[OutputStreamConsumer] = []
        for item in spec:
            stream = coerce_output_stream(item)
            if stream is None:
                continue
            consumers.extend(stream.consumers)
        return OutputStream(consumers)
    raise TypeError(
        "stream must be a mapping, EulerViewStreamConfig, consumer, or sequence of consumers",
    )


def check_stream_handshake(
    config: EulerViewStreamConfig | Mapping[str, Any],
    *,
    run_id: str | None = None,
    stream_attach_token: str | None = None,
    slurm_job_id: int | None = None,
    datasource_id: int | None = None,
    euler_train_dir: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    if isinstance(config, Mapping):
        resolved_config = _config_from_mapping(config)
    elif isinstance(config, EulerViewStreamConfig):
        resolved_config = config
    else:
        raise TypeError(
            "config must be a mapping or EulerViewStreamConfig",
        )

    if resolved_config.stream_token is not None:
        raise ValueError(
            "stream.stream_token bypasses the session handshake; a dry-run handshake requires api_token instead",
        )

    normalized_run_id = (
        _normalize_optional_text(run_id, field_name="run_id")
        if run_id is not None
        else _default_stream_check_run_id()
    )
    if normalized_run_id is None:
        normalized_run_id = _default_stream_check_run_id()
    normalized_stream_attach_token = (
        _normalize_optional_text(
            stream_attach_token,
            field_name="stream_attach_token",
        )
        if stream_attach_token is not None
        else None
    )
    if (
        normalized_stream_attach_token is not None
        and resolved_config.stream_attach_token is not None
        and normalized_stream_attach_token != resolved_config.stream_attach_token
    ):
        raise ValueError(
            "stream_attach_token argument does not match stream.stream_attach_token",
        )
    resolved_stream_attach_token = (
        normalized_stream_attach_token or resolved_config.stream_attach_token
    )
    if resolved_config.model_id is None and resolved_stream_attach_token is None:
        raise ValueError(
            "stream.model_id is required for handshake checks when stream_attach_token is not provided",
        )

    resolved_datasource_id = (
        _require_positive_int(
            datasource_id,
            field_name="datasource_id",
            allow_none=True,
        )
        if datasource_id is not None
        else resolved_config.datasource_id
    )
    resolved_slurm_job_id = (
        _require_positive_int(
            slurm_job_id,
            field_name="slurm_job_id",
            allow_none=True,
        )
        if slurm_job_id is not None
        else None
    )
    resolved_euler_train_dir = (
        _normalize_optional_text(
            euler_train_dir,
            field_name="euler_train_dir",
        )
        if euler_train_dir is not None
        else resolved_config.euler_train_dir
    )
    resolved_run_dir = (
        _normalize_optional_text(run_dir, field_name="run_dir")
        if run_dir is not None
        else resolved_config.run_dir
    )

    url = f"{resolved_config.base_url}/api/model-run-stream/check"
    payload = _build_stream_session_body(
        run_id=normalized_run_id,
        model_id=resolved_config.model_id,
        run_dir=resolved_run_dir,
        euler_train_dir=resolved_euler_train_dir,
        stream_attach_token=resolved_stream_attach_token,
        slurm_job_id=resolved_slurm_job_id,
        datasource_id=resolved_datasource_id,
        session_expires_in_sec=resolved_config.session_expires_in_sec,
    )
    return _post_json_request(
        url,
        payload,
        token=resolved_config.api_token,
        timeout_sec=resolved_config.timeout_sec,
    )


def _looks_like_consumer(value: Any) -> bool:
    return all(
        callable(getattr(value, name, None))
        for name in ("bind", "emit", "flush", "close")
    )


def _config_from_mapping(raw: Mapping[str, Any]) -> EulerViewStreamConfig:
    removed_fields = {
        "access_token": "api_token",
        "accessToken": "api_token",
        "launch_id": "stream_attach_token",
        "launchId": "stream_attach_token",
    }
    for field, replacement in removed_fields.items():
        if field in raw:
            raise ValueError(
                f"Unsupported stream config field {field!r}; use {replacement!r}",
            )

    provider = _mapping_get(raw, "provider", "kind", "type")
    if provider is not None:
        normalized = str(provider).strip().lower().replace("-", "_")
        if normalized not in {"euler_view", "eulerview"}:
            raise ValueError(
                f"Unsupported stream provider {provider!r}; expected 'euler_view'",
            )

    return EulerViewStreamConfig(
        base_url=_mapping_get(raw, "base_url", "baseUrl"),
        model_id=_mapping_get(raw, "model_id", "modelId"),
        api_token=_mapping_get(raw, "api_token", "apiToken"),
        stream_token=_mapping_get(raw, "stream_token", "streamToken"),
        stream_attach_token=_mapping_get(
            raw,
            "stream_attach_token",
            "streamAttachToken",
        ),
        datasource_id=_mapping_get(raw, "datasource_id", "datasourceId"),
        euler_train_dir=_mapping_get(raw, "euler_train_dir", "eulerTrainDir"),
        run_dir=_mapping_get(raw, "run_dir", "runDir"),
        timeout_sec=_mapping_get(raw, "timeout_sec", "timeoutSec", default=10.0),
        batch_size=_mapping_get(raw, "batch_size", "batchSize", default=20),
        flush_interval_sec=_mapping_get(
            raw,
            "flush_interval_sec",
            "flushIntervalSec",
            default=2.0,
        ),
        max_pending_events=_mapping_get(
            raw,
            "max_pending_events",
            "maxPendingEvents",
            default=1000,
        ),
        session_expires_in_sec=_mapping_get(
            raw,
            "session_expires_in_sec",
            "sessionExpiresInSec",
        ),
    )


def _mapping_get(
    mapping: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _extract_slurm_job_id(meta: Mapping[str, Any] | None) -> int | None:
    if not isinstance(meta, Mapping):
        return None
    slurm = meta.get("slurm")
    if not isinstance(slurm, Mapping):
        return None
    raw = slurm.get("job_id")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() and raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = int(text, 10)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _extract_session_stream_attach_token(payload: Mapping[str, Any]) -> str | None:
    run = payload.get("run")
    if not isinstance(run, Mapping):
        return None
    stream_attach_token = run.get("streamAttachToken")
    if not isinstance(stream_attach_token, str):
        return None
    normalized = stream_attach_token.strip()
    return normalized or None


def _build_stream_session_body(
    *,
    run_id: str,
    model_id: int | None,
    run_dir: str | None,
    euler_train_dir: str | None,
    stream_attach_token: str | None,
    slurm_job_id: int | None,
    datasource_id: int | None,
    session_expires_in_sec: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"runId": run_id}
    if model_id is not None:
        body["modelId"] = model_id
    if run_dir is not None:
        body["runDir"] = run_dir
    if euler_train_dir is not None:
        body["eulerTrainDir"] = euler_train_dir
    if stream_attach_token is not None:
        body["streamAttachToken"] = stream_attach_token
    if slurm_job_id is not None:
        body["slurmJobId"] = slurm_job_id
    if datasource_id is not None:
        body["datasourceId"] = datasource_id
    if session_expires_in_sec is not None:
        body["expiresInSec"] = session_expires_in_sec
    return body


def _default_stream_check_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"stream-check-{timestamp}"


def _normalize_optional_text(
    value: Any,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _require_positive_int(
    value: Any,
    *,
    field_name: str,
    allow_none: bool = False,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} is required")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_positive_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _require_non_negative_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return float(value)


def _parse_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value.strip():
        return time.time() + 300

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00"),
        ).timestamp()
    except ValueError:
        return time.time() + 300


def _extract_http_error_message(error: HTTPError) -> str:
    try:
        raw = error.read()
    except OSError:
        raw = b""

    if raw:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                return f"HTTP {error.code}: {text}"
        else:
            if isinstance(payload, dict):
                body_error = payload.get("error")
                if isinstance(body_error, str) and body_error.strip():
                    return f"HTTP {error.code}: {body_error.strip()}"

    return f"HTTP {error.code}: {error.reason}"


def _post_json_request(
    url: str,
    payload: dict[str, Any],
    *,
    token: str | None,
    timeout_sec: float,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        url,
        data=json.dumps(payload, default=_json_default).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_sec) as response:
            raw = response.read()
    except HTTPError as exc:
        message = _extract_http_error_message(exc)
        raise RuntimeError(message) from exc
    except URLError as exc:
        reason = exc.reason if exc.reason else str(exc)
        raise RuntimeError(str(reason)) from exc
    except OSError as exc:
        raise RuntimeError(str(exc)) from exc

    if not raw:
        return {}

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON response from stream API") from exc

    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError("Unexpected stream API response payload")


def _to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


__all__ = [
    "EulerViewStreamConfig",
    "EulerViewStreamConsumer",
    "OutputStream",
    "OutputStreamConsumer",
    "StreamContext",
    "check_stream_handshake",
    "coerce_output_stream",
]
