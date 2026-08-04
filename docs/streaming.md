# Euler View streaming

Streaming is an optional mirror of a normal file-backed run. euler-train writes
the local artifact first, then emits the corresponding event. A network or
Euler View failure therefore cannot remove the on-disk record or raise from the
training loop.

## Session handshake

The usual configuration exchanges a long-lived API credential for a
short-lived ingest token:

```python
import os
import euler_train

run = euler_train.init(
    dir="experiments/dehazing",
    config=config,
    stream={
        "base_url": os.environ["EULER_VIEW_BASE_URL"],
        "api_token": os.environ["EULER_VIEW_API_TOKEN"],
        "stream_attach_token": os.environ.get(
            "EULER_VIEW_STREAM_ATTACH_TOKEN"
        ),
        "model_id": 42,
    },
)
```

`model_id` is optional when `stream_attach_token` is present. Remove `None`
values from a serialized config if another system rejects them; euler-train
itself accepts `None` for optional fields.

The first forced flush performs this sequence:

1. `POST /api/model-run-stream/session` with the run ID and available
   attachment hints.
2. Authenticate that request with `api_token`.
3. Cache the returned short-lived token, ingest URL and canonical attachment
   token.
4. Send queued events to the returned ingest URL.
5. Refresh the session when its token approaches expiry.

The init, checkpoint, output snapshot and finish events force a flush. Metric
events flush when the batch or time threshold is reached.

## Attachment choices

### Explicit attachment token

`stream_attach_token` is the preferred correlation key. It lets Euler View
attach this producer to a particular launch without exposing an internal
launch ID to the training code. It is not an authentication credential, so an
`api_token` is still required for the session handshake.

```python
stream={
    "base_url": "https://view.example.com",
    "api_token": os.environ["EULER_VIEW_API_TOKEN"],
    "stream_attach_token": os.environ["EULER_VIEW_STREAM_ATTACH_TOKEN"],
}
```

### Model and SLURM fallback

Without an explicit attachment token, supply `model_id`. When the run is under
SLURM, euler-train also sends the job ID captured in `meta.json`; Euler View can
use the pair to find the corresponding launch.

```python
stream={
    "base_url": "https://view.example.com",
    "api_token": os.environ["EULER_VIEW_API_TOKEN"],
    "model_id": 42,
}
```

Streaming can still create or update the remote run when launch matching is
unresolved; only the launch-to-run association is affected.

### Pre-issued ingest token

If the launcher already provides a short-lived ingest token, pass
`stream_token`. This bypasses session negotiation, so neither `api_token` nor
`model_id` is needed:

```python
stream={
    "base_url": "https://view.example.com",
    "stream_token": os.environ["EULER_VIEW_STREAM_TOKEN"],
}
```

The token is sent directly to `/api/model-run-stream/ingest` and is treated as
non-expiring by the client. Its real lifetime must therefore cover the run, or
the launcher must start a new producer with a fresh token.

## Configuration reference

The mapping form and `EulerViewStreamConfig(...)` accept the same canonical
fields:

| Field | Default | Purpose |
|---|---:|---|
| `base_url` | required | Euler View origin, without a required trailing slash. |
| `api_token` | — | Credential for creating/refreshing an ingest session. |
| `stream_token` | — | Pre-issued ingest credential; skips the handshake. |
| `stream_attach_token` | — | Opaque launch correlation token. |
| `model_id` | — | Model namespace, required when neither attach nor ingest token is supplied. |
| `datasource_id` | — | Optional datasource hint for the remote run. |
| `euler_train_dir` | derived | Override the runs-directory path sent in a handshake. |
| `run_dir` | derived | Override the concrete run path sent in a handshake. |
| `session_expires_in_sec` | server default | Requested lifetime for a negotiated ingest token. |
| `timeout_sec` | `10.0` | HTTP request timeout. |
| `batch_size` | `20` | Maximum events in one ingest request. |
| `flush_interval_sec` | `2.0` | Metric flush interval and minimum retry backoff. |
| `max_pending_events` | `1000` | In-memory queue bound while the remote destination is unavailable. |

Positive numeric IDs and sizes are validated when the config is constructed.
The flush interval may be zero; other timing and size values must be positive.

## Failure behavior

The built-in consumer is intentionally best-effort:

- serialization, handshake and ingest errors are logged as warnings;
- repeated identical warnings are rate-limited for 30 seconds;
- failed events remain queued for a later flush;
- retries wait at least two seconds; and
- if the queue exceeds `max_pending_events`, the oldest events are dropped and
  a warning reports the loss.

`finish()` attempts one final forced flush, then closes the consumer. There is
no disk-backed retry queue; use the local run files to backfill a remote system
after a prolonged outage.

## Connectivity check

Use the packaged dry-run command on the same node and network path as the real
training job:

```bash
euler-train.stream.check \
  --api-url https://view.example.com \
  --api-key "$EULER_VIEW_API_TOKEN" \
  --stream-attach-token "$EULER_VIEW_STREAM_ATTACH_TOKEN"
```

Or test model/SLURM matching:

```bash
euler-train.stream.check \
  --api-url https://view.example.com \
  --api-key "$EULER_VIEW_API_TOKEN" \
  --model-id 42 \
  --slurm-job-id 123456
```

Useful options include:

| Flag | Purpose |
|---|---|
| `--run-id` | Override the generated `stream-check-*` ID. |
| `--stream-attach-token` | Test explicit attachment resolution. |
| `--model-id` | Select the model namespace for the fallback path. |
| `--slurm-job-id` | Exercise SLURM launch matching. |
| `--datasource-id` | Include a datasource hint. |
| `--euler-train-dir`, `--run-dir` | Include path hints matching a real session. |
| `--timeout-sec` | Override the HTTP timeout. |
| `--json` | Print the raw response for automation. |

The check calls `POST /api/model-run-stream/check`. It verifies the request
shape, credentials and resolution path without starting a run or minting an
ingest token. The command exits nonzero on a failed request.

## Custom consumers

`stream=` may also be a consumer object, an `OutputStream`, or a sequence of
consumers. A consumer implements four methods:

```python
class Consumer:
    def bind(self, context):
        """Receive run paths, config, meta and provenance snapshots."""

    def emit(self, event):
        """Receive one JSON-compatible event mapping."""

    def flush(self):
        """Commit pending events."""

    def close(self):
        """Flush and release resources."""
```

```python
run = euler_train.init(
    dir="experiments/dehazing",
    stream=[my_database_consumer, my_message_queue_consumer],
)
```

`OutputStream` isolates consumer exceptions in the same best-effort style as
the built-in HTTP consumer. Consumers should still keep their own methods
short, bounded and idempotent because they execute in the training process.

## Credential hygiene

Do not put tokens in `config` or `meta`: those mappings are persisted locally
and included in the stream initialization event. Read credentials from the
environment or a secret manager and pass them only inside `stream`. Treat
attachment and ingest tokens as secrets even when their sole intended purpose
is correlation or short-lived ingestion.
