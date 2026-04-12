from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .stream import check_stream_handshake


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euler-train.stream.check",
        description="Dry-run the Euler View stream handshake",
    )
    parser.add_argument("--api-url", required=True, help="Euler View base URL")
    parser.add_argument("--model-id", required=True, type=int, help="Euler View model ID")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Euler View API token used to authorize the handshake check",
    )
    parser.add_argument(
        "--run-id",
        help="Run ID to use for the dry-run handshake. Defaults to an ephemeral stream-check ID.",
    )
    parser.add_argument(
        "--stream-attach-token",
        help="Optional explicit launch attachment token to validate",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=int,
        help="Optional SLURM job ID to test the fallback matching path",
    )
    parser.add_argument(
        "--datasource-id",
        type=int,
        help="Optional datasource hint to include in the handshake payload",
    )
    parser.add_argument(
        "--euler-train-dir",
        help="Optional euler_train runs directory path to include in the handshake payload",
    )
    parser.add_argument(
        "--run-dir",
        help="Optional run directory path to include in the handshake payload",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response instead of a human-readable summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = check_stream_handshake(
            {
                "base_url": args.api_url,
                "model_id": args.model_id,
                "api_token": args.api_key,
                "stream_attach_token": args.stream_attach_token,
                "datasource_id": args.datasource_id,
                "euler_train_dir": args.euler_train_dir,
                "run_dir": args.run_dir,
                "timeout_sec": args.timeout_sec,
            },
            run_id=args.run_id,
            slurm_job_id=args.slurm_job_id,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        else:
            print(f"Handshake check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    run = result.get("run", {})
    print("Handshake check succeeded")
    print(f"  model_id: {run.get('modelId')}")
    print(f"  run_id: {run.get('runId')}")
    print(f"  resolution: {result.get('resolution')}")
    print(f"  stream_attach_token: {run.get('streamAttachToken')}")
    print(f"  ingest_url: {result.get('ingestUrl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
