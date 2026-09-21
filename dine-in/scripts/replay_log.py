#!/usr/bin/env python3
"""Replay a recorded day from the Order Accuracy durable event log (Dine-in).

Developer/validation workflow for Issue #102 AC5 ("a recorded day replays
identically from the log"). This intentionally does **not** expose replay as
an MCP tool: Order Accuracy's MCP surface is read/detect only (Issue #102 —
"A: none (read/detect only)"), and replay is a debugging/audit workflow, not
a runtime capability offered to an agent.

What this does:
  - Opens the *existing* durable event log read-only in effect (it only
    ever calls the SDK's own ``DurableLog.replay()``/``read()``, never
    ``append()``), using whichever backend (SQLite or JSONL) the running
    service is configured with (``MCP_LOG_BACKEND``/``--log-backend``).
  - Re-emits every event envelope, in its original insertion (``seq``) order,
    to stdout.
  - Never touches delivery/webhook/hub — it is pure log replay, not event
    (re)delivery.

Why this is deterministic:
  ``mcp_service_sdk``'s ``SQLiteLog``/``JSONLFileLog`` both store each
  event's full envelope (including its original ``ts_ms``) verbatim at
  append time (see ``src/mcp_service.py``'s ``emit_order_result()`` ->
  ``svc.emit()``). ``replay()`` reads records back in ``seq`` order without
  regenerating or mutating any field, so running this script twice against
  the same, unmodified log always prints byte-identical output.

Why this is safe:
  This script never calls ``svc.emit()`` or ``log.append()``, so it cannot
  create duplicate or new events in the original log, and it has no side
  effects on delivery, business logic, or the running application.

Usage:
    python scripts/replay_log.py [--log-path PATH] [--log-backend sqlite|jsonl] [--format summary|json]

    # or, via the documented Makefile target (run inside the container,
    # where MCP_LOG_PATH/MCP_LOG_BACKEND already point at the live log):
    make replay
    make exec CMD="python scripts/replay_log.py --format json"

Defaults to $MCP_LOG_PATH / $MCP_LOG_BACKEND if set (the same variables the
running application uses), otherwise falls back to the SQLite backend at
``<repo>/dine-in/results/order_accuracy_events.db`` (the host-side path the
``results/`` bind mount maps to ``/app/results`` inside the container — see
docker-compose.yaml).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]

from mcp_service_sdk.log import JSONLFileLog, SQLiteLog  # noqa: E402

_DEFAULT_LOG_BACKEND = os.getenv("MCP_LOG_BACKEND", "sqlite")
_DEFAULT_LOG_PATH = str(
    APP_DIR
    / "results"
    / ("order_accuracy_events.jsonl" if _DEFAULT_LOG_BACKEND == "jsonl" else "order_accuracy_events.db")
)


def _open_log(log_path: str, backend: str):
    """Open the existing durable log read-only in effect (replay-only)."""
    if backend == "jsonl":
        return JSONLFileLog(path=log_path, service="replay-readonly")
    return SQLiteLog(path=log_path, service="replay-readonly")


def replay(log_path: str, fmt: str = "summary", backend: str = "sqlite") -> int:
    """Replay every event in ``log_path`` in original order. Read-only."""
    if not Path(log_path).exists():
        print(f"No event log found at {log_path!r}. Nothing to replay.", file=sys.stderr)
        return 1

    log = _open_log(log_path, backend)
    count = 0
    try:
        for event in log.replay(from_seq=0):
            count += 1
            if fmt == "json":
                print(event.to_json())
            else:
                print(
                    f"[{count:04d}] ts_ms={event.ts_ms} "
                    f"event_type={event.event_type} ref_id={event.ref_id} "
                    f"order_id={event.payload.get('order_id')} "
                    f"station={event.payload.get('station')}"
                )
    finally:
        close = getattr(log, "close", None)
        if callable(close):
            close()

    print(
        f"\nReplayed {count} event(s) from {log_path} in original order.",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the Order Accuracy durable event log in original order (read-only, no side effects)."
    )
    parser.add_argument(
        "--log-path",
        default=os.getenv("MCP_LOG_PATH", _DEFAULT_LOG_PATH),
        help="Path to the durable event log "
        "(default: $MCP_LOG_PATH, else %(default)s)",
    )
    parser.add_argument(
        "--log-backend",
        choices=["sqlite", "jsonl"],
        default=_DEFAULT_LOG_BACKEND if _DEFAULT_LOG_BACKEND in ("sqlite", "jsonl") else "sqlite",
        help="Durable log backend to open the path as "
        "(default: $MCP_LOG_BACKEND, else %(default)s) — must match the "
        "running service's MCP_LOG_BACKEND.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json"],
        default="summary",
        help="Output format: human-readable summary (default) or one JSON envelope per line",
    )
    args = parser.parse_args()
    return replay(args.log_path, args.format, args.log_backend)


if __name__ == "__main__":
    raise SystemExit(main())
