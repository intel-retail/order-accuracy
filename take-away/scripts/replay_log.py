#!/usr/bin/env python3
"""Replay a recorded day from the Order Accuracy durable event log (Take-away).

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
  ``mcp_sensor``'s ``SQLiteLog``/``JSONLFileLog`` (vendored locally — see
  ``src/core/mcp_sensor.py``) both store each event's full envelope
  (including its original ``ts_ms``) verbatim at append time (see
  ``core/mcp_service.py``'s ``emit_order_result()`` -> ``svc.emit()``).
  ``replay()`` reads records back in ``seq`` order without regenerating or
  mutating any field, so running this script twice against the same,
  unmodified log always prints byte-identical output.

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
``/results/order_accuracy_events.db`` (the container path the ``results/``
bind mount maps to — see docker-compose.yaml; matches
``core/mcp_service.py``'s own default so this works with zero flags when
run inside the container with the default configuration).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


def _load_mcp_sensor():
    """Load ``mcp_sensor.py`` directly from its file, bypassing ``core``'s
    package ``__init__.py``.

    ``mcp_sensor.py`` lives inside the ``core`` package (alongside
    application modules such as ``pipeline_runner``/``validation_agent``),
    but is itself a self-contained, dependency-free module (no relative
    imports). A plain ``from core.mcp_sensor import ...`` would first run
    ``core/__init__.py``, which eagerly imports unrelated app dependencies
    (``minio``) and, when ``USE_SEMANTIC_SERVICE=true`` (the Compose
    default), performs a network health check — before this script even
    parses its arguments. Loading the file directly avoids all of that.

    Supports the source-checkout layout (``take-away/src/core/``) and both
    Docker image copies of this script: ``/app/scripts/`` (``APP_DIR`` is
    ``/app``, so ``core`` is a sibling at ``/app/core/``) and the one
    actually invoked by ``make replay``, ``/scripts/`` (``APP_DIR`` is
    ``/``, so ``core`` is *not* a sibling — ``/app/core/`` is checked as a
    fixed fallback, matching the Dockerfile's ``COPY src/core/ /app/core/``
    and ``ENV PYTHONPATH=/app:...``).
    """
    candidates = (
        APP_DIR / "src" / "core" / "mcp_sensor.py",
        APP_DIR / "core" / "mcp_sensor.py",
        Path("/app/core/mcp_sensor.py"),
    )
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_oa_mcp_sensor", candidate)
            module = importlib.util.module_from_spec(spec)
            # dataclasses' internals resolve forward references via
            # sys.modules[cls.__module__], so the module must be registered
            # there before exec_module() runs its class bodies.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(f"mcp_sensor.py not found; checked {', '.join(str(c) for c in candidates)}")


_mcp_sensor = _load_mcp_sensor()
JSONLFileLog = _mcp_sensor.JSONLFileLog
SQLiteLog = _mcp_sensor.SQLiteLog

# Matches core/mcp_service.py's own RESULTS_DIR/MCP_LOG_BACKEND default
# convention ("/results", the path docker-compose's ``results/`` bind mount
# maps to inside the container) so this script finds the live log with zero
# extra flags, whichever backend the running service is configured with.
_DEFAULT_LOG_BACKEND = os.getenv("MCP_LOG_BACKEND", "sqlite")
_DEFAULT_LOG_PATH = (
    "/results/order_accuracy_events.db"
    if _DEFAULT_LOG_BACKEND != "jsonl"
    else "/results/order_accuracy_events.jsonl"
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
