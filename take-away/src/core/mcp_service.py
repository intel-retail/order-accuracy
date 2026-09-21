"""MCP integration for Order Accuracy — Take-away (Issue #102).

Order Accuracy is a **sensor**: it detects and reports, it does not act. This
module is the *only* place that talks to ``mcp_service_sdk``. It:

  1. Declares the two Take-away domain events (``order_validated`` /
     ``order_failed``) that the existing validation pipeline already decides
     (see ``core.vlm_service._run_vlm_internal``, which sets
     ``status: "validated" | "mismatch"``).
  2. Exposes read-only MCP tools required by the ticket: rework rate
     (vs. baseline), order/validation history, and per-station pass/fail
     totals — all answered from the SDK's durable log, not from the
     in-memory ``core.order_results`` tracker, so a restart loses nothing.
  3. Registers **no** action tools. Per Issue #102: "Order Accuracy is a
     sensor (no runtime action)" / "A: none (read/detect only)".

Nothing here duplicates business logic: validation itself still happens in
``core.validation_agent``/``core.vlm_service``; this module only records the
*outcome* as a durable, replayable event and answers questions about it.

Event creation (the ``OrderEvent`` entity + ``create_order_event()`` factory
in ``core.order_events``) follows the same pattern as the alert event in the
storewide-loss-prevention Person-of-Interest (POI) application
(``AlertPayload`` + ``AlertService.create_alert_payload()``): a dedicated,
typed event entity built by a single factory function, not a dict assembled
inline. See ``core/order_events.py`` for the full comparison/rationale.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from mcp_service_sdk import ServiceConfig, ServiceServer

from .order_events import create_order_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-driven, same convention as the rest of the service)
# ---------------------------------------------------------------------------

# Master flag: "a single flag turns the event and subscription layer off for
# clean benchmark runs" (Issue #100 AC11 / Issue #102 "benchmark flag present").
MCP_SERVICE_ENABLED = os.getenv("MCP_SERVICE_ENABLED", "true").lower() == "true"

STORE_ID = os.getenv("STORE_ID", "store-001")

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# One shared durable log per service (not per-station), so per-station totals
# can be answered from a single source of truth even when scaled across
# multiple station containers sharing the same /results volume.
MCP_LOG_BACKEND = os.getenv("MCP_LOG_BACKEND", "sqlite")  # sqlite | jsonl | memory
MCP_LOG_PATH = os.getenv(
    "MCP_LOG_PATH",
    str(RESULTS_DIR / "order_accuracy_events.db")
    if MCP_LOG_BACKEND != "jsonl"
    else str(RESULTS_DIR / "order_accuracy_events.jsonl"),
)

# Delivery: off by default (safe / benchmark-clean). Set MCP_WEBHOOK_URL to
# push events to an agent inbox / event hub callback.
MCP_WEBHOOK_URL = os.getenv("MCP_WEBHOOK_URL")

MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8010"))

_EVENT_TYPES = ("order_validated", "order_failed")


def _build_service() -> ServiceServer:
    cfg = ServiceConfig(
        service="order_accuracy",
        store_id=STORE_ID,
        log_backend=MCP_LOG_BACKEND if MCP_SERVICE_ENABLED else "memory",
        log_path=MCP_LOG_PATH,
        delivery="webhook" if MCP_WEBHOOK_URL else "off",
        webhook_url=MCP_WEBHOOK_URL,
    )
    return ServiceServer.from_config(cfg)


svc = _build_service()

# -- 1. declare event types (feeds `describe`) -----------------------------

svc.register_event_type(
    "order_validated",
    schema={
        "order_id": "str",
        "station": "str",
        "run_number": "int",
        "num_frames": "int",
        "inference_time_sec": "float",
    },
)

svc.register_event_type(
    "order_failed",
    schema={
        "order_id": "str",
        "station": "str",
        "run_number": "int",
        "missing_items": "list[dict]",
        "extra_items": "list[dict]",
        "quantity_mismatch": "list[dict]",
        "reason": "str",
        "num_frames": "int",
        "inference_time_sec": "float",
    },
)


# ---------------------------------------------------------------------------
# Emission — called from the existing validation pipeline once per order
# ---------------------------------------------------------------------------


def emit_order_result(result: dict[str, Any]):
    """Emit ``order_validated``/``order_failed`` for one completed order.

    ``result`` is the same dict ``core.order_results.add_result`` already
    stores (it must be called first so ``run_number``/``completed_at`` are
    stamped on it). Only real pass/fail outcomes are emitted — transient
    statuses such as ``error``/``no_frames`` are not domain events.

    Event creation follows the same pattern as the alert event in the
    Person-of-Interest (POI) application: a dedicated event entity built by
    a single factory function (``create_order_event`` in
    ``core.order_events``, analogous to POI's
    ``AlertService.create_alert_payload()`` building an ``AlertPayload``),
    rather than a dict assembled inline here. This function's only
    remaining job is the terminal-status gate and handing the built event
    off to the MCP Service SDK for durable logging (``svc.emit()`` —
    the SDK equivalent of POI's ``EventBus.publish()`` +
    ``EventRepository.store_alert()``).

    Returns the emitted ``EventEnvelope``, or ``None`` if the service is
    disabled or the result has no meaningful outcome yet.
    """
    if not MCP_SERVICE_ENABLED:
        return None

    status = result.get("status")
    if status not in ("validated", "mismatch"):
        logger.debug("[MCP] Skipping emit for non-terminal status=%s", status)
        return None

    order_event = create_order_event(result)
    event = svc.emit(
        order_event.event_type, order_event.to_dict(), ref_id=order_event.ref_id
    )
    logger.info(
        "[MCP] Emitted %s order_id=%s station=%s ref_id=%s event_id=%s",
        order_event.event_type,
        order_event.order_id,
        order_event.station,
        order_event.ref_id,
        order_event.event_id,
    )
    return event


# ---------------------------------------------------------------------------
# Read tools (Issue #102: "R: rework rate by period vs. baseline;
# order/validation history; per-station pass/fail totals")
# ---------------------------------------------------------------------------


def _all_events(station: Optional[str] = None, limit: int = 100_000):
    events = []
    for event_type in _EVENT_TYPES:
        events.extend(svc.log.read(event_type=event_type, limit=limit))
    if station:
        events = [e for e in events if e.payload.get("station") == station]
    events.sort(key=lambda e: e.ts_ms)
    return events


def _day_bounds_ms(period: str) -> tuple[int, int]:
    """Return ``[start_ms, end_ms)`` epoch-millisecond bounds for a period.

    ``period`` is one of ``"today"``, ``"yesterday"``, ``"all"``, or an
    explicit ``YYYY-MM-DD`` date (UTC calendar day).
    """
    now = time.time()
    day_s = 86400
    today_start = int(now // day_s) * day_s
    if period == "all":
        return 0, int(now * 1000) + 1
    if period == "today":
        start = today_start
    elif period == "yesterday":
        start = today_start - day_s
    else:
        # explicit YYYY-MM-DD
        struct = time.strptime(period, "%Y-%m-%d")
        start = int(time.mktime(struct))
    return start * 1000, (start + day_s) * 1000


def _rate_for(events: list, start_ms: int, end_ms: int) -> dict[str, Any]:
    in_range = [e for e in events if start_ms <= e.ts_ms < end_ms]
    total = len(in_range)
    failed = sum(1 for e in in_range if e.event_type == "order_failed")
    return {
        "orders_seen": total,
        "orders_failed": failed,
        "rework_rate": round(failed / total, 3) if total else 0.0,
    }


@svc.read_tool(
    "get_rework_rate",
    description=(
        "Rework rate (share of orders that failed validation) for a period, "
        "compared against a baseline period. Optionally filtered by station."
    ),
    schema={
        "period": "str (today|yesterday|all|YYYY-MM-DD, default 'today')",
        "station": "str|None",
        "baseline_period": "str|None (default 'yesterday')",
    },
)
def get_rework_rate(
    period: str = "today",
    station: str | None = None,
    baseline_period: str | None = "yesterday",
) -> dict[str, Any]:
    events = _all_events(station=station)
    start_ms, end_ms = _day_bounds_ms(period)
    current = _rate_for(events, start_ms, end_ms)

    result: dict[str, Any] = {
        "period": period,
        "station": station or "all",
        **current,
    }

    if baseline_period and baseline_period != period:
        b_start, b_end = _day_bounds_ms(baseline_period)
        baseline = _rate_for(events, b_start, b_end)
        result["baseline_period"] = baseline_period
        result["baseline_rework_rate"] = baseline["rework_rate"]
        result["baseline_orders_seen"] = baseline["orders_seen"]
        result["delta_vs_baseline"] = round(
            current["rework_rate"] - baseline["rework_rate"], 3
        )

    return result


@svc.read_tool(
    "get_order_history",
    description="Order validation history (order_validated/order_failed events), oldest first.",
    schema={
        "limit": "int (default 50)",
        "station": "str|None",
        "order_id": "str|None",
    },
)
def get_order_history(
    limit: int = 50,
    station: str | None = None,
    order_id: str | None = None,
) -> list[dict[str, Any]]:
    events = _all_events(station=station, limit=max(limit, 1) * 10 or 100_000)
    if order_id:
        events = [e for e in events if e.payload.get("order_id") == order_id]
    events = events[-limit:] if limit else events
    return [
        {
            "event_type": e.event_type,
            "status": "validated" if e.event_type == "order_validated" else "failed",
            "ref_id": e.ref_id,
            "ts_ms": e.ts_ms,
            **e.payload,
        }
        for e in events
    ]


@svc.read_tool(
    "get_station_totals",
    description="Per-station pass/fail totals and rework rate.",
    schema={"station": "str|None"},
)
def get_station_totals(station: str | None = None) -> dict[str, Any]:
    events = _all_events(station=station)
    totals: dict[str, dict[str, Any]] = {}
    for e in events:
        st = e.payload.get("station") or "unknown"
        bucket = totals.setdefault(st, {"validated": 0, "failed": 0})
        if e.event_type == "order_validated":
            bucket["validated"] += 1
        else:
            bucket["failed"] += 1

    for st, bucket in totals.items():
        total = bucket["validated"] + bucket["failed"]
        bucket["total"] = total
        bucket["rework_rate"] = round(bucket["failed"] / total, 3) if total else 0.0

    if station:
        return totals.get(station, {"validated": 0, "failed": 0, "total": 0, "rework_rate": 0.0})
    return totals


# ---------------------------------------------------------------------------
# Baseline history seeding — "ships preloaded with enough history that
# comparative behaviour works; restores in one command" (Issue #102 AC).
# ---------------------------------------------------------------------------

# Deterministic synthetic outcomes for the "yesterday" baseline. Pattern is
# intentionally mixed (~25% fail rate) so get_rework_rate has a meaningful
# baseline to compare "today" against out of the box.
_SEED_PATTERN = [
    ("station_1", "validated"),
    ("station_1", "validated"),
    ("station_1", "mismatch"),
    ("station_1", "validated"),
    ("station_2", "validated"),
    ("station_2", "mismatch"),
    ("station_2", "validated"),
    ("station_1", "validated"),
    ("station_2", "validated"),
    ("station_1", "mismatch"),
]


def seed_history_if_empty() -> int:
    """Seed one day of synthetic baseline history if the log is empty.

    Idempotent: uses fixed ``ref_id``s, so re-running (or multiple station
    containers racing on startup against the shared log) never duplicates
    events. Timestamps are anchored to "yesterday" relative to *now*, so the
    seeded data always serves as a valid ``baseline_period="yesterday"``
    comparison, however many days pass since this file was written.
    """
    if not MCP_SERVICE_ENABLED:
        return 0
    existing = svc.log.read(limit=1)
    if existing:
        return 0

    start_ms, _ = _day_bounds_ms("yesterday")
    seeded = 0
    for i, (station, status) in enumerate(_SEED_PATTERN):
        order_id = f"seed-{i:03d}"
        ts_offset_ms = i * 60_000  # spread across the seeded day
        ref_id = f"seed:{order_id}"
        payload = {
            "order_id": order_id,
            "station": station,
            "run_number": 1,
            "num_frames": 3,
            "inference_time_sec": 1.5,
        }
        event_type = "order_validated"
        if status == "mismatch":
            event_type = "order_failed"
            payload.update(
                {
                    "missing_items": [{"name": "seed-item", "quantity": 1}],
                    "extra_items": [],
                    "quantity_mismatch": [],
                    "reason": "missing:1",
                }
            )
        event = svc.emit(event_type, payload, ref_id=ref_id)
        # Backfill the timestamp to fall within "yesterday" (emit() stamps
        # "now" by default; the log stores whatever ts_ms the envelope
        # carries only when constructed with it, so re-append explicitly).
        _backfill_seed_ts(ref_id, start_ms + ts_offset_ms)
        seeded += 1
    logger.info("[MCP] Seeded %d baseline history events for 'yesterday'", seeded)
    return seeded


def _backfill_seed_ts(ref_id: str, ts_ms: int) -> None:
    """Best-effort: rewrite a just-appended seed event's ts_ms in-place.

    SQLiteLog/JSONLFileLog only expose append/read/replay, so we go through
    the underlying connection when available (sqlite) and otherwise accept
    the emit-time timestamp (memory/jsonl backends used mainly for tests).
    """
    conn = getattr(svc.log, "_conn", None)
    if conn is None:
        return
    try:
        with conn:
            conn.execute(
                "UPDATE events SET ts_ms = ?, envelope = json_set(envelope, '$.ts_ms', ?) "
                "WHERE ref_id = ?",
                (ts_ms, ts_ms, ref_id),
            )
    except Exception as exc:  # pragma: no cover - defensive, non-fatal
        logger.debug("[MCP] Could not backfill seed timestamp for %s: %s", ref_id, exc)


if MCP_SERVICE_ENABLED:
    seed_history_if_empty()
