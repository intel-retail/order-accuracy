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

import calendar
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from mcp_service_sdk import Delivery, ServiceConfig, ServiceServer, WebhookSink

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

# JSONLFileLog keeps its dedup ("_seen") set and sequence counter in
# process-local memory with no inter-process lock around its append (see the
# SDK's log.py). That is safe for one process, but Take-away supports
# multi-station scaling (``docker compose up -d --scale order-accuracy=N``)
# where several containers share this same log file/volume — under jsonl
# that can silently produce duplicate ref_ids or colliding sequence numbers.
# Default to refusing jsonl for that shared scenario and falling back to the
# cross-process-safe sqlite backend; set MCP_ALLOW_UNSAFE_JSONL=true to force
# jsonl anyway for single-instance/dev/test use.
MCP_ALLOW_UNSAFE_JSONL = os.getenv("MCP_ALLOW_UNSAFE_JSONL", "false").lower() == "true"
if MCP_LOG_BACKEND == "jsonl" and not MCP_ALLOW_UNSAFE_JSONL:
    logger.warning(
        "[MCP] MCP_LOG_BACKEND=jsonl is not safe for multi-container/shared "
        "deployments (no inter-process lock around JSONLFileLog's append); "
        "falling back to 'sqlite'. Set MCP_ALLOW_UNSAFE_JSONL=true to force "
        "jsonl anyway (single-instance/dev/test only)."
    )
    MCP_LOG_BACKEND = "sqlite"

MCP_LOG_PATH = os.getenv(
    "MCP_LOG_PATH",
    str(RESULTS_DIR / "order_accuracy_events.db")
    if MCP_LOG_BACKEND != "jsonl"
    else str(RESULTS_DIR / "order_accuracy_events.jsonl"),
)
if MCP_LOG_BACKEND == "sqlite" and MCP_LOG_PATH.endswith(".jsonl"):
    # An explicit MCP_LOG_PATH=*.jsonl combined with the jsonl->sqlite
    # fallback above would otherwise open a JSONL file through SQLiteLog.
    MCP_LOG_PATH = str(RESULTS_DIR / "order_accuracy_events.db")

# Delivery: off by default (safe / benchmark-clean). Set MCP_WEBHOOK_URL to
# push events to an agent inbox / event hub callback.
MCP_WEBHOOK_URL = os.getenv("MCP_WEBHOOK_URL")

MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8010"))

_EVENT_TYPES = ("order_validated", "order_failed")


def _build_service() -> ServiceServer:
    # Delivery is intentionally NOT wired through ServiceConfig here: the
    # SDK's own webhook delivery is synchronous inside svc.emit() (retries
    # 3x with 5s timeouts + backoff), which would block the calling
    # validation/order request for up to ~16.5s if the callback is slow or
    # down. Instead this service always builds with delivery="off" (an
    # instant no-op) and, when MCP_WEBHOOK_URL is set, dispatches the SAME
    # SDK WebhookSink from a background thread after the event is already
    # durably logged (see _safe_emit / _ASYNC_WEBHOOK_DELIVERY below). The
    # durable log write itself remains synchronous — only the outbound HTTP
    # push is moved off the request's critical path.
    cfg = ServiceConfig(
        service="order_accuracy",
        store_id=STORE_ID,
        log_backend=MCP_LOG_BACKEND if MCP_SERVICE_ENABLED else "memory",
        log_path=MCP_LOG_PATH,
        delivery="off",
        webhook_url=None,
    )
    return ServiceServer.from_config(cfg)


svc = _build_service()

# Async webhook delivery (see _build_service docstring above): built once,
# reused for every event's background dispatch. None when no webhook is
# configured, matching the previous "off" delivery behavior exactly.
_ASYNC_WEBHOOK_DELIVERY = Delivery(sinks=[WebhookSink(MCP_WEBHOOK_URL)]) if MCP_WEBHOOK_URL else None

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


def _seed_run_counters_from_durable_log() -> None:
    """Restore ``core.order_results``' in-memory run-number counters at startup.

    ``core.order_results.StationResults.order_run_counts`` (the source of
    ``result["run_number"]``, which in turn feeds this module's ``ref_id``)
    is in-memory only and always starts at ``{}`` on process start. Left
    as-is, the first order processed for a given station/order_id after a
    restart gets ``run_number=1`` again, even if that exact
    ``"{station}:{order_id}:1"`` ref_id was already durably recorded before
    the restart — ``svc.emit()``'s idempotent dedup then silently treats the
    new, distinct run as a duplicate of the old one and drops it.

    Fix: seed each station's counter from the MAX ``run_number`` already
    present in the durable MCP event log for that (station, order_id) pair,
    once, at import time — before any new order is processed. This does not
    change the per-call increment logic in ``order_results.add_result()``
    (still ``+= 1`` per real call, so intra-process retry/idempotency
    semantics are unaffected) — it only fixes the starting point so a
    restart doesn't roll it back to zero.
    """
    if not MCP_SERVICE_ENABLED:
        return
    try:
        from core.order_results import _get_station
    except ImportError:  # pragma: no cover - defensive, order_results always present
        logger.debug("[MCP] core.order_results not importable; skipping run-counter seed")
        return

    max_run_number: dict[tuple[str, str], int] = {}
    for event in _all_events():
        station = event.payload.get("station")
        order_id = event.payload.get("order_id")
        run_number = event.payload.get("run_number")
        if not station or not order_id or not isinstance(run_number, int):
            continue
        key = (station, order_id)
        if run_number > max_run_number.get(key, 0):
            max_run_number[key] = run_number

    for (station, order_id), run_number in max_run_number.items():
        station_state = _get_station(station)
        with station_state.lock:
            if station_state.order_run_counts.get(order_id, 0) < run_number:
                station_state.order_run_counts[order_id] = run_number
    if max_run_number:
        logger.info(
            "[MCP] Restored run-number counters for %d order(s) from durable log",
            len(max_run_number),
        )


_DEAD_LETTER_PATH = RESULTS_DIR / "mcp_dead_letter.jsonl"


def _write_dead_letter(event_type: str, payload: dict[str, Any], ref_id: str, error: BaseException) -> None:
    """Best-effort fallback so a durable-log write failure never silently and
    irrecoverably loses the event: append it (plus the error) to a local
    recovery file instead of only logging it, so it can be reconciled/
    replayed manually later.
    """
    record = {
        "ts_ms": int(time.time() * 1000),
        "event_type": event_type,
        "ref_id": ref_id,
        "payload": payload,
        "error": repr(error),
    }
    try:
        with open(_DEAD_LETTER_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        logger.critical(
            "[MCP] Could not write dead-letter record for ref_id=%s either; "
            "event is only visible in the application logs above.",
            ref_id,
            exc_info=True,
        )


def _find_existing_event(event_type: str, ref_id: str):
    """Return the already-durably-recorded event for ``ref_id``, if any."""
    for e in _read_entire_log():
        if e.event_type == event_type and e.ref_id == ref_id:
            return e
    return None


def _safe_emit(event_type: str, payload: dict[str, Any], ref_id: str):
    """``svc.emit()``, made idempotent-safe end-to-end (not just at the log).

    ``ServiceServer.emit()`` always builds a brand-new envelope and
    unconditionally dispatches it to delivery, even when the durable log
    treats ``ref_id`` as an existing row and silently no-ops the insert —
    so a retry of the same validation would otherwise fire a second,
    distinct webhook for an event already recorded. Checking for an
    existing ``ref_id`` first and short-circuiting (no re-emit, no re-
    dispatch) avoids that duplicate delivery.

    Multiple station containers can also share one SQLite log file (see the
    ``MCP_LOG_BACKEND`` note above). ``SQLiteLog.append()`` does
    SELECT-then-INSERT; if two processes race on the exact same ``ref_id``
    in the (small) window between our own existence check and our own
    insert, the losing INSERT raises ``sqlite3.IntegrityError`` even though
    the event IS durably recorded (by the winner) — that is the intended
    idempotent outcome, not a lost event, so we look the row back up
    instead of letting the exception propagate into the validation request.

    Any other failure to durably log the event (disk full, locked file,
    etc.) is never silently dropped: the event is appended to a local
    dead-letter file for manual recovery before returning ``None``, so the
    caller's existing "log and continue" handling still can't lose data.

    Also fans a genuinely new event out to the async webhook sink (if
    configured) on a background thread, off the caller's critical path —
    see ``_build_service``'s docstring for why delivery is not wired
    through ``svc`` itself.
    """
    existing = _find_existing_event(event_type, ref_id)
    if existing is not None:
        logger.info(
            "[MCP] ref_id=%s already durably recorded; skipping duplicate "
            "emit/dispatch",
            ref_id,
        )
        return existing

    try:
        event = svc.emit(event_type, payload, ref_id=ref_id)
    except sqlite3.IntegrityError:
        logger.info(
            "[MCP] ref_id=%s already recorded by a concurrent writer; "
            "treating as idempotent no-op",
            ref_id,
        )
        return _find_existing_event(event_type, ref_id)
    except Exception as exc:
        _write_dead_letter(event_type, payload, ref_id, exc)
        logger.error(
            "[MCP] Failed to durably log ref_id=%s; wrote to dead-letter "
            "log (%s) for manual recovery: %s",
            ref_id,
            _DEAD_LETTER_PATH,
            exc,
            exc_info=True,
        )
        return None

    if _ASYNC_WEBHOOK_DELIVERY is not None:
        threading.Thread(
            target=_ASYNC_WEBHOOK_DELIVERY.dispatch,
            args=(event,),
            daemon=True,
            name="mcp-webhook-delivery",
        ).start()
    return event


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
    event = _safe_emit(
        order_event.event_type, order_event.to_dict(), order_event.ref_id
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

_LOG_READ_PAGE = 5000


def _read_entire_log() -> list:
    """Read the FULL durable log (all event types), oldest → newest.

    ``DurableLog.read()`` takes a bounded ``limit`` and does not hand the
    row's ``seq`` back to the caller (``EventEnvelope`` has no ``seq``
    field), so a single fixed-size call — e.g. the previous
    ``limit=100_000`` — silently truncates once the log grows past that
    cap, and per-``event_type`` pagination can't reliably resume either
    (matching rows can be sparse across a much larger scanned range).

    Paging over the *unfiltered* log side-steps both problems: both
    backends are strictly append-only with a gapless, contiguous ``seq``
    (SQLiteLog's autoincrement PK; JSONLFileLog's local counter), so with no
    ``event_type`` filter each page's length is exactly the number of seq
    values consumed — advancing ``since_seq`` by ``len(page)`` is always
    correct, with no risk of skipping or re-reading rows, regardless of how
    large the log grows.
    """
    events: list = []
    since_seq = 0
    while True:
        page = svc.log.read(since_seq=since_seq, limit=_LOG_READ_PAGE)
        if not page:
            break
        events.extend(page)
        since_seq += len(page)
        if len(page) < _LOG_READ_PAGE:
            break
    return events


def _all_events(station: Optional[str] = None):
    events = [e for e in _read_entire_log() if e.event_type in _EVENT_TYPES]
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
        # explicit YYYY-MM-DD (UTC calendar day, per the docstring above —
        # time.mktime() would interpret the parsed date in the host's local
        # timezone, shifting the requested day on a non-UTC host).
        struct = time.strptime(period, "%Y-%m-%d")
        start = calendar.timegm(struct)
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
    if limit < 0:
        raise ValueError("limit must be >= 0 (0 means unlimited)")
    events = _all_events(station=station)
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
        event = _safe_emit(event_type, payload, ref_id)
        # Backfill the timestamp to fall within "yesterday" (emit() stamps
        # "now" by default; the log stores whatever ts_ms the envelope
        # carries only when constructed with it, so re-append explicitly).
        _backfill_seed_ts(ref_id, start_ms + ts_offset_ms)
        seeded += 1
    logger.info("[MCP] Seeded %d baseline history events for 'yesterday'", seeded)
    return seeded


def _backfill_seed_ts(ref_id: str, ts_ms: int) -> None:
    """Best-effort: rewrite a just-appended seed event's ts_ms in-place.

    Needed so seeded "yesterday" baseline events keep their backdated
    timestamp: ``svc.emit()``/``DurableLog.append()`` always stamp "now",
    and ``ts_ms`` as actually stored in the log is what period bucketing
    (``_day_bounds_ms``/``get_rework_rate``) reads. Supports both durable
    backends this service can be configured with (``MCP_LOG_BACKEND``):
    SQLite (via its connection) and JSONL (by rewriting the matching
    record in place). The in-memory backend (tests only) has neither and
    is left as emit-time, which is fine there.
    """
    conn = getattr(svc.log, "_conn", None)
    if conn is not None:
        try:
            with conn:
                conn.execute(
                    "UPDATE events SET ts_ms = ?, envelope = json_set(envelope, '$.ts_ms', ?) "
                    "WHERE ref_id = ?",
                    (ts_ms, ts_ms, ref_id),
                )
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.debug("[MCP] Could not backfill seed timestamp for %s: %s", ref_id, exc)
        return

    path = getattr(svc.log, "_path", None)
    if path is None:
        return  # in-memory backend (tests) — emit-time timestamp is fine there

    lock = getattr(svc.log, "_lock", None)
    try:
        import json as _json

        def _rewrite() -> None:
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            records = [_json.loads(ln) for ln in lines]
            changed = False
            for rec in records:
                if rec.get("event", {}).get("ref_id") == ref_id:
                    rec["event"]["ts_ms"] = ts_ms
                    changed = True
            if changed:
                with open(path, "w", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(_json.dumps(rec, separators=(",", ":")) + "\n")

        if lock is not None:
            with lock:
                _rewrite()
        else:
            _rewrite()
    except Exception as exc:  # pragma: no cover - defensive, non-fatal
        logger.debug("[MCP] Could not backfill seed timestamp for %s (jsonl): %s", ref_id, exc)


if MCP_SERVICE_ENABLED:
    seed_history_if_empty()
    _seed_run_counters_from_durable_log()
