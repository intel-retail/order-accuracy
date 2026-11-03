"""QSR agent contract for Order Accuracy.

The Central QSR Agent's built-in ``accuracy`` capability reads
``get_order_accuracy_context`` and, after an operator approves, calls
``request_remake``. Both are answered from this service's durable event log.
Shared verbatim by take-away (``core/agent_contract.py``) and dine-in
(``agent_contract.py``).

A remake request is recorded as a ``remake_requested`` event, which reaches
the kitchen through ``MCP_WEBHOOK_URL`` like every other event.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

OUTCOMES = ("order_validated", "order_failed")
REMAKE_EVENT = "remake_requested"
ISSUE_FIELDS = (
    ("missing_items", "missing_item"),
    ("extra_items", "extra_item"),
    ("quantity_mismatch", "quantity_mismatch"),
    ("quantity_mismatches", "quantity_mismatch"),
)
MAX_REASON_CHARS = 500


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")


def _issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    for field, kind in ISSUE_FIELDS:
        for item in payload.get(field) or []:
            detail = item if isinstance(item, dict) else {"name": str(item)}
            issues.append({**detail, "type": kind})
    return issues


def _live(events: Iterable[Any]) -> list[Any]:
    # Synthetic baseline history ("seed:" ref ids) is not a real order to act on.
    return sorted((e for e in events if not str(e.ref_id).startswith("seed:")), key=lambda e: e.ts_ms)


def latest_orders(events: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Latest validation outcome per order id, with any remake already requested."""
    orders: dict[str, dict[str, Any]] = {}
    for event in _live(events):
        order_id = event.payload.get("order_id")
        if not order_id:
            continue
        if event.event_type == REMAKE_EVENT:
            if order_id in orders:
                orders[order_id]["remake_requested_at"] = _iso(event.ts_ms)
            continue
        if event.event_type not in OUTCOMES:
            continue
        failed = event.event_type == "order_failed"
        orders.pop(order_id, None)  # re-insert so dict order follows the latest outcome
        orders[order_id] = {
            "order_id": order_id,
            "station_id": event.payload.get("station") or "unknown",
            "result": "flagged" if failed else "accurate",
            "issues": _issues(event.payload) if failed else [],
            "observed_at": _iso(event.ts_ms),
            **({"accuracy_score": event.payload["accuracy_score"]}
               if isinstance(event.payload.get("accuracy_score"), (int, float)) else {}),
        }
    return orders


def accuracy_context(events: Iterable[Any], store_id: str, venue: str, source: str,
                     window_minutes: int = 30, limit: int = 20) -> dict[str, Any]:
    """Snapshot in the shape the QSR agent's ``accuracy`` capability validates against."""
    events = list(events)
    now_ms = int(time.time() * 1000)
    since = now_ms - window_minutes * 60_000
    stations: dict[str, dict[str, Any]] = {}
    observed = flagged = 0
    for event in _live(events):
        if event.event_type not in OUTCOMES or event.ts_ms < since:
            continue
        station = stations.setdefault(event.payload.get("station") or "unknown",
                                      {"type": venue, "orders": 0, "flagged": 0})
        station["orders"] += 1
        observed += 1
        if event.event_type == "order_failed":
            station["flagged"] += 1
            flagged += 1
    recent = list(latest_orders(events).values())[-limit:]
    for order in recent:
        order["venue"] = venue
    return {
        "schema_version": "1.0",
        "source": source,
        "restaurant": {"id": store_id},
        "summary": {
            "window_minutes": window_minutes,
            "orders_observed": observed,
            "orders_accurate": observed - flagged,
            "orders_flagged": flagged,
            "accuracy_rate": round((observed - flagged) / observed, 3) if observed else None,
        },
        "stations": [{"id": name, **counts} for name, counts in sorted(stations.items())],
        "recent_orders": recent,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


def register(svc: Any, read_events: Callable[[], list], emit: Callable[[str, dict, str], Any],
             store_id: str, venue: str) -> None:
    """Add the agent's context read and remake action to ``svc``."""
    label = venue.replace("_", "-")
    svc.register_event_type(REMAKE_EVENT, {"remake_id": "str", "order_id": "str", "station": "str",
                                           "issues": "list[dict]", "reason": "str"})

    @svc.read_tool(
        "get_order_accuracy_context",
        description=(
            f"Current {label} order-accuracy snapshot for the QSR agent: accuracy over the last 30 "
            "minutes, per-station totals, and recent orders with their result (flagged/accurate) and "
            "issues (missing, extra, quantity mismatch)."
        ),
        schema={},
    )
    def get_order_accuracy_context() -> dict[str, Any]:
        return accuracy_context(read_events(), store_id, venue, svc.service)

    @svc.act_tool(
        "request_remake",
        description=(
            f"Ask the {label} line to remake one of its flagged orders (orders listed by this "
            f"service's get_order_accuracy_context). Only {label} orders whose latest validation "
            "failed can be remade; the request is logged and sent to the kitchen webhook."
        ),
        schema={"order_id": "str", "reason": "str"},
    )
    def request_remake(order_id: str, reason: str) -> dict[str, Any]:
        reason = (reason or "").strip()[:MAX_REASON_CHARS]
        if not reason:
            raise ValueError("reason is required")
        order = latest_orders(read_events()).get(order_id)
        if order is None or order["result"] != "flagged":
            # Two venues expose the same tool; point the caller at the other one.
            other = "dine-in" if label == "take-away" else "take-away"
            raise ValueError(
                f"order {order_id} is not a flagged {label} order. If it is a {other} order, call "
                f"request_remake on the order-accuracy-{other} service instead.")
        remake_id = f"remake-{uuid.uuid4().hex[:12]}"
        payload = {"remake_id": remake_id, "order_id": order_id, "station": order["station_id"],
                   "issues": order["issues"], "reason": reason}
        if emit(REMAKE_EVENT, payload, f"remake:{order_id}:{remake_id}") is None:
            raise RuntimeError("remake request could not be recorded")
        return {"executed": True, "remake_id": remake_id, "order_id": order_id,
                "station_id": order["station_id"], "reason": reason,
                "requested_at": datetime.now(timezone.utc).isoformat()}
