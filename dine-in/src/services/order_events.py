"""Order Accuracy domain events — Dine-in (Issue #102).

Same event-creation pattern used in the Take-away application
(``take-away/src/core/order_events.py``), which itself mirrors the alert
event in the storewide-loss-prevention Person-of-Interest (POI) application:

  * A dedicated ``@dataclass`` entity (``OrderEvent``) instead of a bare dict
    assembled inline at the call site.
  * A single factory function (``create_order_event()``) that builds that
    entity from the business-logic result — mirrors POI's
    ``AlertService.create_alert_payload()`` / Take-away's
    ``create_order_event()``.
  * A human-traceable ``event_id`` (UTC timestamp + short uuid4 suffix),
    separate from the idempotency key.
  * An explicit ISO-8601 UTC ``dispatched_at`` timestamp.
  * A ``to_dict()`` method for serialization to the MCP Service SDK payload.

Dine-in has no "station"/"run_number" concept the way Take-away does (one
video → one order per station run). Instead:

  * ``station`` is populated from the order's ``table_number`` (the closest
    Dine-in equivalent of a station — see ``configs/orders.json``), falling
    back to ``"unknown"`` if not provided.
  * The idempotency key (``ref_id``) is ``"{order_id}:{image_id}"`` — each
    plate image submitted for a given order is treated as one discrete,
    dedupable validation event. This is a deliberate, simpler choice than
    Take-away's run-counter scheme: Dine-in has no existing run-counter
    state to build on, and re-validating the exact same order+image pair is
    the only case that should collapse to a single durable event.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _new_event_id(event_type: str) -> str:
    """Human-traceable, sortable, unique-per-emission identifier.

    Same construction as Take-away's/POI's id pattern
    (``f"{prefix}-{utc_ts}-{uuid4().hex[:8]}"``). Distinct from ``ref_id``,
    which is the SDK-level idempotency/dedup key.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{event_type}-{ts}-{uuid.uuid4().hex[:8]}"


def _build_reason(missing_items: list, extra_items: list, quantity_mismatches: list) -> str:
    parts = []
    if missing_items:
        parts.append(f"missing:{len(missing_items)}")
    if extra_items:
        parts.append(f"extra:{len(extra_items)}")
    if quantity_mismatches:
        parts.append(f"quantity_mismatch:{len(quantity_mismatches)}")
    return ",".join(parts) if parts else "unknown"


@dataclass
class OrderEvent:
    """A single Dine-in domain event (``order_validated``/``order_failed``).

    Analogous to Take-away's ``OrderEvent``/POI's ``AlertPayload``: a typed
    entity produced by a factory function (``create_order_event``), not a
    dict built ad hoc at the call site.
    """

    event_type: str  # "order_validated" | "order_failed"
    ref_id: str  # idempotency key: "{order_id}:{image_id}"
    order_id: str
    station: str  # populated from table_number, "unknown" if absent
    image_id: str
    accuracy_score: float = 0.0
    missing_items: list = field(default_factory=list)
    extra_items: list = field(default_factory=list)
    quantity_mismatches: list = field(default_factory=list)
    reason: str = ""
    event_id: str = field(init=False)
    # ISO-8601 UTC timestamp stamped on creation — same style/field name as
    # Take-away's/POI's ``dispatched_at`` (default_factory, millisecond
    # precision). This is in addition to (not a replacement for) the SDK
    # envelope's own ``ts_ms``.
    dispatched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def __post_init__(self) -> None:
        self.event_id = _new_event_id(self.event_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the flat payload handed to ``ServiceServer.emit()``.

        Mirrors Take-away's/POI's ``to_dict()``: base fields always present,
        failure-specific fields only added for ``order_failed``.
        """
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "dispatched_at": self.dispatched_at,
            "order_id": self.order_id,
            "station": self.station,
            "image_id": self.image_id,
            "accuracy_score": self.accuracy_score,
        }
        if self.event_type == "order_failed":
            payload.update(
                {
                    "missing_items": self.missing_items,
                    "extra_items": self.extra_items,
                    "quantity_mismatches": self.quantity_mismatches,
                    "reason": self.reason,
                }
            )
        return payload


def create_order_event(result: dict[str, Any]) -> OrderEvent:
    """Build an ``OrderEvent`` from one completed plate-validation result.

    Mirrors Take-away's/POI's factory function: a single function, called
    once per completed ``ValidationService.validate_plate()`` call, that
    assembles the typed event entity from the business-logic result.

    ``result`` is a plain dict built by the API layer (``api.py``) right
    after ``validate_plate()`` returns — this function does not decide
    validity, it only records the outcome
    ``services.validation_service.ValidationService`` already determined.
    Expected keys: ``order_id``, ``station`` (table_number or "unknown"),
    ``image_id``, ``order_complete``, ``accuracy_score``, ``missing_items``,
    ``extra_items``, ``quantity_mismatches``.
    """
    order_id = result.get("order_id")
    station = result.get("station") or "unknown"
    image_id = result.get("image_id")
    order_complete = bool(result.get("order_complete"))
    event_type = "order_validated" if order_complete else "order_failed"
    ref_id = f"{order_id}:{image_id}"

    missing_items = result.get("missing_items", []) or []
    extra_items = result.get("extra_items", []) or []
    quantity_mismatches = result.get("quantity_mismatches", []) or []

    return OrderEvent(
        event_type=event_type,
        ref_id=ref_id,
        order_id=order_id,
        station=station,
        image_id=image_id,
        accuracy_score=result.get("accuracy_score", 0.0),
        missing_items=missing_items,
        extra_items=extra_items,
        quantity_mismatches=quantity_mismatches,
        reason=(
            _build_reason(missing_items, extra_items, quantity_mismatches)
            if event_type == "order_failed"
            else ""
        ),
    )
