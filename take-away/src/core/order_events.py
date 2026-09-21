"""Order Accuracy domain events — Take-away (Issue #102).

Event-creation pattern mirrored from the alert event in the storewide-loss-
prevention Person-of-Interest (POI) application:

  * ``backend/domain/entities/match_result.py::AlertPayload`` — a dedicated
    ``@dataclass`` for the event, not a bare dict assembled inline.
  * ``backend/service/alert_service.py::AlertService.create_alert_payload()``
    — a single factory function that builds that entity from the
    business-logic result.

This module gives Order Accuracy the same shape: ``OrderEvent`` is the typed
entity (analogous to ``AlertPayload``) and ``create_order_event()`` is the
factory (analogous to ``create_alert_payload()``). ``core.mcp_service`` calls
the factory, then hands ``event.to_dict()``/``event.ref_id`` to the MCP
Service SDK's ``ServiceServer.emit()`` — which is the durable-log/publish
step (the SDK equivalent of POI's ``EventBus.publish()`` +
``EventRepository.store_alert()`` combined into one call).

One deliberate difference from POI, kept for a documented reason: POI's
``alert_id`` doubles as the only identifier — each alert dispatch is
distinct, no de-duplication is required at that layer. Order Accuracy's MCP
read tools require a *stable, idempotent* key per completed order/run so a
retried/replayed emit does not create a duplicate durable-log row. That key
remains ``ref_id`` (``"{station}:{order_id}:{run_number}"``), unchanged from
before and still what ``ServiceServer.emit()`` dedups on. The new
``event_id`` below is an *additional*, POI-style human-traceable identifier
(UTC timestamp + short uuid4 suffix) carried in the payload for
observability — it does not replace or affect ``ref_id``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _new_event_id(event_type: str) -> str:
    """Human-traceable, sortable, unique-per-emission identifier.

    Same construction as POI's ``alert_id``
    (``f"alert-{utc_ts}-{poi_id}-{uuid4().hex[:8]}"``): a UTC timestamp
    prefix plus a short random suffix. Distinct from ``ref_id``, which is
    the SDK-level idempotency/dedup key.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{event_type}-{ts}-{uuid.uuid4().hex[:8]}"


def _build_reason(validation: dict[str, Any]) -> str:
    parts = []
    missing = validation.get("missing") or []
    extra = validation.get("extra") or []
    qty = validation.get("quantity_mismatch") or []
    if missing:
        parts.append(f"missing:{len(missing)}")
    if extra:
        parts.append(f"extra:{len(extra)}")
    if qty:
        parts.append(f"quantity_mismatch:{len(qty)}")
    return ",".join(parts) if parts else "unknown"


@dataclass
class OrderEvent:
    """A single Take-away domain event (``order_validated``/``order_failed``).

    Analogous to POI's ``AlertPayload``: a typed entity produced by a
    factory function (``create_order_event``), not a dict built ad hoc at
    the call site.
    """

    event_type: str  # "order_validated" | "order_failed"
    ref_id: str  # idempotency key: "{station}:{order_id}:{run_number}"
    order_id: str
    station: str
    run_number: int
    num_frames: Optional[int] = None
    inference_time_sec: Optional[float] = None
    missing_items: list = field(default_factory=list)
    extra_items: list = field(default_factory=list)
    quantity_mismatch: list = field(default_factory=list)
    reason: str = ""
    event_id: str = field(init=False)
    # ISO-8601 UTC timestamp stamped on creation — same style/field name as
    # POI's ``AlertPayload.dispatched_at`` (default_factory, millisecond
    # precision). This is in addition to (not a replacement for) the SDK
    # envelope's own ``ts_ms``.
    dispatched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def __post_init__(self) -> None:
        self.event_id = _new_event_id(self.event_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the flat payload handed to ``ServiceServer.emit()``.

        Mirrors ``AlertPayload.to_dict()``: base fields always present,
        failure-specific fields only added for ``order_failed``.
        """
        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "dispatched_at": self.dispatched_at,
            "order_id": self.order_id,
            "station": self.station,
            "run_number": self.run_number,
            "num_frames": self.num_frames,
            "inference_time_sec": self.inference_time_sec,
        }
        if self.event_type == "order_failed":
            payload.update(
                {
                    "missing_items": self.missing_items,
                    "extra_items": self.extra_items,
                    "quantity_mismatch": self.quantity_mismatch,
                    "reason": self.reason,
                }
            )
        return payload


def create_order_event(result: dict[str, Any]) -> OrderEvent:
    """Build an ``OrderEvent`` from one completed order result.

    Mirrors POI's ``AlertService.create_alert_payload()``: a single factory
    function, called once per completed detection/validation, that
    assembles the typed event entity from the business-logic result.
    ``result`` is the same dict ``core.order_results.add_result`` already
    stores — this function does not decide validity, it only records the
    outcome that ``core.vlm_service``/``core.validation_agent`` already
    determined.
    """
    order_id = result.get("order_id")
    station = result.get("station_id")
    run_number = result.get("run_number", 1)
    status = result.get("status")
    event_type = "order_validated" if status == "validated" else "order_failed"
    ref_id = f"{station}:{order_id}:{run_number}"

    validation = result.get("validation", {}) or {}
    return OrderEvent(
        event_type=event_type,
        ref_id=ref_id,
        order_id=order_id,
        station=station,
        run_number=run_number,
        num_frames=result.get("num_frames"),
        inference_time_sec=result.get("inference_time_sec"),
        missing_items=validation.get("missing", []),
        extra_items=validation.get("extra", []),
        quantity_mismatch=validation.get("quantity_mismatch", []),
        reason=_build_reason(validation) if event_type == "order_failed" else "",
    )
