"""Tests for the Order Accuracy event entity/factory (Issue #102 — Dine-in).

Covers the POI-alert-event-pattern parity introduced in
``services.order_events`` (same pattern as Take-away's
``core.order_events``): a dedicated ``OrderEvent`` dataclass built by a
single ``create_order_event()`` factory, including the traceable
``event_id``, the explicit ``dispatched_at`` ISO-8601 UTC timestamp, and the
``to_dict()`` serialization used as the MCP Service SDK emit payload.

Run with: python -m pytest tests/test_order_events.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from services.order_events import OrderEvent, create_order_event  # noqa: E402


def test_create_order_event_validated():
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    event = create_order_event(result)

    assert isinstance(event, OrderEvent)
    assert event.event_type == "order_validated"
    assert event.ref_id == "MCD-1001:MCD-1001"
    assert event.order_id == "MCD-1001"
    assert event.station == "T12"
    assert event.image_id == "MCD-1001"


def test_create_order_event_failed_has_reason_and_items():
    result = {
        "order_id": "MCD-1002",
        "station": "T18",
        "image_id": "MCD-1002",
        "order_complete": False,
        "accuracy_score": 0.4,
        "missing_items": [{"name": "Veg Burger", "quantity": 1}],
        "extra_items": [{"name": "Soft Drink", "quantity": 1}],
        "quantity_mismatches": [],
    }
    event = create_order_event(result)

    assert event.event_type == "order_failed"
    assert event.missing_items == [{"name": "Veg Burger", "quantity": 1}]
    assert event.extra_items == [{"name": "Soft Drink", "quantity": 1}]
    assert event.reason == "missing:1,extra:1"


def test_station_defaults_to_unknown_when_absent():
    result = {
        "order_id": "MCD-1003",
        "image_id": "MCD-1003",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    event = create_order_event(result)
    assert event.station == "unknown"


def test_event_id_is_unique_and_traceable():
    """Mirrors Take-away's/POI's id pattern: f'{prefix}-{utc_ts}-{uuid4hex8}'."""
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    event_a = create_order_event(result)
    event_b = create_order_event(result)

    # Same ref_id (idempotency key, deterministic) ...
    assert event_a.ref_id == event_b.ref_id == "MCD-1001:MCD-1001"
    # ... but distinct event_id (traceable identifier, unique per creation).
    assert event_a.event_id != event_b.event_id
    assert event_a.event_id.startswith("order_validated-")
    suffix = event_a.event_id.rsplit("-", 1)[-1]
    assert len(suffix) == 8


def test_dispatched_at_is_iso8601_utc():
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    event = create_order_event(result)
    parsed = datetime.fromisoformat(event.dispatched_at)
    assert parsed.tzinfo is not None


def test_to_dict_shape_matches_expected_mcp_payload():
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 0.92,
    }
    event = create_order_event(result)
    payload = event.to_dict()

    assert payload["order_id"] == "MCD-1001"
    assert payload["station"] == "T12"
    assert payload["image_id"] == "MCD-1001"
    assert payload["accuracy_score"] == 0.92
    assert "event_id" in payload
    assert "dispatched_at" in payload
    # Failure-only fields absent for a validated order.
    assert "missing_items" not in payload
    assert "reason" not in payload


def test_to_dict_includes_failure_fields_only_for_order_failed():
    result = {
        "order_id": "MCD-1002",
        "station": "T18",
        "image_id": "MCD-1002",
        "order_complete": False,
        "accuracy_score": 0.5,
        "missing_items": [],
        "extra_items": [],
        "quantity_mismatches": [{"item": "Cheeseburger", "expected_quantity": 2, "detected_quantity": 1}],
    }
    event = create_order_event(result)
    payload = event.to_dict()

    assert payload["reason"] == "quantity_mismatch:1"
    assert payload["quantity_mismatches"] == [
        {"item": "Cheeseburger", "expected_quantity": 2, "detected_quantity": 1}
    ]
