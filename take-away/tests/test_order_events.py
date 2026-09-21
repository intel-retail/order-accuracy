"""Tests for the Order Accuracy event entity/factory (Issue #102 — Take-away).

Covers the POI-alert-event-pattern parity introduced in
``core.order_events``: a dedicated ``OrderEvent`` dataclass built by a single
``create_order_event()`` factory (analogous to POI's ``AlertPayload`` +
``AlertService.create_alert_payload()``), including the traceable
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

from core.order_events import OrderEvent, create_order_event  # noqa: E402


def test_create_order_event_validated():
    result = {
        "order_id": "925",
        "station_id": "station_1",
        "status": "validated",
        "num_frames": 3,
        "inference_time_sec": 2.1,
        "run_number": 1,
    }
    event = create_order_event(result)

    assert isinstance(event, OrderEvent)
    assert event.event_type == "order_validated"
    assert event.ref_id == "station_1:925:1"
    assert event.order_id == "925"
    assert event.station == "station_1"
    assert event.run_number == 1


def test_create_order_event_failed_has_reason_and_items():
    result = {
        "order_id": "539",
        "station_id": "station_2",
        "status": "mismatch",
        "validation": {
            "missing": [{"name": "apple", "quantity": 1}],
            "extra": [{"name": "banana", "quantity": 1}],
            "quantity_mismatch": [],
        },
        "num_frames": 4,
        "inference_time_sec": 3.0,
        "run_number": 1,
    }
    event = create_order_event(result)

    assert event.event_type == "order_failed"
    assert event.missing_items == [{"name": "apple", "quantity": 1}]
    assert event.extra_items == [{"name": "banana", "quantity": 1}]
    assert event.reason == "missing:1,extra:1"


def test_event_id_is_unique_and_traceable_like_poi_alert_id():
    """Mirrors POI's alert_id pattern: f'{prefix}-{utc_ts}-{uuid4hex8}'."""
    result = {
        "order_id": "1",
        "station_id": "station_1",
        "status": "validated",
        "run_number": 1,
    }
    event_a = create_order_event(result)
    event_b = create_order_event(result)

    # Same ref_id (idempotency key, deterministic) ...
    assert event_a.ref_id == event_b.ref_id == "station_1:1:1"
    # ... but distinct event_id (traceable identifier, unique per creation).
    assert event_a.event_id != event_b.event_id
    assert event_a.event_id.startswith("order_validated-")
    # 8-char uuid4 hex suffix after the timestamp segment.
    suffix = event_a.event_id.rsplit("-", 1)[-1]
    assert len(suffix) == 8


def test_dispatched_at_is_iso8601_utc():
    result = {
        "order_id": "1",
        "station_id": "station_1",
        "status": "validated",
        "run_number": 1,
    }
    event = create_order_event(result)
    # Must be parseable as an ISO-8601 timestamp (POI's AlertPayload style).
    parsed = datetime.fromisoformat(event.dispatched_at)
    assert parsed.tzinfo is not None


def test_to_dict_shape_matches_expected_mcp_payload():
    result = {
        "order_id": "925",
        "station_id": "station_1",
        "status": "validated",
        "num_frames": 3,
        "inference_time_sec": 2.1,
        "run_number": 1,
    }
    event = create_order_event(result)
    payload = event.to_dict()

    # Required MCP payload fields (unchanged from before the refactor).
    assert payload["order_id"] == "925"
    assert payload["station"] == "station_1"
    assert payload["run_number"] == 1
    assert payload["num_frames"] == 3
    assert payload["inference_time_sec"] == 2.1
    # New POI-pattern fields present on every event.
    assert "event_id" in payload
    assert "dispatched_at" in payload
    # Failure-only fields absent for a validated order.
    assert "missing_items" not in payload
    assert "reason" not in payload


def test_to_dict_includes_failure_fields_only_for_order_failed():
    result = {
        "order_id": "539",
        "station_id": "station_2",
        "status": "mismatch",
        "validation": {"missing": [], "extra": [], "quantity_mismatch": [{"name": "banana", "expected": 2, "detected": 3}]},
        "run_number": 1,
    }
    event = create_order_event(result)
    payload = event.to_dict()

    assert payload["reason"] == "quantity_mismatch:1"
    assert payload["quantity_mismatch"] == [{"name": "banana", "expected": 2, "detected": 3}]
