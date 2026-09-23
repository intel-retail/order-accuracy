"""Tests for the MCP service integration (Issue #102 — Dine-in).

Covers: event schema/emission, idempotent replay, read tools (rework rate
vs. baseline, order history, per-station/table totals), durable persistence
across restarts, and no action tools exposed (Order Accuracy is a sensor).

Mirrors take-away/tests/test_mcp_service.py, adapted for Dine-in's flat
module layout (``mcp_service`` at ``src/`` top level, not under a ``core``
package) and its ``order_complete``/``accuracy_score`` result shape (no
``station_id``/``run_number`` — see ``services/order_events.py`` for why).

Run with: python -m pytest tests/test_mcp_service.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


def _reload_mcp_service(tmp_path, **env_overrides):
    """Import a fresh mcp_service with env vars applied and module-level
    caches cleared, so each test gets an isolated log file/service instance.
    """
    env = {
        "RESULTS_DIR": str(tmp_path),
        "MCP_LOG_PATH": str(tmp_path / "events.db"),
        "MCP_SERVICE_ENABLED": "true",
        "MCP_WEBHOOK_URL": "",
    }
    env.update(env_overrides)
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    sys.modules.pop("mcp_service", None)
    module = importlib.import_module("mcp_service")
    try:
        yield module
    finally:
        sys.modules.pop("mcp_service", None)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def mcp(tmp_path):
    yield from _reload_mcp_service(tmp_path)


@pytest.fixture
def mcp_disabled(tmp_path):
    yield from _reload_mcp_service(tmp_path, MCP_SERVICE_ENABLED="false")


# ---------------------------------------------------------------------------
# Event schema / emission
# ---------------------------------------------------------------------------


def test_seed_history_preloaded(mcp):
    """Ships preloaded with baseline history so comparative queries work.

    Mirrors take-away's test_seed_history_preloaded — verifies Issue #102 AC4
    ("ships preloaded with enough history that comparative behaviour works;
    restores in one command") for Dine-in.
    """
    events = mcp.svc.log.read(limit=1000)
    assert len(events) == len(mcp._SEED_PATTERN)
    assert {e.event_type for e in events} <= {"order_validated", "order_failed"}


def test_seed_history_is_idempotent_across_reimport(tmp_path):
    """Restarting the app (re-importing the module against the same log
    file) must not duplicate baseline events — the module-level
    ``if MCP_SERVICE_ENABLED: seed_history_if_empty()`` call only seeds when
    the log is empty, and re-running it a second time is a no-op.
    """
    gen1 = _reload_mcp_service(tmp_path)
    mcp1 = next(gen1)
    count_after_first_import = len(mcp1.svc.log.read(limit=1000))
    try:
        next(gen1)
    except StopIteration:
        pass

    gen2 = _reload_mcp_service(tmp_path)
    mcp2 = next(gen2)
    # Re-running seeding explicitly against the same (now non-empty) log
    # must be a no-op.
    seeded_again = mcp2.seed_history_if_empty()
    count_after_second_import = len(mcp2.svc.log.read(limit=1000))
    try:
        next(gen2)
    except StopIteration:
        pass

    assert seeded_again == 0
    assert count_after_second_import == count_after_first_import


def test_baseline_yesterday_vs_today_comparison(mcp):
    """The seeded 'yesterday' baseline must be usable by get_rework_rate as
    a real comparison point against 'today', per Issue #102 AC3."""
    # Emit one real "today" event on top of the seeded baseline.
    mcp.emit_order_result(
        {
            "order_id": "today-1",
            "station": "T1",
            "image_id": "today-1",
            "order_complete": True,
            "accuracy_score": 1.0,
        }
    )
    result = mcp.get_rework_rate(period="today", baseline_period="yesterday")
    assert result["baseline_period"] == "yesterday"
    # 10 seeded events yesterday, 3 of which are "mismatch" -> 0.3 rework rate.
    assert result["baseline_orders_seen"] == len(mcp._SEED_PATTERN)
    assert result["baseline_rework_rate"] == pytest.approx(0.3)
    assert result["orders_seen"] == 1
    assert result["rework_rate"] == 0.0


def test_emit_order_validated(mcp):
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    event = mcp.emit_order_result(result)
    assert event.event_type == "order_validated"
    assert event.payload["order_id"] == "MCD-1001"
    assert event.payload["station"] == "T12"
    assert event.ref_id == "MCD-1001:MCD-1001"


def test_emit_order_failed_payload_has_missing_extra_reason(mcp):
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
    event = mcp.emit_order_result(result)
    assert event.event_type == "order_failed"
    assert event.payload["missing_items"] == [{"name": "Veg Burger", "quantity": 1}]
    assert event.payload["extra_items"] == [{"name": "Soft Drink", "quantity": 1}]
    assert event.payload["reason"] == "missing:1,extra:1"


def test_emit_is_idempotent_on_same_ref_id(mcp):
    result = {
        "order_id": "MCD-1002",
        "station": "T18",
        "image_id": "MCD-1002",
        "order_complete": False,
        "accuracy_score": 0.4,
        "missing_items": [],
        "extra_items": [],
        "quantity_mismatches": [],
    }
    before = len(mcp.svc.log.read(limit=10_000))
    mcp.emit_order_result(result)
    after_first = len(mcp.svc.log.read(limit=10_000))
    mcp.emit_order_result(result)  # duplicate emit, same ref_id
    after_second = len(mcp.svc.log.read(limit=10_000))

    assert after_first == before + 1
    assert after_second == after_first  # no duplicate appended


def test_disabled_service_does_not_emit(mcp_disabled):
    result = {
        "order_id": "MCD-1001",
        "station": "T12",
        "image_id": "MCD-1001",
        "order_complete": True,
        "accuracy_score": 1.0,
    }
    assert mcp_disabled.emit_order_result(result) is None


# ---------------------------------------------------------------------------
# Durable persistence across "restart"
# ---------------------------------------------------------------------------


def test_log_survives_restart(tmp_path):
    gen1 = _reload_mcp_service(tmp_path)
    mcp1 = next(gen1)
    mcp1.emit_order_result(
        {
            "order_id": "MCD-1001",
            "station": "T12",
            "image_id": "MCD-1001",
            "order_complete": True,
            "accuracy_score": 1.0,
        }
    )
    count_before_restart = len(mcp1.svc.log.read(limit=10_000))
    gen1.close()  # simulate process shutdown (releases the module reference)

    # "Restart": re-import against the SAME on-disk log path/tmp_path.
    gen2 = _reload_mcp_service(tmp_path)
    mcp2 = next(gen2)
    count_after_restart = len(mcp2.svc.log.read(limit=10_000))

    assert count_after_restart == count_before_restart
    # Seeding must not duplicate history on restart (log already non-empty).
    assert count_after_restart == len(mcp2._SEED_PATTERN) + 1
    gen2.close()


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


def test_get_rework_rate_today(mcp):
    mcp.emit_order_result(
        {
            "order_id": "MCD-1001",
            "station": "T12",
            "image_id": "MCD-1001",
            "order_complete": True,
            "accuracy_score": 1.0,
        }
    )
    mcp.emit_order_result(
        {
            "order_id": "MCD-1002",
            "station": "T12",
            "image_id": "MCD-1002",
            "order_complete": False,
            "accuracy_score": 0.4,
            "missing_items": [{"name": "Veg Burger", "quantity": 1}],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )

    rate = mcp.get_rework_rate(period="today", baseline_period=None)
    assert rate["orders_seen"] == 2
    assert rate["orders_failed"] == 1
    assert rate["rework_rate"] == 0.5


def test_get_order_history_filters_and_shape(mcp):
    mcp.emit_order_result(
        {
            "order_id": "MCD-1001",
            "station": "T12",
            "image_id": "MCD-1001",
            "order_complete": True,
            "accuracy_score": 1.0,
        }
    )
    history = mcp.get_order_history(limit=5, order_id="MCD-1001")
    assert len(history) == 1
    entry = history[0]
    assert entry["order_id"] == "MCD-1001"
    assert entry["status"] == "validated"
    assert entry["event_type"] == "order_validated"
    assert "ts_ms" in entry and "ref_id" in entry


def test_get_station_totals_pass_fail_counts(mcp):
    mcp.emit_order_result(
        {
            "order_id": "MCD-1001",
            "station": "T12",
            "image_id": "MCD-1001",
            "order_complete": True,
            "accuracy_score": 1.0,
        }
    )
    mcp.emit_order_result(
        {
            "order_id": "MCD-1002",
            "station": "T18",
            "image_id": "MCD-1002",
            "order_complete": False,
            "accuracy_score": 0.4,
            "missing_items": [{"name": "Veg Burger", "quantity": 1}],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )

    totals = mcp.get_station_totals()
    # Seeded baseline history ("T1"/"T2") coexists with these real-traffic
    # stations — assert real stations are present rather than requiring
    # exact equality, mirroring take-away's equivalent test.
    assert {"T12", "T18"} <= set(totals)
    assert totals["T12"]["validated"] == 1
    assert totals["T18"]["failed"] == 1
    for bucket in totals.values():
        assert bucket["total"] == bucket["validated"] + bucket["failed"]
        assert 0.0 <= bucket["rework_rate"] <= 1.0

    single = mcp.get_station_totals("T12")
    assert single == totals["T12"]


# ---------------------------------------------------------------------------
# Sensor-only contract: no action tools, describe() is agent-discoverable
# ---------------------------------------------------------------------------


def test_no_action_tools_registered(mcp):
    """Issue #102: 'A: none (read/detect only)' — Order Accuracy is a sensor."""
    described = mcp.svc.describe()
    assert described["act_tools"] == {}
    assert mcp.svc._act_tools == {}


def test_describe_exposes_expected_read_tools_and_event_types(mcp):
    described = mcp.svc.describe()
    assert set(described["read_tools"]) == {
        "get_rework_rate",
        "get_order_history",
        "get_station_totals",
    }
    assert set(described["event_types"]) == {"order_validated", "order_failed"}
    assert described["service"] == "order_accuracy_dine_in"
