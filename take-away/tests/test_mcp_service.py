"""Tests for the MCP service integration (Issue #102 — Take-away).

Covers: event schema/emission, idempotent replay, read tools (rework rate
vs. baseline, order history, per-station totals), durable persistence across
restarts, no action tools exposed (Order Accuracy is a sensor), and the
disabled/benchmark-flag path.

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
    """Import a fresh core.mcp_service with env vars applied and module-level
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

    # Drop any previously imported instance of the module (and its package
    # ancestors' cached submodule reference) so re-import re-runs __post_init__
    # / seeding against the new tmp_path.
    sys.modules.pop("core.mcp_service", None)
    module = importlib.import_module("core.mcp_service")
    try:
        yield module
    finally:
        sys.modules.pop("core.mcp_service", None)
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
    """Ships preloaded with baseline history so comparative queries work."""
    events = mcp.svc.log.read(limit=1000)
    assert len(events) == len(mcp._SEED_PATTERN)
    assert {e.event_type for e in events} <= {"order_validated", "order_failed"}


def test_emit_order_validated(mcp):
    result = {
        "order_id": "925",
        "station_id": "station_1",
        "status": "validated",
        "num_frames": 3,
        "inference_time_sec": 2.1,
        "run_number": 1,
    }
    event = mcp.emit_order_result(result)
    assert event.event_type == "order_validated"
    assert event.payload["order_id"] == "925"
    assert event.payload["station"] == "station_1"
    assert event.ref_id == "station_1:925:1"


def test_emit_order_failed_payload_has_missing_extra_reason(mcp):
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
    event = mcp.emit_order_result(result)
    assert event.event_type == "order_failed"
    assert event.payload["missing_items"] == [{"name": "apple", "quantity": 1}]
    assert event.payload["extra_items"] == [{"name": "banana", "quantity": 1}]
    assert event.payload["reason"] == "missing:1,extra:1"


@pytest.mark.parametrize("status", ["error", "no_frames", None])
def test_emit_skips_non_terminal_statuses(mcp, status):
    result = {"order_id": "1", "station_id": "station_1", "status": status}
    assert mcp.emit_order_result(result) is None


def test_emit_is_idempotent_on_same_ref_id(mcp):
    result = {
        "order_id": "539",
        "station_id": "station_2",
        "status": "mismatch",
        "validation": {"missing": [], "extra": [], "quantity_mismatch": []},
        "num_frames": 1,
        "inference_time_sec": 1.0,
        "run_number": 1,
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
        "order_id": "1",
        "station_id": "station_1",
        "status": "validated",
        "num_frames": 1,
        "inference_time_sec": 1.0,
        "run_number": 1,
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
            "order_id": "1",
            "station_id": "station_1",
            "status": "validated",
            "num_frames": 1,
            "inference_time_sec": 1.0,
            "run_number": 1,
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


def test_get_rework_rate_today_vs_baseline(mcp):
    mcp.emit_order_result(
        {
            "order_id": "925",
            "station_id": "station_1",
            "status": "validated",
            "num_frames": 1,
            "inference_time_sec": 1.0,
            "run_number": 1,
        }
    )
    mcp.emit_order_result(
        {
            "order_id": "539",
            "station_id": "station_1",
            "status": "mismatch",
            "validation": {"missing": [{"name": "apple", "quantity": 1}], "extra": [], "quantity_mismatch": []},
            "num_frames": 1,
            "inference_time_sec": 1.0,
            "run_number": 1,
        }
    )

    rate = mcp.get_rework_rate(period="today")
    assert rate["orders_seen"] == 2
    assert rate["orders_failed"] == 1
    assert rate["rework_rate"] == 0.5
    assert rate["baseline_period"] == "yesterday"
    assert rate["baseline_orders_seen"] == len(mcp._SEED_PATTERN)
    assert "delta_vs_baseline" in rate


def test_get_rework_rate_filtered_by_station(mcp):
    rate_all = mcp.get_rework_rate(period="yesterday", baseline_period=None)
    rate_station_1 = mcp.get_rework_rate(period="yesterday", station="station_1", baseline_period=None)
    assert rate_all["orders_seen"] == len(mcp._SEED_PATTERN)
    assert rate_station_1["orders_seen"] < rate_all["orders_seen"]


def test_get_order_history_filters_and_shape(mcp):
    mcp.emit_order_result(
        {
            "order_id": "925",
            "station_id": "station_1",
            "status": "validated",
            "num_frames": 1,
            "inference_time_sec": 1.0,
            "run_number": 1,
        }
    )
    history = mcp.get_order_history(limit=5, order_id="925")
    assert len(history) == 1
    entry = history[0]
    assert entry["order_id"] == "925"
    assert entry["status"] == "validated"
    assert entry["event_type"] == "order_validated"
    assert "ts_ms" in entry and "ref_id" in entry


def test_get_station_totals_pass_fail_counts(mcp):
    totals = mcp.get_station_totals()
    assert set(totals) == {"station_1", "station_2"}
    for bucket in totals.values():
        assert bucket["total"] == bucket["validated"] + bucket["failed"]
        assert 0.0 <= bucket["rework_rate"] <= 1.0

    single = mcp.get_station_totals("station_1")
    assert single == totals["station_1"]


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
    assert described["service"] == "order_accuracy"
