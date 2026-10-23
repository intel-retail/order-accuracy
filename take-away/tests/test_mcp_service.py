"""Tests for the MCP service integration (Issue #102 — Take-away).

Covers: event schema/emission, idempotent replay, read tools (rework rate
vs. baseline, order history, per-station totals), durable persistence across
restarts, no action tools exposed (Order Accuracy is a sensor), and the
disabled/benchmark-flag path.

Run with: python -m pytest tests/test_mcp_service.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        "QSR_AUTONOMY_EVENTS_URL": "",
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
    # SensorService has no act-tool machinery at all (by construction),
    # unlike the old SDK-backed ServiceServer which had an (unused) _act_tools.
    assert not hasattr(mcp.svc, "_act_tools")


def test_describe_exposes_expected_read_tools_and_event_types(mcp):
    described = mcp.svc.describe()
    assert set(described["read_tools"]) == {
        "get_rework_rate",
        "get_order_history",
        "get_station_totals",
    }
    assert set(described["event_types"]) == {"order_validated", "order_failed"}
    assert described["service"] == "order_accuracy"


# ---------------------------------------------------------------------------
# QSR agent autonomy events (POST /autonomy/events, weather_changed envelope)
# ---------------------------------------------------------------------------

_MISMATCH_RESULT = {
    "order_id": "539",
    "station_id": "station_2",
    "status": "mismatch",
    "validation": {
        "missing": [{"name": "apple", "quantity": 1}],
        "extra": [],
        "quantity_mismatch": [],
    },
    "num_frames": 4,
    "inference_time_sec": 3.0,
    "run_number": 1,
}


class _FakeQsrAgent:
    """Accepts events with the same rules as qsr-agent's AutonomyController._validate_event."""

    def __init__(self, restaurant_id: str = "qsr-001") -> None:
        self.restaurant_id = restaurant_id
        self.events: list[dict] = []
        self.headers: list[dict] = []
        self.rejections: list[str] = []
        self.received = threading.Event()
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                event = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                error = agent.validate(event)
                if error:
                    agent.rejections.append(error)
                    code, reply = 400, {"ok": False, "error": error}
                else:
                    agent.events.append(event)
                    agent.headers.append(dict(self.headers))
                    code = 202
                    reply = {"ok": True, "event_id": event["event_id"], "duplicate": False, "status": "queued"}
                    agent.received.set()
                raw = json.dumps(reply).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/autonomy/events"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def validate(self, event: dict) -> str | None:
        event_id = event.get("event_id", "")
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
            return "event_id is required"
        event_type = event.get("event_type", "")
        if not isinstance(event_type, str) or not event_type.strip() or len(event_type) > 200:
            return "event_type is required"
        if not isinstance(event.get("data"), dict):
            return "data must be an object"
        if len(json.dumps(event, allow_nan=False)) > 32768:
            return "event exceeds 32 KiB"
        if event.get("store_id", self.restaurant_id) != self.restaurant_id:
            return "event restaurant does not match this deployment"
        return None


@pytest.fixture
def qsr_agent(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    agent = _FakeQsrAgent()
    yield agent
    agent.server.shutdown()


def _envelope(event_type="order_failed"):
    from core.mcp_sensor import new_event

    return new_event(
        event_type,
        "order_accuracy",
        "store-001",
        {"event_id": "order_failed-20261014-120000-abcd1234",
         "dispatched_at": "2026-10-14T12:00:00.000+00:00", "order_id": "539"},
        ref_id="station_2:539:1",
    )


def test_autonomy_delivery_off_by_default(mcp):
    assert mcp._AUTONOMY_DELIVERY is None


def test_autonomy_event_posted_in_qsr_envelope(tmp_path, qsr_agent):
    gen = _reload_mcp_service(tmp_path, QSR_AUTONOMY_EVENTS_URL=qsr_agent.url)
    mcp = next(gen)
    try:
        event = mcp.emit_order_result(dict(_MISMATCH_RESULT))
        assert qsr_agent.received.wait(5), qsr_agent.rejections
    finally:
        gen.close()

    assert len(qsr_agent.events) == 1, "seed history must not be forwarded"
    posted = qsr_agent.events[0]
    assert set(posted) == {"event_id", "event_type", "store_id", "occurred_at", "data"}
    assert posted["event_type"] == "order_mismatch_detected"
    assert posted["store_id"] == "qsr-001"
    assert posted["event_id"] == f"order_accuracy-{event.payload['event_id']}"
    assert posted["occurred_at"] == event.payload["dispatched_at"]
    data = posted["data"]
    assert data["order_id"] == "539"
    assert data["station"] == "station_2"
    assert data["run_number"] == 1
    assert data["missing_items"] == [{"name": "apple", "quantity": 1}]
    assert data["reason"] == "missing:1"
    assert data["source"] == "order_accuracy"
    assert data["source_event_type"] == "order_failed"
    assert data["ref_id"] == "station_2:539:1"
    assert event.event_type == "order_failed"
    assert event.store_id == "store-001"


def test_autonomy_posts_both_validated_and_failed_orders(tmp_path, qsr_agent):
    import time

    gen = _reload_mcp_service(tmp_path, QSR_AUTONOMY_EVENTS_URL=qsr_agent.url)
    mcp = next(gen)
    try:
        mcp.emit_order_result({"order_id": "925", "station_id": "station_1", "status": "validated",
                               "num_frames": 3, "inference_time_sec": 2.1, "run_number": 1})
        mcp.emit_order_result(dict(_MISMATCH_RESULT))
        mcp.emit_order_result({"order_id": "926", "station_id": "station_1", "status": "error"})
        deadline = time.time() + 5
        while len(qsr_agent.events) < 2 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
    finally:
        gen.close()

    by_type = {e["event_type"]: e for e in qsr_agent.events}
    assert len(qsr_agent.events) == 2, qsr_agent.rejections
    assert set(by_type) == {"order_validated", "order_mismatch_detected"}
    validated = by_type["order_validated"]["data"]
    assert validated["order_id"] == "925"
    assert validated["run_number"] == 1
    assert validated["source_event_type"] == "order_validated"
    assert "missing_items" not in validated


def test_autonomy_skips_seed_history():
    from core.mcp_sensor import AutonomyEventSink, new_event

    calls = []
    sink = AutonomyEventSink("http://qsr", store_id="qsr-001",
                             post_json=lambda *a: calls.append(a) or 202)
    seed = new_event("order_failed", "order_accuracy", "store-001",
                     {"order_id": "seed-002"}, ref_id="seed:seed-002")
    assert sink.push(seed) is True
    assert calls == []


def test_autonomy_event_types_can_be_restricted():
    from core.mcp_sensor import AutonomyEventSink

    calls = []
    sink = AutonomyEventSink("http://qsr", store_id="qsr-001", event_types=("order_failed",),
                             post_json=lambda *a: calls.append(a) or 202)
    assert sink.push(_envelope("order_validated")) is True
    assert calls == []


def test_autonomy_sends_bearer_token(qsr_agent):
    from core.mcp_sensor import AutonomyEventSink

    sink = AutonomyEventSink(qsr_agent.url, store_id="qsr-001", token="s3cret")
    assert sink.push(_envelope()) is True
    assert qsr_agent.headers[0]["Authorization"] == "Bearer s3cret"


def test_autonomy_store_mismatch_is_rejected_without_retry(qsr_agent):
    from core.mcp_sensor import AutonomyEventSink, Delivery

    sink = AutonomyEventSink(qsr_agent.url, store_id="store-001")
    Delivery(sinks=[sink], backoff_s=0).dispatch(_envelope())
    assert qsr_agent.rejections == ["event restaurant does not match this deployment"]


@pytest.mark.parametrize("status, attempts", [(202, 1), (400, 1), (422, 1), (429, 3), (500, 3)])
def test_autonomy_retry_policy_matches_weather_service(status, attempts):
    from core.mcp_sensor import AutonomyEventSink, Delivery

    calls = []
    sink = AutonomyEventSink("http://qsr", store_id="qsr-001",
                             post_json=lambda *a: calls.append(a) or status)
    result = Delivery(sinks=[sink], backoff_s=0).dispatch(_envelope())
    assert result == {"autonomy": 200 <= status < 300}
    assert len(calls) == attempts


def test_autonomy_does_not_follow_redirects_or_leak_token(monkeypatch):
    from core.mcp_sensor import AutonomyEventSink

    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    target_hits = []

    class Target(BaseHTTPRequestHandler):
        def do_GET(self):
            target_hits.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    threading.Thread(target=target.serve_forever, daemon=True).start()
    location = f"http://127.0.0.1:{target.server_port}/"

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, *args):
            pass

    redirector = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    threading.Thread(target=redirector.serve_forever, daemon=True).start()
    try:
        sink = AutonomyEventSink(f"http://127.0.0.1:{redirector.server_port}/",
                                 store_id="qsr-001", token="s3cret")
        assert sink.push(_envelope()) is False
        assert target_hits == []
    finally:
        redirector.shutdown()
        target.shutdown()


def test_autonomy_concurrent_same_ref_emits_dispatch_once(tmp_path, qsr_agent):
    import time

    gen = _reload_mcp_service(tmp_path, QSR_AUTONOMY_EVENTS_URL=qsr_agent.url)
    mcp = next(gen)
    try:
        real_find = mcp._find_existing_event
        barrier = threading.Barrier(2)
        seen = threading.local()

        def racing_find(event_type, ref_id):
            # Force both threads past the pre-emit existence check together.
            if not getattr(seen, "checked", False):
                seen.checked = True
                barrier.wait(5)
                return None
            return real_find(event_type, ref_id)

        mcp._find_existing_event = racing_find
        payloads = [
            {"event_id": f"order_failed-race-{i}", "dispatched_at": "2026-10-14T12:00:00.000+00:00",
             "order_id": "MCD-RACE", "missing_items": []}
            for i in range(2)
        ]
        threads = [
            threading.Thread(target=mcp._safe_emit, args=("order_failed", p, "MCD-RACE:MCD-RACE"))
            for p in payloads
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        time.sleep(0.5)
        mcp._find_existing_event = real_find
        recorded = [e for e in mcp._read_entire_log() if e.ref_id == "MCD-RACE:MCD-RACE"]
    finally:
        gen.close()

    assert len(recorded) == 1
    assert len(qsr_agent.events) == 1
    assert qsr_agent.events[0]["event_id"].endswith(recorded[0].payload["event_id"])


def test_autonomy_opposite_outcome_for_same_ref_is_not_dispatched(tmp_path, qsr_agent):
    import time

    gen = _reload_mcp_service(tmp_path, QSR_AUTONOMY_EVENTS_URL=qsr_agent.url)
    mcp = next(gen)
    try:
        first = mcp._safe_emit("order_failed", {"event_id": "order_failed-opp-1", "order_id": "MCD-OPP",
                               "dispatched_at": "2026-10-14T12:00:00.000+00:00"}, "MCD-OPP:MCD-OPP")
        second = mcp._safe_emit("order_validated", {"event_id": "order_validated-opp-2", "order_id": "MCD-OPP",
                                "dispatched_at": "2026-10-14T12:00:01.000+00:00"}, "MCD-OPP:MCD-OPP")
        time.sleep(0.5)
        recorded = [e for e in mcp._read_entire_log() if e.ref_id == "MCD-OPP:MCD-OPP"]
    finally:
        gen.close()

    assert len(recorded) == 1 and recorded[0].event_type == "order_failed"
    assert second.event_type == "order_failed" and second.payload == first.payload
    assert [e["event_type"] for e in qsr_agent.events] == ["order_mismatch_detected"]


def test_autonomy_does_not_change_existing_webhook(tmp_path):
    gen = _reload_mcp_service(tmp_path, MCP_WEBHOOK_URL="http://legacy/hook")
    mcp = next(gen)
    try:
        assert mcp._AUTONOMY_DELIVERY is None
        assert [s.name for s in mcp._ASYNC_WEBHOOK_DELIVERY._sinks] == ["webhook"]
    finally:
        gen.close()
