"""Tests for the recorded-day replay workflow (Issue #102 AC5 — Dine-in).

Covers ``scripts/replay_log.py``, the application-level developer/validation
workflow that replays the durable MCP event log in original order. This is
intentionally NOT an MCP tool (Order Accuracy's MCP surface is read/detect
only) — it is a standalone script invoked via ``make replay``.

Verifies:
  - same event count as the underlying log
  - same sequence/order (by ``ref_id``, in order)
  - same event types
  - same payload
  - deterministic output on repeated runs
  - the original log is not modified (no new rows) by running replay

Run with: python -m pytest tests/test_replay_log.py -v
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_log.py"
sys.path.insert(0, str(SRC_DIR))


def _reload_mcp_service(tmp_path, **env_overrides):
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


def _run_replay(log_path: Path, fmt: str = "json") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--log-path", str(log_path), "--format", fmt],
        capture_output=True,
        text=True,
        check=True,
    )


def test_replay_missing_log_is_a_clean_no_op(tmp_path):
    """Replaying a path with no log yet must fail cleanly (exit 1), never
    create a new (empty) database file as a side effect."""
    missing = tmp_path / "does_not_exist.db"
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--log-path", str(missing)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert not missing.exists()


def test_replay_reproduces_same_count_order_types_and_payload(mcp):
    """Replay must reproduce every event, in original order, with identical
    event_type/ref_id/payload -- Issue #102 AC5 ('a recorded day replays
    identically from the log')."""
    mcp.emit_order_result(
        {
            "order_id": "replay-1",
            "station": "T1",
            "image_id": "replay-1",
            "order_complete": True,
            "accuracy_score": 1.0,
            "missing_items": [],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )
    mcp.emit_order_result(
        {
            "order_id": "replay-2",
            "station": "T2",
            "image_id": "replay-2",
            "order_complete": False,
            "accuracy_score": 0.5,
            "missing_items": ["fries"],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )

    log_path = Path(mcp.MCP_LOG_PATH)
    original_events = mcp.svc.log.read(limit=10_000)

    result = _run_replay(log_path, fmt="json")
    replayed_lines = [
        json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith("{")
    ]

    assert len(replayed_lines) == len(original_events)
    for original, replayed in zip(original_events, replayed_lines):
        assert replayed["event_type"] == original.event_type
        assert replayed["ref_id"] == original.ref_id
        assert replayed["ts_ms"] == original.ts_ms
        assert replayed["payload"] == original.payload


def test_replay_is_deterministic_across_repeated_runs(mcp):
    """Running replay twice against an unmodified log must produce
    byte-identical output."""
    mcp.emit_order_result(
        {
            "order_id": "det-1",
            "station": "T1",
            "image_id": "det-1",
            "order_complete": True,
            "accuracy_score": 1.0,
            "missing_items": [],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )
    log_path = Path(mcp.MCP_LOG_PATH)

    run1 = _run_replay(log_path, fmt="json")
    run2 = _run_replay(log_path, fmt="json")

    assert run1.stdout == run2.stdout


def test_replay_does_not_modify_the_original_log(mcp):
    """Replay is read-only: row count in the log before and after running
    the replay script must be identical."""
    mcp.emit_order_result(
        {
            "order_id": "ro-1",
            "station": "T1",
            "image_id": "ro-1",
            "order_complete": True,
            "accuracy_score": 1.0,
            "missing_items": [],
            "extra_items": [],
            "quantity_mismatches": [],
        }
    )
    log_path = Path(mcp.MCP_LOG_PATH)
    count_before = len(mcp.svc.log.read(limit=10_000))

    _run_replay(log_path, fmt="summary")
    _run_replay(log_path, fmt="json")

    count_after = len(mcp.svc.log.read(limit=10_000))
    assert count_after == count_before
