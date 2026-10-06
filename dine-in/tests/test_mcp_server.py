"""Tests for the MCP server entrypoint (Issue #102 — Dine-in).

Covers Issue #102 AC7 ("MCP server exposes describe + read tools ... — no
subscribe/callback"): ``SensorService.to_mcp()`` (the local ``mcp_sensor``
replacement for ``mcp_service_sdk``) only ever binds ``describe`` plus the
registered read tools — there is no ``subscribe`` tool to strip because
nothing in the new scaffolding ever adds one (unlike the old SDK's
``ServiceServer.to_mcp()``, which unconditionally bound ``subscribe`` and
required ``mcp_server._build_app()`` to remove it post-hoc).

Mirrors take-away/tests/test_mcp_server.py, adapted for Dine-in's flat
module layout (``mcp_service``/``mcp_server`` at ``src/`` top level, not
under a ``core`` package).

Run with: python -m pytest tests/test_mcp_server.py -v
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

_REQUIRED_TOOLS = {
    "describe",
    "get_rework_rate",
    "get_order_history",
    "get_station_totals",
}


def _reload_mcp_server(tmp_path):
    """Import fresh mcp_service + mcp_server modules against an isolated
    log file, mirroring test_mcp_service.py's _reload_mcp_service.
    """
    env = {
        "RESULTS_DIR": str(tmp_path),
        "MCP_LOG_PATH": str(tmp_path / "events.db"),
        "MCP_SERVICE_ENABLED": "true",
        "MCP_WEBHOOK_URL": "",
    }
    old_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    sys.modules.pop("mcp_server", None)
    sys.modules.pop("mcp_service", None)
    module = importlib.import_module("mcp_server")
    try:
        yield module
    finally:
        sys.modules.pop("mcp_server", None)
        sys.modules.pop("mcp_service", None)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture
def mcp_server_module(tmp_path):
    yield from _reload_mcp_server(tmp_path)


def test_subscribe_tool_is_removed_from_mcp_surface(mcp_server_module):
    app = mcp_server_module._build_app()
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert "subscribe" not in names


def test_all_required_read_tools_still_exposed(mcp_server_module):
    app = mcp_server_module._build_app()
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert _REQUIRED_TOOLS <= names


def test_mcp_surface_is_exactly_the_required_set(mcp_server_module):
    """No subscribe, no action tools, no unexpected extras."""
    app = mcp_server_module._build_app()
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    assert names == _REQUIRED_TOOLS


def test_describe_and_service_internals_unaffected(mcp_server_module):
    """Building the MCP app must not mutate SensorService.describe()'s
    contract, and the service has no subscribe/act-tool machinery at all —
    by construction, not by post-hoc removal (see mcp_sensor.py)."""
    from mcp_service import svc

    described = svc.describe()
    assert described["act_tools"] == {}
    assert set(described["read_tools"]) == {
        "get_rework_rate",
        "get_order_history",
        "get_station_totals",
    }
    # Order Accuracy is a sensor: SensorService never had a subscribe() or
    # _subscriptions attribute to begin with (unlike the old SDK-backed
    # ServiceServer), so there is nothing to assert is "unused" here.
    assert not hasattr(svc, "subscribe")
