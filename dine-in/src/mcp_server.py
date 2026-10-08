#!/usr/bin/env python3
"""Order Accuracy (Dine-in) MCP server entrypoint.

Exposes ``describe`` and the three required read tools
(``get_rework_rate``, ``get_order_history``, ``get_station_totals``) over
MCP, built directly on FastMCP via ``mcp_sensor.SensorService`` (see that
module's docstring for the rationale for no longer depending on the shared
``mcp-service-sdk`` package).

Order Accuracy is a sensor (Issue #102: "A: none (read/detect only)"), so
``SensorService.to_mcp()`` only ever binds ``describe`` plus the registered
read tools — there is no ``subscribe``/callback tool and no action tool to
strip after construction; the previous SDK-era workaround that removed a
forced ``subscribe`` tool post-hoc no longer applies because nothing here
ever adds one.

Run directly:
    python -m mcp_server

Or import ``run_mcp_server`` to start it on a background thread from the
main application process (see ``main.py``).
"""
from __future__ import annotations

import logging
import os

from mcp_service import MCP_HOST, MCP_PORT, MCP_SERVICE_ENABLED, MCP_TRANSPORT, svc

logger = logging.getLogger(__name__)


def _build_app():
    """Build the MCP app: ``describe`` plus the registered read tools only."""
    return svc.to_mcp()


def run_mcp_server() -> None:
    """Start the MCP server (blocking). No-op if disabled via env flag."""
    if not MCP_SERVICE_ENABLED:
        logger.info("[MCP] MCP_SERVICE_ENABLED=false, MCP server not started")
        return
    logger.info(
        "[MCP] Starting MCP server: transport=%s host=%s port=%s",
        MCP_TRANSPORT,
        MCP_HOST,
        MCP_PORT,
    )
    app = _build_app()
    if MCP_TRANSPORT == "stdio":
        app.run("stdio")
    else:
        app.run(MCP_TRANSPORT, host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    run_mcp_server()
