#!/usr/bin/env python3
"""Order Accuracy (Take-away) MCP server entrypoint.

Exposes ``describe`` and the three required read tools
(``get_rework_rate``, ``get_order_history``, ``get_station_totals``) over
MCP. No action tools are registered: Order Accuracy is a sensor.

Issue #102 explicitly requires "no subscribe/callback" on this server, but
``mcp_service_sdk.ServiceServer.to_mcp()`` unconditionally binds a
``subscribe`` tool as part of its own MCP scaffolding — there is no
``ServiceConfig`` flag or supported constructor argument to suppress it (see
the SDK's ``server.py``: the ``@app.tool(name="subscribe", ...)`` binding is
hard-coded inside ``to_mcp()``, right alongside ``describe``). Since the SDK
is pinned to a shared upstream commit used by both applications, patching it
directly would be a fork with its own maintenance cost for one tool.

Instead, this module builds the MCP app the same way ``ServiceServer.run()``
would (``svc.to_mcp()``), then removes the unwanted ``subscribe`` tool at the
FastMCP layer via the underlying provider's public ``remove_tool()`` API —
an application-level, SDK-untouched fix. This only changes what is exposed
over the wire; it does not touch ``ServiceServer.subscribe()`` or
``_subscriptions`` internals, which remain unused by this application either
way.

Run directly:
    python -m mcp_server

Or import ``run_mcp_server`` to start it on a background thread from the
main application process (see ``main.py``).
"""
from __future__ import annotations

import logging
import os

from core.mcp_service import MCP_HOST, MCP_PORT, MCP_SERVICE_ENABLED, MCP_TRANSPORT, svc

logger = logging.getLogger(__name__)

# Application-defined MCP tools this service is allowed to expose over the
# wire, per Issue #102 ("no subscribe/callback"). Anything the SDK's
# to_mcp() adds beyond this set is stripped in _build_app().
_DISALLOWED_TOOLS = ("subscribe",)


def _build_app():
    """Build the MCP app via the SDK, then strip disallowed SDK-default tools.

    ``ServiceServer.to_mcp()`` returns a live FastMCP app object; removing a
    tool afterwards is a normal FastMCP operation (``LocalProvider.
    remove_tool``), not an SDK modification. If a future FastMCP/SDK version
    changes this internal shape, we log a warning and continue rather than
    crash the server — worst case the extra tool remains visible, which is
    the same behavior as before this fix.
    """
    app = svc.to_mcp()
    for name in _DISALLOWED_TOOLS:
        try:
            app._local_provider.remove_tool(name)
            logger.info("[MCP] Removed disallowed tool '%s' from MCP surface", name)
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.warning("[MCP] Could not remove tool '%s': %s", name, exc)
    return app


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
