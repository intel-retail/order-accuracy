"""
Main launcher for Dine-In application with both API and Gradio UI
"""

import logging
import threading
import uvicorn
from api import app as fastapi_app
from app import app as gradio_app

logger = logging.getLogger(__name__)


def run_fastapi():
    """Run FastAPI server"""
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )


def run_gradio():
    """Run Gradio UI"""
    gradio_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


def start_mcp_server():
    """Start the MCP server (describe + read tools) in a background thread.

    Order Accuracy is a sensor: no action tools are exposed. Disabled
    entirely when MCP_SERVICE_ENABLED=false (clean benchmark runs).
    """
    try:
        from mcp_server import run_mcp_server
        from mcp_service import MCP_SERVICE_ENABLED

        if not MCP_SERVICE_ENABLED:
            logger.info("[MCP] MCP_SERVICE_ENABLED=false, skipping MCP server startup")
            return None

        thread = threading.Thread(
            target=run_mcp_server, name="MCPServer", daemon=True
        )
        thread.start()
        logger.info("[MCP] MCP server started in background thread")
        return thread
    except Exception as e:
        logger.error(f"[MCP] Failed to start MCP server: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Dine-In Order Accuracy Services")
    print("=" * 60)
    print("FastAPI Server: http://localhost:8080")
    print("API Docs: http://localhost:8080/docs")
    print("Gradio UI: http://localhost:7860")
    print("=" * 60)
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()

    # Start the MCP server (events/read-tools) in background
    start_mcp_server()
    
    # Run Gradio in main thread
    run_gradio()
