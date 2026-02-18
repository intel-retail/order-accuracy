"""
Main launcher for Dine-In application with both API and Gradio UI
"""

import threading
import uvicorn
from api import app as fastapi_app
from app import app as gradio_app


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
    
    # Run Gradio in main thread
    run_gradio()
