"""Main entry point for the Order Accuracy App."""

from config import logger, SERVER_HOST, SERVER_PORT
from messaging import test_rabbitmq_connection
from interface import build_interface
from final_report import reset_metrics_file


def main():
    """Main application entry point."""
    logger.info("Starting Grocery Video Detection App")
    logger.info(f"Configuration loaded - Server: {SERVER_HOST}:{SERVER_PORT}")
    
    # Initialize fresh metrics for new run
    reset_metrics_file()
    # Test RabbitMQ connection
    logger.info("Testing RabbitMQ MQTT connection...")
    test_ok, test_msg = test_rabbitmq_connection()
    if test_ok:
        logger.info(f"✓ {test_msg}")
    else:
        logger.error(f"✗ {test_msg}")
        logger.warning("RabbitMQ connection test failed, but continuing anyway...")
    
    # Build and launch the interface
    demo = build_interface()
    demo.queue(max_size=20, default_concurrency_limit=10).launch(
        server_name="0.0.0.0",
        server_port=SERVER_PORT, 
        debug=False,
        show_error=True,
        quiet=False,
        share=True
    )


if __name__ == "__main__":
    main()
