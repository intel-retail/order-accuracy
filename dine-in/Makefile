# Benchmark configuration
CONCURRENCY_START ?= 1
CONCURRENCY_INCREMENT ?= 1
CONCURRENCY_MAX ?= 10
REQUESTS_PER_LEVEL ?= 10
REQUEST_TIMEOUT ?= 30
RESULTS_DIR ?= results/stream-density
OOM_PROTECTION ?= 1

.PHONY: help build up down logs restart benchmark benchmark-results benchmark-metrics benchmark-stream-density clean

help:
	@echo "Dine-In Order Accuracy - Available Commands:"
	@echo ""
	@echo "  make build                    - Build Docker images"
	@echo "  make up                       - Start services (UI mode)"
	@echo "  make down                     - Stop all services"
	@echo "  make logs                     - View application logs"
	@echo "  make restart                  - Restart dine-in service"
	@echo "  make benchmark                - Run single image benchmark (image_01.png)"
	@echo "  make benchmark-results        - View benchmark results"
	@echo "  make benchmark-metrics        - View benchmark performance metrics"
	@echo "  make benchmark-stream-density - Test concurrent request capacity"
	@echo "  make clean                    - Remove containers and volumes"
	@echo ""
	@echo "Stream Density Benchmark Options:"
	@echo "  CONCURRENCY_START=1           - Starting concurrent requests"
	@echo "  CONCURRENCY_INCREMENT=1       - Increment concurrent requests by"
	@echo "  CONCURRENCY_MAX=10            - Maximum concurrent requests"
	@echo "  REQUESTS_PER_LEVEL=10         - Requests to send at each level"
	@echo "  OOM_PROTECTION=1              - Enable memory protection (0=disable)"
	@echo ""

build:
	@echo "Building dine-in Docker images..."
	docker compose build

up:
	@echo "Starting dine-in services..."
	docker compose up -d
	@echo "Services started:"
	@echo "  - Gradio UI: http://localhost:7861"
	@echo "  - API: http://localhost:8083"
	@echo "  - API Docs: http://localhost:8083/docs"
	@echo ""
	@echo "Run 'make benchmark' to test with image_01.png"

down:
	@echo "Stopping dine-in services..."
	docker compose down

logs:
	@echo "Showing dine-in application logs (Ctrl+C to exit)..."
	docker logs -f dinein_app

restart:
	@echo "Restarting dine-in service..."
	docker compose restart dine-in

benchmark:
	@echo "Running benchmark with image_01.png..."
	@mkdir -p results
	@bash -c '\
		IMAGE_FILE="images/image_01.png"; \
		IMAGE_ID="image_01_mcd_combo"; \
		ORDER_JSON=$$(jq -c ".orders[] | select(.image_id == \"$$IMAGE_ID\")" orders/orders.json); \
		if [ -z "$$ORDER_JSON" ]; then \
			echo "Error: Order not found for $$IMAGE_ID"; \
			exit 1; \
		fi; \
		ORDER_MANIFEST=$$(echo $$ORDER_JSON | jq -c "{items: [.items_ordered[] | {name: .item, quantity: .quantity}]}"); \
		echo "=== Starting Benchmark ==="; \
		echo "Image: $$IMAGE_FILE"; \
		echo "Order: $$ORDER_MANIFEST" | jq .; \
		echo ""; \
		START_TIME=$$(date +%s%3N); \
		RESPONSE=$$(curl -s -X POST http://localhost:8083/api/validate \
			-F "image=@$$IMAGE_FILE" \
			-F "order=$$ORDER_MANIFEST"); \
		END_TIME=$$(date +%s%3N); \
		LATENCY=$$((END_TIME - START_TIME)); \
		echo ""; \
		echo "=== Benchmark Results ==="; \
		echo "$$RESPONSE" | jq .; \
		echo ""; \
		echo "=== Logging Results ==="; \
		TIMESTAMP=$$(date "+%Y-%m-%d %H:%M:%S"); \
		echo "[$$TIMESTAMP] Benchmark Result for $$IMAGE_ID" >> results/benchmark_result.log; \
		echo "$$RESPONSE" | jq . >> results/benchmark_result.log; \
		echo "" >> results/benchmark_result.log; \
		ACCURACY=$$(echo "$$RESPONSE" | jq -r ".accuracy_score // 0"); \
		ORDER_COMPLETE=$$(echo "$$RESPONSE" | jq -r ".order_complete // false"); \
		VALIDATION_ID=$$(echo "$$RESPONSE" | jq -r ".validation_id // \"N/A\""); \
		echo "[$$TIMESTAMP] validation_id=$$VALIDATION_ID, image=$$IMAGE_ID, latency_ms=$$LATENCY, accuracy=$$ACCURACY, complete=$$ORDER_COMPLETE" >> results/benchmark_performance.log; \
		if [ ! -z "$$RESPONSE" ] && echo "$$RESPONSE" | jq -e ".metrics" > /dev/null 2>&1; then \
			echo "$$RESPONSE" | jq -r ".metrics | to_entries | map(\"  \\(.key)=\\(.value)\") | .[]" >> results/benchmark_performance.log; \
		fi; \
		echo "" >> results/benchmark_performance.log; \
		echo "Results logged to results/benchmark_result.log"; \
		echo "Performance metrics logged to results/benchmark_performance.log"; \
	'

benchmark-results:
	@echo "=== Benchmark Results ==="
	@if [ -f results/benchmark_result.log ]; then \
		cat results/benchmark_result.log; \
	else \
		echo "No benchmark results found. Run 'make benchmark' first."; \
	fi

benchmark-metrics:
	@echo "=== Benchmark Performance Metrics ==="
	@if [ -f results/benchmark_performance.log ]; then \
		cat results/benchmark_performance.log; \
	else \
		echo "No benchmark metrics found. Run 'make benchmark' first."; \
	fi

benchmark-stream-density:
	@if [ "$(OOM_PROTECTION)" = "0" ]; then \
		echo "╔════════════════════════════════════════════════════════════╗"; \
		echo "║ WARNING                                                    ║"; \
		echo "║                                                            ║"; \
		echo "║ OOM Protection is DISABLED. This test may:                 ║"; \
		echo "║ • Cause system instability or crashes                      ║"; \
		echo "║ • Require hard reboot if system becomes unresponsive       ║"; \
		echo "║ • Result in data loss in other applications                ║"; \
		echo "║                                                            ║"; \
		echo "║ Press Ctrl+C now to cancel, or wait 5 seconds...           ║"; \
		echo "╚════════════════════════════════════════════════════════════╝"; \
		sleep 5; \
	fi
	@( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r benchmark_requirements.txt && \
		python3 benchmark_density.py \
			--concurrency_start $(CONCURRENCY_START) \
			--concurrency_max $(CONCURRENCY_MAX) \
			--concurrency_increment $(CONCURRENCY_INCREMENT) \
			--requests_per_level $(REQUESTS_PER_LEVEL) \
			--timeout $(REQUEST_TIMEOUT) \
			--results_dir $(RESULTS_DIR) \
			--oom_protection $(OOM_PROTECTION); \
		deactivate \
	)

clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v
	@echo "Cleanup complete"

# Development helpers
shell:
	@echo "Opening shell in dine-in container..."
	docker exec -it dinein_app /bin/bash

test-api:
	@echo "Testing API health..."
	@curl -s http://localhost:8083/health | jq
