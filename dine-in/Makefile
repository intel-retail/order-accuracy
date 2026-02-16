# Copyright © 2025 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Dine-In Order Accuracy - Makefile
# =============================================================================

# Worker configuration for stream density testing
WORKERS ?= 1
ITERATIONS ?= 0
REQUEST_DELAY ?= 0

# Benchmark configuration (legacy)
CONCURRENCY_START ?= 1
CONCURRENCY_INCREMENT ?= 1
CONCURRENCY_MAX ?= 10
REQUESTS_PER_LEVEL ?= 10
REQUEST_TIMEOUT ?= 30
RESULTS_DIR ?= results
OOM_PROTECTION ?= 1

# Performance tools path
PERF_TOOLS_DIR ?= ../performance-tools/benchmark-scripts

# Dine-In Stream Density Benchmark Settings
BENCHMARK_TARGET_LATENCY_MS ?= 15000
BENCHMARK_LATENCY_METRIC ?= avg
BENCHMARK_INIT_DURATION ?= 60
BENCHMARK_DENSITY_INCREMENT ?= 1
BENCHMARK_MIN_REQUESTS ?= 3
BENCHMARK_REQUEST_TIMEOUT ?= 300
BENCHMARK_API_ENDPOINT ?= http://localhost:8083

# Docker compose file
COMPOSE_FILE ?= docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

.PHONY: help build up down logs restart benchmark benchmark-results benchmark-metrics \
        benchmark-stream-density benchmark-density benchmark-density-results clean \
        shell test-api benchmark-test benchmark-single \
        worker worker-start worker-stop worker-logs worker-results

help:
	@echo "$(BLUE)Dine-In Order Accuracy - Available Commands$(NC)"
	@echo "=============================================="
	@echo ""
	@echo "$(YELLOW)Basic Commands:$(NC)"
	@echo "  make build                    - Build Docker images"
	@echo "  make up                       - Start services (UI mode)"
	@echo "  make down                     - Stop all services"
	@echo "  make logs                     - View application logs"
	@echo "  make restart                  - Restart dine-in service"
	@echo "  make clean                    - Remove containers and volumes"
	@echo ""
	@echo "$(YELLOW)Worker Commands (Stream Density Testing):$(NC)"
	@echo "  make worker                   - Run worker locally (WORKERS=N)"
	@echo "  make worker-start             - Start N workers in Docker (WORKERS=N)"
	@echo "  make worker-stop              - Stop all workers"
	@echo "  make worker-logs              - View worker logs"
	@echo "  make worker-results           - View worker results"
	@echo ""
	@echo "$(YELLOW)Benchmark Commands:$(NC)"
	@echo "  make benchmark-test           - Quick single image test via API"
	@echo "  make benchmark-single         - Test with specific IMAGE_ID"
	@echo "  make benchmark                - Run single image benchmark (legacy)"
	@echo "  make benchmark-results        - View benchmark results"
	@echo "  make benchmark-metrics        - View benchmark performance metrics"
	@echo ""
	@echo "$(YELLOW)Stream Density Benchmark (Image-Based):$(NC)"
	@echo "  make benchmark-density        - Run stream density test (latency-based)"
	@echo "  make benchmark-density-results - View density benchmark results"
	@echo "  make benchmark-density-help   - Show density benchmark options"
	@echo ""
	@echo "$(YELLOW)Worker Configuration:$(NC)"
	@echo "  WORKERS=$(WORKERS)              - Number of concurrent workers"
	@echo "  ITERATIONS=$(ITERATIONS)           - Iterations per worker (0=infinite)"
	@echo "  REQUEST_DELAY=$(REQUEST_DELAY)         - Delay between requests (seconds)"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make shell                    - Open shell in dine-in container"
	@echo "  make test-api                 - Test API health endpoint"
	@echo ""

benchmark-density-help:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Dine-In Stream Density Benchmark - Image Based              ║"
	@echo "╠═══════════════════════════════════════════════════════════════════╣"
	@echo "║ This benchmark tests how many concurrent images the system can    ║"
	@echo "║ process while maintaining target latency.                         ║"
	@echo "║                                                                   ║"
	@echo "║ Density = Number of concurrent image validation requests          ║"
	@echo "║                                                                   ║"
	@echo "║ Flow:                                                             ║"
	@echo "║   1. Start with density=1 (1 concurrent image)                    ║"
	@echo "║   2. Send N concurrent image validations to VLM                   ║"
	@echo "║   3. Measure latency from API responses                           ║"
	@echo "║   4. If latency <= target: increase density by 1, repeat          ║"
	@echo "║   5. If latency > target: STOP, report max density                ║"
	@echo "╠═══════════════════════════════════════════════════════════════════╣"
	@echo "║ Configuration Variables:                                          ║"
	@echo "║   BENCHMARK_TARGET_LATENCY_MS=$(BENCHMARK_TARGET_LATENCY_MS)ms (latency threshold)           ║"
	@echo "║   BENCHMARK_LATENCY_METRIC=$(BENCHMARK_LATENCY_METRIC) (avg, p95, max)                    ║"
	@echo "║   BENCHMARK_INIT_DURATION=$(BENCHMARK_INIT_DURATION)s (service warmup time)              ║"
	@echo "║   BENCHMARK_DENSITY_INCREMENT=$(BENCHMARK_DENSITY_INCREMENT) (images to add per iteration)   ║"
	@echo "║   BENCHMARK_MIN_REQUESTS=$(BENCHMARK_MIN_REQUESTS) (min requests per iteration)          ║"
	@echo "║   BENCHMARK_REQUEST_TIMEOUT=$(BENCHMARK_REQUEST_TIMEOUT)s (individual request timeout)     ║"
	@echo "║   BENCHMARK_API_ENDPOINT=$(BENCHMARK_API_ENDPOINT)                ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"

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

# ==============================================================================
# Worker Commands for Stream Density Testing
# ==============================================================================

worker: ## Run worker locally (WORKERS=N ITERATIONS=M)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║            Dine-In Worker - Local Execution                       ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Workers: $(WORKERS)"
	@echo "Iterations: $(ITERATIONS) (0=infinite)"
	@echo "Request Delay: $(REQUEST_DELAY)s"
	@echo ""
	@mkdir -p $(RESULTS_DIR)
	cd src && python3 worker.py \
		--workers $(WORKERS) \
		--iterations $(ITERATIONS) \
		--delay $(REQUEST_DELAY) \
		--images-dir $(CURDIR)/images \
		--orders-file $(CURDIR)/configs/orders.json \
		--results-dir $(CURDIR)/$(RESULTS_DIR)

worker-start: ## Start N workers in Docker (WORKERS=N)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║         Starting Dine-In Workers (Docker)                         ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Workers: $(WORKERS)"
	@echo "Iterations: $(ITERATIONS) (0=infinite)"
	@echo ""
	WORKERS=$(WORKERS) ITERATIONS=$(ITERATIONS) REQUEST_DELAY=$(REQUEST_DELAY) \
		docker compose --profile benchmark up -d --scale dinein-worker=$(WORKERS)
	@echo ""
	@echo "$(GREEN)✓ $(WORKERS) worker(s) started$(NC)"
	@echo "Run 'make worker-logs' to view logs"
	@echo "Run 'make worker-stop' to stop workers"
	@echo "Run 'make worker-results' to view results"

worker-stop: ## Stop all workers
	@echo "Stopping all workers..."
	docker compose --profile benchmark down
	@echo "$(GREEN)✓ Workers stopped$(NC)"

worker-logs: ## View worker logs
	@echo "Showing worker logs (Ctrl+C to exit)..."
	docker compose --profile benchmark logs -f dinein-worker

worker-results: ## View worker results
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Worker Results Summary                          ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@if [ -d $(RESULTS_DIR) ]; then \
		echo "=== Results Files ==="; \
		ls -la $(RESULTS_DIR)/worker_*.json 2>/dev/null || echo "No worker results found"; \
		echo ""; \
		for f in $(RESULTS_DIR)/worker_*_results_*.json; do \
			if [ -f "$$f" ]; then \
				echo "--- $$f ---"; \
				python3 -c "import json; d=json.load(open('$$f')); \
					print(f\"  Worker {d['worker_id']}: {d['successful_iterations']}/{d['total_iterations']} iterations\"); \
					print(f\"  Avg Latency: {d['avg_latency_ms']:.0f}ms\"); \
					print(f\"  Min/Max: {d['min_latency_ms']:.0f}ms / {d['max_latency_ms']:.0f}ms\"); \
					print(f\"  Avg TPS: {d['avg_tps']:.2f}\"); \
					print()"; \
			fi; \
		done; \
	else \
		echo "No results directory found. Run 'make worker' or 'make worker-start' first."; \
	fi

logs:
	@echo "Showing dine-in application logs (Ctrl+C to exit)..."
	docker logs -f dinein_app

restart:
	@echo "Restarting dine-in service..."
	docker compose restart dine-in

# Quick single image test using worker
benchmark-test: ## Quick single image test (1 worker, 1 iteration)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║           Dine-In Single Image Benchmark Test                     ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@mkdir -p $(RESULTS_DIR)
	@echo "Running 1 worker with 1 iteration..."
	cd src && python3 worker.py \
		--workers 1 \
		--iterations 1 \
		--delay 0 \
		--images-dir $(CURDIR)/images \
		--orders-file $(CURDIR)/configs/orders.json \
		--results-dir $(CURDIR)/$(RESULTS_DIR)
	@echo ""
	@echo "$(GREEN)✓ Test complete. Check results:$(NC)"
	@ls -la $(RESULTS_DIR)/worker_*.json 2>/dev/null | tail -1

# Test with specific image ID using curl (for quick API testing)
benchmark-single: ## Test with specific image via API (IMAGE_ID=DD-6993)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║           Dine-In Single Image Test - $(IMAGE_ID)                 ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@mkdir -p $(RESULTS_DIR)
	@if [ -z "$(IMAGE_ID)" ]; then \
		echo "$(RED)Error: IMAGE_ID not specified$(NC)"; \
		echo "Usage: make benchmark-single IMAGE_ID=DD-6993"; \
		exit 1; \
	fi
	@echo "Testing image: $(IMAGE_ID)"
	@bash -c '\
		IMAGE_FILE="images/$(IMAGE_ID).jpg"; \
		if [ ! -f "$$IMAGE_FILE" ]; then \
			echo "$(RED)Error: Image $$IMAGE_FILE not found$(NC)"; \
			exit 1; \
		fi; \
		ORDER_JSON=$$(jq -c ".orders[] | select(.image_id == \"$(IMAGE_ID)\")" configs/orders.json); \
		if [ -z "$$ORDER_JSON" ]; then \
			echo "$(RED)Error: Order not found for $(IMAGE_ID)$(NC)"; \
			exit 1; \
		fi; \
		ORDER_MANIFEST=$$(echo $$ORDER_JSON | jq -c "{items: [.items_ordered[] | {name: .item, quantity: .quantity}]}"); \
		echo "Order: $$ORDER_MANIFEST" | jq .; \
		echo ""; \
		START_TIME=$$(date +%s%3N); \
		RESPONSE=$$(curl -s -X POST $(BENCHMARK_API_ENDPOINT)/api/validate \
			-F "image=@$$IMAGE_FILE" \
			-F "order=$$ORDER_MANIFEST"); \
		END_TIME=$$(date +%s%3N); \
		LATENCY=$$((END_TIME - START_TIME)); \
		echo "=== Response ==="; \
		echo "$$RESPONSE" | jq .; \
		echo ""; \
		echo "$(GREEN)Latency: $${LATENCY}ms$(NC)"; \
	'

# Legacy benchmark (original implementation)
benchmark:
	@echo "Running benchmark with DD-6993.jpg..."
	@mkdir -p results
	@bash -c '\
		IMAGE_FILE="images/DD-6993.jpg"; \
		IMAGE_ID="DD-6993"; \
		ORDER_JSON=$$(jq -c ".orders[] | select(.image_id == \"$$IMAGE_ID\")" configs/orders.json); \
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

benchmark-stream-density: benchmark-density
	@echo "Note: benchmark-stream-density is deprecated, use 'make benchmark-density' instead"

benchmark-density: ## Run Dine-In stream density benchmark (image-based latency)
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
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Dine-In Stream Density - Image-Based Latency Mode           ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Target Latency: $(BENCHMARK_TARGET_LATENCY_MS)ms"
	@echo "Latency Metric: $(BENCHMARK_LATENCY_METRIC)"
	@echo "Init Duration: $(BENCHMARK_INIT_DURATION)s"
	@echo "Density Increment: +$(BENCHMARK_DENSITY_INCREMENT) image(s) per iteration"
	@echo "Min Requests: $(BENCHMARK_MIN_REQUESTS)"
	@echo "Request Timeout: $(BENCHMARK_REQUEST_TIMEOUT)s"
	@echo "API Endpoint: $(BENCHMARK_API_ENDPOINT)"
	@echo ""
	mkdir -p $(RESULTS_DIR)
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 stream_density_oa_dine_in.py \
			--compose_file $(CURDIR)/$(COMPOSE_FILE) \
			--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
			--latency_metric $(BENCHMARK_LATENCY_METRIC) \
			--init_duration $(BENCHMARK_INIT_DURATION) \
			--density_increment $(BENCHMARK_DENSITY_INCREMENT) \
			--min_requests $(BENCHMARK_MIN_REQUESTS) \
			--request_timeout $(BENCHMARK_REQUEST_TIMEOUT) \
			--api_endpoint $(BENCHMARK_API_ENDPOINT) \
			--images_dir $(CURDIR)/images \
			--orders_file $(CURDIR)/configs/orders.json \
			--results_dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)

benchmark-quickstart: ## Quick benchmark with workers (uses venv)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║           Dine-In Quickstart Benchmark                            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	mkdir -p $(RESULTS_DIR)
	cd src && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r ../requirements.txt && \
		python3 worker.py \
			--workers $(WORKERS) \
			--iterations $(ITERATIONS) \
			--delay $(REQUEST_DELAY) \
			--images-dir $(CURDIR)/images \
			--orders-file $(CURDIR)/configs/orders.json \
			--results-dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)
	$(MAKE) consolidate-metrics

benchmark-density-results: ## View stream density benchmark results
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Stream Density Results Summary                  ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "=== Results Directory: $(RESULTS_DIR) ==="
	@if [ -d $(RESULTS_DIR) ]; then \
		ls -la $(RESULTS_DIR)/; \
		echo ""; \
		for f in $(RESULTS_DIR)/dinein_density_*.json; do \
			if [ -f "$$f" ]; then \
				echo "--- $$f ---"; \
				cat "$$f" | python3 -m json.tool 2>/dev/null || cat "$$f"; \
				echo ""; \
			fi; \
		done; \
	else \
		echo "No results directory found. Run 'make benchmark-density' first."; \
	fi

benchmark-vlm-metrics: ## View VLM metrics from vlm_metrics_logger
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                     VLM Metrics Logs                              ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@if ls $(RESULTS_DIR)/vlm_application_metrics_*.txt 1>/dev/null 2>&1; then \
		echo "=== Application Metrics ==="; \
		cat $(RESULTS_DIR)/vlm_application_metrics_*.txt; \
		echo ""; \
	else \
		echo "No VLM application metrics found."; \
	fi
	@if ls $(RESULTS_DIR)/vlm_performance_metrics_*.txt 1>/dev/null 2>&1; then \
		echo "=== Performance Metrics ==="; \
		cat $(RESULTS_DIR)/vlm_performance_metrics_*.txt; \
	else \
		echo "No VLM performance metrics found."; \
	fi

consolidate-metrics: ## Consolidate metrics from multiple benchmark runs
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Consolidating Metrics                           ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 consolidate_multiple_run_of_metrics.py \
			--results_dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)

plot-metrics: ## Generate plots from benchmark metrics
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Generating Metrics Plots                        ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 usage_graph_plot.py \
			--results_dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)

clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Development helpers
shell:
	@echo "Opening shell in dine-in container..."
	docker exec -it dinein_app /bin/bash

test-api:
	@echo "Testing API health..."
	@curl -s http://localhost:8083/health | jq
