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
METRICS_DIR ?= results
OOM_PROTECTION ?= 1

# Performance tools path
PERF_TOOLS_DIR ?= ../performance-tools/benchmark-scripts

# Dine-In Stream Density Benchmark Settings
BENCHMARK_TARGET_LATENCY_MS ?= 25000
BENCHMARK_LATENCY_METRIC ?= avg
BENCHMARK_INIT_DURATION ?= 60
BENCHMARK_DENSITY_INCREMENT ?= 1
BENCHMARK_MIN_REQUESTS ?= 3
BENCHMARK_REQUEST_TIMEOUT ?= 300
BENCHMARK_API_ENDPOINT ?= http://localhost:8083

# Short-name aliases (documented in get-started.md)
# Allows: make benchmark-density TARGET_LATENCY_MS=15000 LATENCY_METRIC=max
# Both forms are equivalent; BENCHMARK_* takes precedence if both are set.
ifdef TARGET_LATENCY_MS
BENCHMARK_TARGET_LATENCY_MS := $(TARGET_LATENCY_MS)
endif
ifdef LATENCY_METRIC
BENCHMARK_LATENCY_METRIC := $(LATENCY_METRIC)
endif
ifdef DENSITY_INCREMENT
BENCHMARK_DENSITY_INCREMENT := $(DENSITY_INCREMENT)
endif
ifdef INIT_DURATION
BENCHMARK_INIT_DURATION := $(INIT_DURATION)
endif
ifdef MIN_REQUESTS
BENCHMARK_MIN_REQUESTS := $(MIN_REQUESTS)
endif
ifdef REQUEST_TIMEOUT
BENCHMARK_REQUEST_TIMEOUT := $(REQUEST_TIMEOUT)
endif
ifdef API_ENDPOINT
BENCHMARK_API_ENDPOINT := $(API_ENDPOINT)
endif

# Docker compose file
COMPOSE_FILE ?= docker-compose.yml

# Version tag for registry images
TAG ?= 2026.0-rc1

# Dine-In image configuration
DINEIN_IMAGE_NAME ?= intel/order-accuracy-dine-in
DINEIN_IMAGE ?= $(DINEIN_IMAGE_NAME):$(TAG)

# Registry configuration
# REGISTRY=false : Build and use local images
# REGISTRY=true  : Use images from registry (default for up, pull for build)
REGISTRY ?= true

# Benchmark registry configuration
REGISTRY_BENCHMARK ?= intel/retail-benchmark:$(TAG)

# Benchmark Order Accuracy settings
BENCHMARK_WORKERS ?= 1
BENCHMARK_ITERATIONS ?= 1
BENCHMARK_DURATION ?= 180
BENCHMARK_TARGET_FPS ?= 15.0
BENCHMARK_TARGET_DEVICE ?= GPU

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

.PHONY: help build up down logs restart benchmark benchmark-results benchmark-metrics \
        benchmark-stream-density benchmark-density benchmark-density-results clean \
        shell test-api benchmark-test benchmark-single \
        worker worker-start worker-stop worker-logs worker-results \
        update-submodules fetch-benchmark build-benchmark \
        consolidate-metrics plot-metrics clean-images clean-containers clean-all

help:
	@echo "$(BLUE)Dine-In Order Accuracy - Available Commands$(NC)"
	@echo "=============================================="
	@echo ""
	@echo "$(YELLOW)Image Configuration:$(NC)"
	@echo "  REGISTRY=false              - Build/use local images"
	@echo "  REGISTRY=true (default)     - Use images from registry"
	@echo "  TAG=$(TAG)            - Image version tag"
	@echo "  Image: $(DINEIN_IMAGE)"
	@echo ""
	@echo "$(YELLOW)Basic Commands:$(NC)"
	@echo "  make update-submodules        - Clone/update git submodules"
	@echo "  make build                    - Pull image from registry (default)"
	@echo "  make build REGISTRY=false     - Build Docker image locally"
	@echo "  make up                       - Start services with registry image (default)"
	@echo "  make up REGISTRY=false        - Start services with local image"
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
	@echo "  make fetch-benchmark          - Pull benchmark image from registry"
	@echo "  make build-benchmark          - Build benchmark Docker image"
	@echo "  make benchmark                - Run Order Accuracy benchmark (uses benchmark_order_accuracy.py)"
	@echo "  make benchmark-single         - Quick single image test via curl"
	@echo "  make benchmark-test           - Quick single image test via API"
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
	@echo "$(YELLOW)Metrics:$(NC)"
	@echo "  make consolidate-metrics      - Consolidate benchmark metrics to CSV"
	@echo "  make plot-metrics             - Generate plots from benchmark metrics"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  make clean                    - Stop containers and remove volumes"
	@echo "  make clean-images             - Remove dangling Docker images"
	@echo "  make clean-containers         - Remove stopped containers"
	@echo "  make clean-all                - Remove all unused Docker resources"
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

update-submodules:
	@echo "$(BLUE)Cloning/updating performance tool repositories...$(NC)"
	cd .. && git submodule deinit -f .
	cd .. && git submodule update --init --recursive --remote
	@echo "$(GREEN)Submodules updated successfully.$(NC)"

fetch-benchmark:
	@echo "$(BLUE)Fetching benchmark image from registry...$(NC)"
	docker pull $(REGISTRY_BENCHMARK)
	@echo "$(GREEN)Benchmark image ready$(NC)"

build-benchmark:
	@echo "$(BLUE)Building benchmark Docker image...$(NC)"
	@if [ "$(REGISTRY)" = "true" ]; then \
		$(MAKE) fetch-benchmark; \
	else \
		cd ../performance-tools && $(MAKE) build-benchmark-docker; \
	fi

build:
	@echo "Building dine-in Docker images..."
	@if [ "$(REGISTRY)" = "false" ]; then \
		echo "$(BLUE)Building locally and tagging as $(DINEIN_IMAGE)...$(NC)"; \
		docker compose build; \
		docker tag $$(docker compose images dine-in -q 2>/dev/null || echo "dine-in-dine-in") $(DINEIN_IMAGE) 2>/dev/null || \
			docker build -t $(DINEIN_IMAGE) .; \
		echo "$(GREEN)✓ Local image built and tagged as $(DINEIN_IMAGE)$(NC)"; \
	else \
		echo "$(BLUE)Pulling image from registry: $(DINEIN_IMAGE)...$(NC)"; \
		docker pull $(DINEIN_IMAGE) || echo "$(YELLOW)Warning: Could not pull $(DINEIN_IMAGE). Build locally with REGISTRY=false$(NC)"; \
	fi

up:
	@echo "Starting dine-in services..."
	@mkdir -p results && chmod 777 results
	@mkdir -p metrics && chmod 777 metrics
	@if [ "$(REGISTRY)" = "false" ]; then \
		echo "$(BLUE)Using locally built image: $(DINEIN_IMAGE)$(NC)"; \
		DINEIN_IMAGE=$(DINEIN_IMAGE) docker compose up -d --no-build; \
	else \
		echo "$(BLUE)Using registry image: $(DINEIN_IMAGE)$(NC)"; \
		DINEIN_IMAGE=$(DINEIN_IMAGE) docker compose up -d --no-build; \
	fi
	@echo "$(GREEN)Services started:$(NC)"
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
	cd src && PYTHONPATH=$(CURDIR)/../performance-tools/benchmark-scripts:$$PYTHONPATH python3 worker.py \
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
	cd src && PYTHONPATH=$(CURDIR)/../performance-tools/benchmark-scripts:$$PYTHONPATH python3 worker.py \
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
benchmark-single: ## Test with specific image via API (IMAGE_ID=MCD-1001)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║           Dine-In Single Image Test - $(IMAGE_ID)                 ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@mkdir -p $(RESULTS_DIR)
	@if [ -z "$(IMAGE_ID)" ]; then \
		echo "$(RED)Error: IMAGE_ID not specified$(NC)"; \
		echo "Usage: make benchmark-single IMAGE_ID=MCD-1001"; \
		exit 1; \
	fi
	@echo "Testing image: $(IMAGE_ID)"
	@bash -c '\
                RED="\033[0;31m"; GREEN="\033[0;32m"; NC="\033[0m"; \
                IMAGE_FILE=""; \
                for EXT in jpg jpeg png; do \
                        if [ -f "images/$(IMAGE_ID).$$EXT" ]; then \
                                IMAGE_FILE="images/$(IMAGE_ID).$$EXT"; \
                                break; \
                        fi; \
                done; \
                if [ -z "$$IMAGE_FILE" ]; then \
                        printf "$${RED}Error: No image found for $(IMAGE_ID) (tried .jpg, .jpeg, .png)$${NC}\n"; \
                        exit 1; \
                fi; \
                ORDER_JSON=$$(jq -c ".orders[] | select(.image_id == \"$(IMAGE_ID)\")" configs/orders.json); \
                if [ -z "$$ORDER_JSON" ]; then \
                        printf "$${RED}Error: Order not found for $(IMAGE_ID)$${NC}\n"; \
                        exit 1; \
                fi; \
                ORDER_MANIFEST=$$(echo $$ORDER_JSON | jq -c "{items: [.items_ordered[] | {name: .item, quantity: .quantity}]}"); \
                echo "Order:"; \
                echo "$$ORDER_MANIFEST" | jq . 2>/dev/null || echo "$$ORDER_MANIFEST"; \
                echo ""; \
                START_TIME=$$(date +%s%3N); \
                RESPONSE=$$(curl -s -w "\n__HTTP_STATUS__:%{http_code}" \
                        -X POST $(BENCHMARK_API_ENDPOINT)/api/validate \
                        -F "image=@$$IMAGE_FILE" \
                        -F "order=$$ORDER_MANIFEST"); \
                HTTP_STATUS=$$(echo "$$RESPONSE" | grep "__HTTP_STATUS__" | cut -d: -f2); \
                BODY=$$(echo "$$RESPONSE" | sed "s/__HTTP_STATUS__:.*//"); \
                END_TIME=$$(date +%s%3N); \
                LATENCY=$$((END_TIME - START_TIME)); \
                echo "=== Response (HTTP $$HTTP_STATUS) ==="; \
                if echo "$$BODY" | jq . 2>/dev/null; then \
                        : ; \
                else \
                        printf "$${RED}Warning: Response is not valid JSON$${NC}\n"; \
                        echo "Raw response: $$BODY"; \
                fi; \
                echo ""; \
                printf "$${GREEN}Latency: $${LATENCY}ms$${NC}\n"; \
        '

benchmark:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Dine-In Order Accuracy Benchmark                            ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Workers: $(BENCHMARK_WORKERS)"
	@echo "Iterations: $(BENCHMARK_ITERATIONS)"
	@echo "Target Device: $(BENCHMARK_TARGET_DEVICE)"
	@echo "Results Dir: $(RESULTS_DIR)"
	@echo ""
	mkdir -p $(RESULTS_DIR)
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 benchmark_order_accuracy.py \
			--compose_file $(CURDIR)/$(COMPOSE_FILE) \
			--workers $(BENCHMARK_WORKERS) \
			--iterations $(BENCHMARK_ITERATIONS) \
			--profile benchmark \
			--init_duration $(BENCHMARK_INIT_DURATION) \
			--duration $(BENCHMARK_DURATION) \
			--results_dir $(CURDIR)/$(RESULTS_DIR) \
			--target_device $(BENCHMARK_TARGET_DEVICE) \
			--skip_perf_tools; \
		deactivate \
	)

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

benchmark-stream-density: ## Run Dine-In stream density benchmark (image-based latency)
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
		PYTHONPATH=$(CURDIR)/../performance-tools/benchmark-scripts:$$PYTHONPATH python3 worker.py \
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
			--root_directory $(CURDIR)/$(RESULTS_DIR) \
			--output $(CURDIR)/$(RESULTS_DIR)/consolidated_metrics.csv; \
		deactivate \
	)

plot-metrics: ## Generate plots from benchmark metrics
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Generating Metrics Plots                        ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@chmod -R ugo+rw $(CURDIR)/$(METRICS_DIR) 2>/dev/null || true
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 parse_qmassa_metrics_to_json.py \
			--directory $(CURDIR)/$(METRICS_DIR) && \
		python3 usage_graph_plot.py \
			--dir $(CURDIR)/$(METRICS_DIR); \
		deactivate \
	)

clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-images:
	@echo "Cleaning up dangling Docker images..."
	@docker image prune -f
	@echo "Cleaning up unused Docker images..."
	@docker images -f "dangling=true" -q | xargs -r docker rmi
	@echo "$(GREEN)Dangling images cleaned up$(NC)"

clean-containers:
	@echo "Cleaning up stopped containers..."
	@docker container prune -f
	@echo "$(GREEN)Stopped containers cleaned up$(NC)"

clean-all:
	@echo "Cleaning up all unused Docker resources..."
	@docker system prune -f
	@echo "Cleaning up build cache..."
	@docker builder prune -f
	@echo "$(GREEN)All unused Docker resources cleaned up$(NC)"

# Development helpers
shell:
	@echo "Opening shell in dine-in container..."
	docker exec -it dinein_app /bin/bash

test-api:
	@echo "Testing API health..."
	@curl -s http://localhost:8083/health | jq
