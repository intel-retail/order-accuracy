# Copyright © 2025 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Take-Away Order Accuracy - Makefile
# =============================================================================

.PHONY: help build up down logs restart status clean benchmark benchmark-quick benchmark-results \
        logs-vlm logs-gradio logs-frame-selector shell test-api ps exec dev-build \
        check-env show-config init-env benchmark-oa benchmark-oa-density \
        benchmark-oa-metrics benchmark-oa-results benchmark-oa-help \
        update-submodules fetch-benchmark build-benchmark \
        consolidate-metrics plot-metrics clean-images clean-containers clean-all \
        clean-metrics clean-results clean-minio

# =============================================================================
# Load Environment Variables from .env file
# =============================================================================
-include .env
export

# =============================================================================
# Configuration Variables (with defaults)
# =============================================================================

# Service mode
SERVICE_MODE ?= single
WORKERS ?= 0
SCALING_MODE ?= fixed

# VLM Backend
VLM_BACKEND ?= ovms
OVMS_ENDPOINT ?= http://ovms-vlm:8000
OVMS_MODEL_NAME ?= Qwen/Qwen2.5-VL-7B-Instruct
OPENVINO_DEVICE ?= GPU

# Semantic Service
SEMANTIC_VLM_BACKEND ?= ovms
DEFAULT_MATCHING_STRATEGY ?= hybrid
SIMILARITY_THRESHOLD ?= 0.85
OVMS_TIMEOUT ?= 60
SEMANTIC_LOG_LEVEL ?= INFO

# Benchmark configuration
CONCURRENCY_START ?= 1
CONCURRENCY_INCREMENT ?= 1
CONCURRENCY_MAX ?= 10
REQUESTS_PER_LEVEL ?= 10
REQUEST_TIMEOUT ?= 60
RESULTS_DIR ?= results
OOM_PROTECTION ?= 1

# Performance tools path
PERF_TOOLS_DIR ?= ../performance-tools/benchmark-scripts

# Version tag for registry images
TAG ?= 1.0.0

# Benchmark registry configuration
REGISTRY ?= false
REGISTRY_BENCHMARK ?= intel/retail-benchmark:$(TAG)

# Benchmark Order Accuracy settings
BENCHMARK_WORKERS ?= 1
BENCHMARK_TARGET_LATENCY_MS ?= 30000
BENCHMARK_LATENCY_METRIC ?= avg
BENCHMARK_INIT_DURATION ?= 30
BENCHMARK_DURATION ?= 200
BENCHMARK_WORKER_INCREMENT ?= 1
BENCHMARK_MIN_TRANSACTIONS ?= 3
BENCHMARK_CONTAINER ?= oa_service

# Docker compose file
COMPOSE_FILE = docker-compose.yaml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m  # No Color

# =============================================================================
# Environment Validation
# =============================================================================

# Required variables for specific modes
REQUIRED_OVMS_VARS := OVMS_ENDPOINT OVMS_MODEL_NAME
REQUIRED_OPENVINO_VARS := OPENVINO_DEVICE VLM_MODEL_PATH

check-env:
	@echo "$(BLUE)Checking environment configuration...$(NC)"
	@# Check if .env exists
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Using defaults.$(NC)"; \
		echo "$(YELLOW)Run 'make init-env' to create from template.$(NC)"; \
	fi
	@# Validate VLM_BACKEND value
	@if [ "$(VLM_BACKEND)" != "ovms" ] && [ "$(VLM_BACKEND)" != "openvino" ]; then \
		echo "$(RED)Error: VLM_BACKEND must be 'ovms' or 'openvino' (got: $(VLM_BACKEND))$(NC)"; \
		exit 1; \
	fi
	@# Validate SEMANTIC_VLM_BACKEND value
	@if [ "$(SEMANTIC_VLM_BACKEND)" != "ovms" ] && [ "$(SEMANTIC_VLM_BACKEND)" != "openvino" ]; then \
		echo "$(RED)Error: SEMANTIC_VLM_BACKEND must be 'ovms' or 'openvino' (got: $(SEMANTIC_VLM_BACKEND))$(NC)"; \
		exit 1; \
	fi
	@# Validate DEFAULT_MATCHING_STRATEGY value
	@if [ "$(DEFAULT_MATCHING_STRATEGY)" != "exact" ] && [ "$(DEFAULT_MATCHING_STRATEGY)" != "semantic" ] && [ "$(DEFAULT_MATCHING_STRATEGY)" != "hybrid" ]; then \
		echo "$(RED)Error: DEFAULT_MATCHING_STRATEGY must be 'exact', 'semantic', or 'hybrid' (got: $(DEFAULT_MATCHING_STRATEGY))$(NC)"; \
		exit 1; \
	fi
	@# Validate SERVICE_MODE value
	@if [ "$(SERVICE_MODE)" != "single" ] && [ "$(SERVICE_MODE)" != "parallel" ]; then \
		echo "$(RED)Error: SERVICE_MODE must be 'single' or 'parallel' (got: $(SERVICE_MODE))$(NC)"; \
		exit 1; \
	fi
	@# Check OVMS settings if VLM_BACKEND is ovms
	@if [ "$(VLM_BACKEND)" = "ovms" ]; then \
		if [ -z "$(OVMS_ENDPOINT)" ]; then \
			echo "$(RED)Error: OVMS_ENDPOINT is required when VLM_BACKEND=ovms$(NC)"; \
			exit 1; \
		fi; \
		if [ -z "$(OVMS_MODEL_NAME)" ]; then \
			echo "$(RED)Error: OVMS_MODEL_NAME is required when VLM_BACKEND=ovms$(NC)"; \
			exit 1; \
		fi; \
	fi
	@echo "$(GREEN)✓ Environment configuration valid$(NC)"

# =============================================================================
# Git Submodules
# =============================================================================

update-submodules:
	@echo "$(BLUE)Cloning/updating performance tool repositories...$(NC)"
	cd .. && git submodule deinit -f .
	cd .. && git submodule update --init --recursive
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

init-env:
	@if [ -f .env ]; then \
		echo "$(YELLOW).env file already exists. Backup and overwrite? [y/N]$(NC)"; \
		read -r answer; \
		if [ "$$answer" = "y" ] || [ "$$answer" = "Y" ]; then \
			cp .env .env.backup; \
			echo "$(GREEN)Backed up to .env.backup$(NC)"; \
			cp .env.example .env; \
			echo "$(GREEN)✓ Created .env from .env.example$(NC)"; \
		else \
			echo "$(YELLOW)Skipped$(NC)"; \
		fi; \
	else \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env from .env.example$(NC)"; \
	fi
	@echo "Edit .env to customize your configuration."

show-config:
	@echo "$(BLUE)Current Configuration:$(NC)"
	@echo "======================"
	@echo ""
	@echo "$(YELLOW)Service Mode:$(NC)"
	@echo "  SERVICE_MODE         = $(SERVICE_MODE)"
	@echo "  WORKERS              = $(WORKERS)"
	@echo "  SCALING_MODE         = $(SCALING_MODE)"
	@echo ""
	@echo "$(YELLOW)VLM Backend (Order Accuracy):$(NC)"
	@echo "  VLM_BACKEND          = $(VLM_BACKEND)"
	@if [ "$(VLM_BACKEND)" = "ovms" ]; then \
		echo "  OVMS_ENDPOINT        = $(OVMS_ENDPOINT)"; \
		echo "  OVMS_MODEL_NAME      = $(OVMS_MODEL_NAME)"; \
	else \
		echo "  VLM_MODEL_PATH       = $(VLM_MODEL_PATH)"; \
		echo "  OPENVINO_DEVICE      = $(OPENVINO_DEVICE)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Semantic Service:$(NC)"
	@echo "  SEMANTIC_VLM_BACKEND = $(SEMANTIC_VLM_BACKEND)"
	@echo "  MATCHING_STRATEGY    = $(DEFAULT_MATCHING_STRATEGY)"
	@echo "  SIMILARITY_THRESHOLD = $(SIMILARITY_THRESHOLD)"
	@echo "  OVMS_TIMEOUT         = $(OVMS_TIMEOUT)s"
	@echo "  LOG_LEVEL            = $(SEMANTIC_LOG_LEVEL)"
	@echo ""

# =============================================================================
# Help
# =============================================================================

help:
	@echo "$(BLUE)Take-Away Order Accuracy - Available Commands$(NC)"
	@echo "=============================================="
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make update-submodules        - Clone/update git submodules"
	@echo "  make init-env                 - Create .env from template"
	@echo "  make show-config              - Show current configuration"
	@echo "  make check-env                - Validate environment variables"
	@echo ""
	@echo "$(YELLOW)Basic Commands:$(NC)"
	@echo "  make build                    - Build Docker images"
	@echo "  make up                       - Start all services"
	@echo "  make up-parallel              - Start with RTSP streamer (parallel mode)"
	@echo "  make down                     - Stop all services"
	@echo "  make restart                  - Restart all services"
	@echo "  make status                   - Show service status"
	@echo ""
	@echo "$(YELLOW)Logs:$(NC)"
	@echo "  make logs                     - View order-accuracy logs"
	@echo "  make logs-vlm                 - View OVMS VLM logs"
	@echo "  make logs-gradio              - View Gradio UI logs"
	@echo "  make logs-semantic            - View semantic service logs"
	@echo "  make logs-all                 - View all logs (combined)"
	@echo ""
	@echo "$(YELLOW)Benchmark Commands:$(NC)"
	@echo "  make fetch-benchmark          - Pull benchmark image from registry"
	@echo "  make build-benchmark          - Build benchmark Docker image"
	@echo "  make benchmark                - Run Order Accuracy benchmark (benchmark_order_accuracy.py)"
	@echo "  make benchmark-quick          - Quick API test with curl (sanity check)"
	@echo "  make benchmark-oa-density     - Run stream density test (latency-based)"
	@echo "  make benchmark-oa-metrics     - View VLM metrics logs"
	@echo "  make benchmark-oa-results     - View all benchmark results"
	@echo "  make benchmark-oa-help        - Show benchmark options"
	@echo ""
	@echo "$(YELLOW)Metrics Processing:$(NC)"
	@echo "  make consolidate-metrics      - Consolidate benchmark metrics to CSV"
	@echo "  make plot-metrics             - Generate plots from benchmark metrics"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  make clean                    - Stop containers and remove volumes"
	@echo "  make clean-images             - Remove dangling Docker images"
	@echo "  make clean-containers         - Remove stopped containers"
	@echo "  make clean-all                - Remove all unused Docker resources"
	@echo "  make clean-metrics            - Remove metrics files only"
	@echo "  make clean-results            - Remove all results files"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make shell                    - Open shell in order-accuracy container"
	@echo "  make test-api                 - Test API health endpoints"
	@echo "  make dev-build                - Build and restart services"
	@echo ""
	@echo "$(YELLOW)Current Configuration:$(NC)"
	@echo "  SERVICE_MODE=$(SERVICE_MODE)   VLM_BACKEND=$(VLM_BACKEND)"
	@echo "  MATCHING_STRATEGY=$(DEFAULT_MATCHING_STRATEGY)  SEMANTIC_BACKEND=$(SEMANTIC_VLM_BACKEND)"
	@echo ""
	@echo "Run 'make show-config' for full configuration details"
	@echo ""

# =============================================================================
# Build & Run
# =============================================================================

build: check-env
	@echo "$(BLUE)Building Take-Away Docker images...$(NC)"
	docker compose -f $(COMPOSE_FILE) build
	@echo "$(GREEN)✓ Build complete$(NC)"

up: check-env
	@echo "$(BLUE)Starting Take-Away services...$(NC)"
	docker compose -f $(COMPOSE_FILE) up -d
	@echo ""
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "  - Order Accuracy API: http://localhost:8000"
	@echo "  - Gradio UI: http://localhost:7860"
	@echo "  - MinIO Console: http://localhost:9001"
	@echo "  - OVMS VLM: http://localhost:8001"
	@echo "  - Semantic Service: http://localhost:8080"
	@echo ""
	@echo "Run 'make logs' to view logs"

up-parallel: check-env
	@echo "$(BLUE)Starting Take-Away services with RTSP streamer ($(WORKERS) stations)...$(NC)"
	SERVICE_MODE=parallel WORKERS=$(WORKERS) docker compose -f $(COMPOSE_FILE) --profile parallel up -d
	@echo ""
	@echo "$(GREEN)✓ Services started (parallel mode with $(WORKERS) stations)$(NC)"
	@echo "  - Order Accuracy API: http://localhost:8000"
	@echo "  - Gradio UI: http://localhost:7860"
	@echo "  - RTSP Streamer: rtsp://localhost:8554"
	@echo "  - Stations: $(WORKERS) (single frame-selector handles all)"
	@echo ""

down:
	@echo "Stopping Take-Away services..."
	docker compose -f $(COMPOSE_FILE) --profile parallel down
	@echo "✓ All services stopped"

clean-minio:
	@echo "Removing MinIO data volume..."
	@docker stop oa_minio 2>/dev/null || true
	@docker rm oa_minio 2>/dev/null || true
	@docker volume rm take-away_minio_data 2>/dev/null || true
	@echo "✓ MinIO data cleared"

restart: down clean-minio up
	@echo "✓ Services restarted with clean MinIO"

# =============================================================================
# Status & Monitoring
# =============================================================================

status:
	@echo "Service Status:"
	@echo "==============="
	@docker ps --filter "name=oa_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""

ps:
	docker compose -f $(COMPOSE_FILE) ps

# =============================================================================
# Logs
# =============================================================================

logs:
	@echo "Showing order-accuracy logs (Ctrl+C to exit)..."
	docker logs -f oa_service

logs-vlm:
	@echo "Showing OVMS VLM logs (Ctrl+C to exit)..."
	docker logs -f oa_ovms_vlm

logs-gradio:
	@echo "Showing Gradio UI logs (Ctrl+C to exit)..."
	docker logs -f oa_gradio

logs-frame-selector:
	@echo "Showing frame-selector logs (Ctrl+C to exit)..."
	docker compose -f $(COMPOSE_FILE) logs -f frame-selector

logs-minio:
	@echo "Showing MinIO logs (Ctrl+C to exit)..."
	docker logs -f oa_minio

logs-semantic:
	@echo "Showing Semantic Service logs (Ctrl+C to exit)..."
	docker logs -f oa_semantic_service

logs-all:
	@echo "Showing all logs (Ctrl+C to exit)..."
	docker compose -f $(COMPOSE_FILE) logs -f

# =============================================================================
# Benchmark
# =============================================================================

benchmark: build-benchmark ## Run Order Accuracy benchmark (uses benchmark_order_accuracy.py)
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Take-Away Order Accuracy Benchmark                          ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Workers: $(BENCHMARK_WORKERS)"
	@echo "Duration: $(BENCHMARK_DURATION)s"
	@echo "Init Duration: $(BENCHMARK_INIT_DURATION)s"
	@echo "Target Device: $(OPENVINO_DEVICE)"
	@echo ""
	@mkdir -p $(RESULTS_DIR)
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 benchmark_order_accuracy.py \
			--compose_file $(CURDIR)/$(COMPOSE_FILE) \
			--workers $(BENCHMARK_WORKERS) \
			--duration $(BENCHMARK_DURATION) \
			--init_duration $(BENCHMARK_INIT_DURATION) \
			--results_dir $(CURDIR)/$(RESULTS_DIR) \
			--target_device $(OPENVINO_DEVICE) \
			--skip_perf_tools; \
		deactivate \
	)

benchmark-quick: ## Quick API test with curl (latency only)
	@echo "Running quick API test with test video..."
	@mkdir -p results
	@bash -c '\
		if [ ! -f storage/videos/test.mp4 ]; then \
			echo "Error: No test video found at storage/videos/test.mp4"; \
			echo "Please add a test video first."; \
			exit 1; \
		fi; \
		echo "=== Starting Quick API Test ==="; \
		echo "Video: storage/videos/test.mp4"; \
		echo ""; \
		START_TIME=$$(date +%s%3N); \
		RESPONSE=$$(curl -s -X POST http://localhost:8000/upload-video \
			-F "file=@storage/videos/test.mp4" \
			-F "video_id=benchmark_test"); \
		END_TIME=$$(date +%s%3N); \
		LATENCY=$$((END_TIME - START_TIME)); \
		echo ""; \
		echo "=== API Test Results ==="; \
		echo "$$RESPONSE" | jq . 2>/dev/null || echo "$$RESPONSE"; \
		echo ""; \
		echo "Latency: $${LATENCY}ms"; \
		TIMESTAMP=$$(date "+%Y-%m-%d %H:%M:%S"); \
		echo "[$$TIMESTAMP] latency_ms=$$LATENCY" >> results/benchmark_performance.log; \
		echo "Results logged to results/benchmark_performance.log"; \
	'

benchmark-stream-density:
	@if [ "$(OOM_PROTECTION)" = "0" ]; then \
		echo "╔════════════════════════════════════════════════════════════╗"; \
		echo "║ WARNING: OOM Protection DISABLED                           ║"; \
		echo "║                                                            ║"; \
		echo "║ This test may:                                             ║"; \
		echo "║ • Exhaust system memory                                    ║"; \
		echo "║ • Cause system instability                                 ║"; \
		echo "║ • Require hard reboot                                      ║"; \
		echo "║                                                            ║"; \
		echo "║ Press Ctrl+C to cancel, or wait 5 seconds...               ║"; \
		echo "╚════════════════════════════════════════════════════════════╝"; \
		sleep 5; \
	fi
	@echo "Running stream density benchmark..."
	@mkdir -p $(RESULTS_DIR)
	@if [ -f benchmark_density.py ]; then \
		python3 -m venv venv 2>/dev/null || true && \
		. venv/bin/activate && \
		pip3 install -q aiohttp psutil && \
		python3 benchmark_density.py \
			--concurrency_start $(CONCURRENCY_START) \
			--concurrency_max $(CONCURRENCY_MAX) \
			--concurrency_increment $(CONCURRENCY_INCREMENT) \
			--requests_per_level $(REQUESTS_PER_LEVEL) \
			--timeout $(REQUEST_TIMEOUT) \
			--results_dir $(RESULTS_DIR) \
			--oom_protection $(OOM_PROTECTION); \
		deactivate; \
	else \
		echo "benchmark_density.py not found. Skipping stream density test."; \
	fi

benchmark-results:
	@echo "=== Benchmark Results ==="
	@if [ -f results/benchmark_performance.log ]; then \
		cat results/benchmark_performance.log; \
	else \
		echo "No benchmark results found. Run 'make benchmark' first."; \
	fi

# =============================================================================
# Order Accuracy Performance Benchmarks (using performance-tools scripts)
# =============================================================================

benchmark-oa-help:
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║         Order Accuracy Benchmark Commands                         ║"
	@echo "╠═══════════════════════════════════════════════════════════════════╣"
	@echo "║ make benchmark-oa           - Run fixed workers benchmark         ║"
	@echo "║ make benchmark-oa-density   - Run stream density test (latency)   ║"
	@echo "║ make benchmark-oa-metrics   - View VLM metrics logs               ║"
	@echo "║ make benchmark-oa-results   - View all benchmark results          ║"
	@echo "╠═══════════════════════════════════════════════════════════════════╣"
	@echo "║ Configuration Variables:                                          ║"
	@echo "║   BENCHMARK_WORKERS=$(BENCHMARK_WORKERS) (fixed workers benchmark)                      ║"
	@echo "║   BENCHMARK_TARGET_LATENCY_MS=$(BENCHMARK_TARGET_LATENCY_MS)ms (latency threshold)           ║"
	@echo "║   BENCHMARK_MIN_TRANSACTIONS=$(BENCHMARK_MIN_TRANSACTIONS) (transactions per iteration)      ║"
	@echo "║   BENCHMARK_INIT_DURATION=$(BENCHMARK_INIT_DURATION)s (warmup time)                      ║"
	@echo "║   BENCHMARK_WORKER_INCREMENT=$(BENCHMARK_WORKER_INCREMENT) (workers to add per iteration)    ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"

benchmark-oa: ## Run Order Accuracy benchmark with fixed workers
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Take-Away Order Accuracy Benchmark                          ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Workers: $(BENCHMARK_WORKERS)"
	@echo "Duration: $(BENCHMARK_DURATION)s"
	@echo "Init Duration: $(BENCHMARK_INIT_DURATION)s"
	@echo "Target Device: $(OPENVINO_DEVICE)"
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
			--duration $(BENCHMARK_DURATION) \
			--init_duration $(BENCHMARK_INIT_DURATION) \
			--results_dir $(CURDIR)/$(RESULTS_DIR) \
			--target_device $(OPENVINO_DEVICE) \
			--skip_perf_tools; \
		deactivate \
	)

benchmark-oa-density: ## Run Order Accuracy stream density benchmark (latency-based)
	@if [ "$(OOM_PROTECTION)" = "0" ]; then \
		echo "╔════════════════════════════════════════════════════════════╗"; \
		echo "║ WARNING: OOM Protection DISABLED                           ║"; \
		echo "║ This test may exhaust system memory.                       ║"; \
		echo "║ Press Ctrl+C to cancel, or wait 5 seconds...               ║"; \
		echo "╚════════════════════════════════════════════════════════════╝"; \
		sleep 5; \
	fi
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       Order Accuracy Stream Density - Latency Based               ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo "Target Latency: $(BENCHMARK_TARGET_LATENCY_MS)ms ($(shell echo $$(($(BENCHMARK_TARGET_LATENCY_MS)/1000)))s)"
	@echo "Latency Metric: $(BENCHMARK_LATENCY_METRIC)"
	@echo "Init Duration: $(BENCHMARK_INIT_DURATION)s"
	@echo "Min Transactions: $(BENCHMARK_MIN_TRANSACTIONS)"
	@echo "Worker Increment: $(BENCHMARK_WORKER_INCREMENT)"
	@echo ""
	mkdir -p $(RESULTS_DIR)
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		OOM_PROTECTION=$(OOM_PROTECTION) python3 stream_density_latency_oa.py \
			--compose_file $(CURDIR)/$(COMPOSE_FILE) \
			--target_latency_ms $(BENCHMARK_TARGET_LATENCY_MS) \
			--latency_metric $(BENCHMARK_LATENCY_METRIC) \
			--init_duration $(BENCHMARK_INIT_DURATION) \
			--min_transactions $(BENCHMARK_MIN_TRANSACTIONS) \
			--worker_increment $(BENCHMARK_WORKER_INCREMENT) \
			--results_dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)

benchmark-oa-metrics: ## View VLM metrics from vlm_metrics_logger
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                     VLM Metrics Logs                              ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@if ls results/vlm_application_metrics_*.txt 1>/dev/null 2>&1; then \
		echo "=== Application Metrics ==="; \
		cat results/vlm_application_metrics_*.txt; \
		echo ""; \
	else \
		echo "No VLM application metrics found."; \
	fi
	@if ls results/vlm_performance_metrics_*.txt 1>/dev/null 2>&1; then \
		echo "=== Performance Metrics ==="; \
		cat results/vlm_performance_metrics_*.txt; \
	else \
		echo "No VLM performance metrics found."; \
	fi

benchmark-oa-results: ## View all Order Accuracy benchmark results
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                   Benchmark Results Summary                       ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "=== Results Directory: $(RESULTS_DIR) ==="
	@if [ -d $(RESULTS_DIR) ]; then \
		ls -la $(RESULTS_DIR)/; \
		echo ""; \
		for f in $(RESULTS_DIR)/*.json; do \
			if [ -f "$$f" ]; then \
				echo "--- $$f ---"; \
				cat "$$f" | python3 -m json.tool 2>/dev/null || cat "$$f"; \
				echo ""; \
			fi; \
		done; \
	else \
		echo "No results directory found. Run a benchmark first."; \
	fi
	@echo ""
	@echo "=== VLM Metrics ==="
	@$(MAKE) --no-print-directory benchmark-oa-metrics

# =============================================================================
# Metrics Processing
# =============================================================================

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
	cd $(PERF_TOOLS_DIR) && \
	( \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt && \
		python3 usage_graph_plot.py \
			--results_dir $(CURDIR)/$(RESULTS_DIR); \
		deactivate \
	)

# =============================================================================
# Development Helpers
# =============================================================================

shell:
	@echo "Opening shell in order-accuracy container..."
	docker exec -it oa_service /bin/bash

exec:
	docker exec -it oa_service $(CMD)

test-api:
	@echo "Testing API endpoints..."
	@echo ""
	@echo "=== Health Check ==="
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
	@echo ""
	@echo "=== OVMS Config ==="
	@curl -s http://localhost:8001/v1/config | jq . 2>/dev/null || curl -s http://localhost:8001/v1/config
	@echo ""

dev-build: build restart
	@echo "✓ Development build complete"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning up containers and volumes..."
	docker compose -f $(COMPOSE_FILE) --profile parallel down -v
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

clean-all: clean clean-images
	@echo "Removing all unused Docker resources..."
	docker system prune -f
	@echo "Cleaning up build cache..."
	@docker builder prune -f
	@echo "$(GREEN)✓ Full cleanup complete$(NC)"

clean-metrics:
	@echo "Cleaning up metrics files..."
	@rm -f $(RESULTS_DIR)/vlm_application_metrics_*.txt $(RESULTS_DIR)/vlm_performance_metrics_*.txt
	@rm -f $(RESULTS_DIR)/*.csv
	@echo "$(GREEN)✓ Metrics files cleaned$(NC)"

clean-results:
	@echo "Cleaning up results directory..."
	@rm -rf $(RESULTS_DIR)/*
	@echo "$(GREEN)✓ Results directory cleaned$(NC)"

# =============================================================================
# Service-specific Commands
# =============================================================================

restart-vlm:
	@echo "Restarting OVMS VLM service..."
	docker compose -f $(COMPOSE_FILE) restart ovms-vlm

restart-service:
	@echo "Restarting order-accuracy service..."
	docker compose -f $(COMPOSE_FILE) restart order-accuracy

restart-gradio:
	@echo "Restarting Gradio UI..."
	docker compose -f $(COMPOSE_FILE) restart gradio-ui

# Check OVMS model status
check-model:
	@echo "Checking OVMS model status..."
	@curl -s http://localhost:8001/v1/config | jq '."$(OVMS_MODEL_NAME)"' 2>/dev/null || \
		echo "OVMS not responding. Is the service running?"

# View results
results:
	@echo "=== Order Accuracy Results ==="
	@if [ -f results/station_1_results.jsonl ]; then \
		tail -20 results/station_1_results.jsonl | jq .; \
	else \
		echo "No results found yet."; \
	fi
