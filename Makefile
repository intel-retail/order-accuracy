# Copyright © 2025 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


.PHONY: update-submodules run-demo


HOST_IP := $(shell hostname -I | cut -d' ' -f1 2>/dev/null || ipconfig getifaddr en0)
APP_HOST_PORT=7860

export VSS_IP := $(HOST_IP)
export MINIO_SERVER=$(HOST_IP):4001
export RABBITMQ_DEFAULT_USER=$(RABBITMQ_USER)
export RABBITMQ_DEFAULT_PASS=$(RABBITMQ_PASSWORD)
export USER_GROUP_ID=$(id -g)
export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}')
export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}')


update-submodules:
	@echo "🔹 Cloning performance tool repositories"
	@git submodule deinit -f . || true
	@git submodule update --init --recursive --progress
	@git submodule update --remote --merge --progress
	@echo "✅ Submodules updated (if any present)."


build:
	@echo "Building video-ingestion app"
	@cd edge-ai-libraries/microservices/vlm-openvino-serving && \
		echo y | docker compose -f docker/compose.yaml build
	@echo "vlm-openvino-serving build completed........"
	@cd edge-ai-libraries/sample-applications/video-search-and-summarization/video-ingestion && \
		echo y | docker compose -f docker/compose.yaml build
	@echo "Video Ingestion build completed........"

run-demo:	
	@echo "🔹 Running setup.sh with --summary"
	cd edge-ai-libraries/sample-applications/video-search-and-summarization && \
	chmod +x setup.sh && ./setup.sh --summary || { echo "setup.sh failed, aborting run-demo."; exit 1; }
	@PID=$$(lsof -t -i:$(APP_HOST_PORT)); \
	if [ -n "$$PID" ]; then \
		kill -9 $$PID; \
	fi
    @if [ -d "clips" ]; then \
		rm -rf clips/; \
		echo "✅ clips/ folder and files deleted"; \
	else \
		echo "ℹ️  clips/ folder does not exist"; \
	fi
    rm session_metrics.json 
	@echo "Starting UI app in background..."
	nohup uv run python src/main.py > order_accuracy_app.log 2>&1 &		
	@echo "Order Accuracy VLM Application is up: Access the UI at: http://${HOST_IP}:${APP_HOST_PORT}"


down:
	@echo "🔹 Running setup.sh with --down"
	cd edge-ai-libraries/sample-applications/video-search-and-summarization && \
	./setup.sh --down
	@echo "🔹 Cleaning up clips folder"
	@if [ -d "clips" ]; then \
		rm -rf clips/; \
		echo "✅ clips/ folder and files deleted"; \
	else \
		echo "ℹ️  clips/ folder does not exist"; \
	fi
	@PID=$$(lsof -t -i:$(APP_HOST_PORT)); \
	if [ -n "$$PID" ]; then \
		echo "Order Accuracy VLM Application has been stopped successfully....."; \
		kill -9 $$PID; \
	else \
		echo "Order Accuracy VLM Application is not running......."; \
	fi
	

