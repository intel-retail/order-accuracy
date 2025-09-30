# Copyright © 2024 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: build build-realsense run down
.PHONY: build-telegraf run-telegraf run-portainer clean-all clean-results clean-telegraf clean-models down-portainer
.PHONY: download-models clean-test run-demo run-headless

MKDOCS_IMAGE ?= asc-mkdocs
PIPELINE_COUNT ?= 1
INIT_DURATION ?= 30
TARGET_FPS ?= 8
CONTAINER_NAMES ?= gst0
DOCKER_COMPOSE ?= docker-compose.yml
DOCKER_COMPOSE_SENSORS ?= docker-compose-sensors.yml
RETAIL_USE_CASE_ROOT ?= $(PWD)
DENSITY_INCREMENT ?= 1
RESULTS_DIR ?= $(shell pwd)/benchmark

download-models: | build-download-models run-download-models

build-download-models:
	docker build  --build-arg  HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t modeldownloader -f docker/Dockerfile.downloader .

run-download-models:
	docker run --rm -e HTTP_PROXY=${HTTP_PROXY} -e HTTPS_PROXY=${HTTPS_PROXY} -e MODELS_DIR=/workspace/models -v "$(shell pwd)/models:/workspace/models" modeldownloader

download-sample-videos:
	cd performance-tools/benchmark-scripts && ./download_sample_videos.sh

clean-models:
	@find ./models/ -mindepth 1 -maxdepth 1 -type d -exec sudo rm -r {} \;

run-smoke-tests: | download-models update-submodules download-sample-videos
	@echo "Running smoke tests for OVMS profiles"
	@./smoke_test.sh > smoke_tests_output.log
	@echo "results of smoke tests recorded in the file smoke_tests_output.log"
	@grep "Failed" ./smoke_tests_output.log || true
	@grep "===" ./smoke_tests_output.log || true

update-submodules:
	@git submodule update --init --recursive
	@git submodule update --remote --merge

build: download-models update-submodules download-qsr-video download-sample-videos compress-qsr-video
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t dlstreamer:dev -f docker/Dockerfile.pipeline .

run:
	docker compose -f src/$(DOCKER_COMPOSE) up -d

run-render-mode:
	@if [ -z "$(DISPLAY)" ] || ! echo "$(DISPLAY)" | grep -qE "^:[0-9]+(\.[0-9]+)?$$"; then \
		echo "ERROR: Invalid or missing DISPLAY environment variable."; \
		echo "Please set DISPLAY in the format ':<number>' (e.g., ':0')."; \
		echo "Usage: make <target> DISPLAY=:<number>"; \
		echo "Example: make $@ DISPLAY=:0"; \
		exit 1; \
	fi
	@echo "Using DISPLAY=$(DISPLAY)"
	@xhost +local:docker
	@RENDER_MODE=1 docker compose -f src/$(DOCKER_COMPOSE) up -d


down:
	docker compose -f src/$(DOCKER_COMPOSE) down

down-sensors:
	docker compose -f src/${DOCKER_COMPOSE_SENSORS} down

download-qsr-video:
	@echo "Downloading additional QSR videos..."
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t qsr-video-downloader:vid -f docker/Dockerfile.qsrDownloader .
	docker run --rm \
        -v $(shell pwd)/config/sample-videos:/sample-videos \
         qsr-video-downloader:vid

compress-qsr-video:
	@echo "Increasing the duration and Compressing the QSR video..."
	docker build --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg HTTP_PROXY=${HTTP_PROXY} -t qsr-video-compressor:0.0 -f docker/Dockerfile.videoDurationIncrease .
	docker run --rm \
        -v $(shell pwd)/config/sample-videos:/sample-videos \
         qsr-video-compressor:0.0

run-demo:
	@echo "Building order-accuracy app"	
	$(MAKE) build
	@echo Running order-accuracy pipeline
	@if [ "$(RENDER_MODE)" != "0" ]; then \
		$(MAKE) run-render-mode; \
	else \
		$(MAKE) run; \
	fi

run-headless: | download-models update-submodules download-sample-videos
	@echo "Building order accuracy app"
	$(MAKE) build
	@echo Running order accuracy pipeline
	$(MAKE) run

build-benchmark:
	cd performance-tools && $(MAKE) build-benchmark-docker

benchmark: build-benchmark download-models download-sample-videos
	cd performance-tools/benchmark-scripts && \
	pip3 install -r requirements.txt && \
	python3 benchmark.py --compose_file ../../src/docker-compose.yml --pipeline $(PIPELINE_COUNT) --results_dir $(RESULTS_DIR)

benchmark-stream-density: build-benchmark download-models
	@if [ "$(OOM_PROTECTION)" = "0" ]; then \
        	echo "╔════════════════════════════════════════════════════════════╗";\
		echo "║ WARNING                                                    ║";\
		echo "║                                                            ║";\
		echo "║ OOM Protection is DISABLED. This test may:                 ║";\
		echo "║ • Cause system instability or crashes                      ║";\
		echo "║ • Require hard reboot if system becomes unresponsive       ║";\
		echo "║ • Result in data loss in other applications                ║";\
		echo "║                                                            ║";\
		echo "║ Press Ctrl+C now to cancel, or wait 5 seconds...           ║";\
		echo "╚════════════════════════════════════════════════════════════╝";\
		sleep 5;\
    fi
	cd performance-tools/benchmark-scripts && \
	python3 benchmark.py \
	  --compose_file ../../src/docker-compose.yml \
	  --init_duration $(INIT_DURATION) \
	  --target_fps $(TARGET_FPS) \
	  --container_names $(CONTAINER_NAMES) \
	  --density_increment $(DENSITY_INCREMENT) \
	  --results_dir $(RESULTS_DIR)

benchmark-quickstart:
	DEVICE_ENV=res/all-gpu.env RENDER_MODE=0 $(MAKE) benchmark
	$(MAKE) consolidate-metrics

clean-results:
	rm -rf results/*

clean-all: 
	docker rm -f $(docker ps -aq)

docs: clean-docs
	mkdocs build
	mkdocs serve -a localhost:8008

docs-builder-image:
	docker build \
		-f Dockerfile.docs \
		-t $(MKDOCS_IMAGE) \
		.

build-docs: docs-builder-image
	docker run --rm \
		-u $(shell id -u):$(shell id -g) \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE) \
		build

serve-docs: docs-builder-image
	docker run --rm \
		-it \
		-u $(shell id -u):$(shell id -g) \
		-p 8008:8000 \
		-v $(PWD):/docs \
		-w /docs \
		$(MKDOCS_IMAGE)

clean-docs:
	rm -rf docs/

consolidate-metrics:
	cd performance-tools/benchmark-scripts && \
	( \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt && \
	python3 consolidate_multiple_run_of_metrics.py --root_directory $(RESULTS_DIR) --output $(RESULTS_DIR)/metrics.csv && \
	deactivate \
	)

plot-metrics:
	cd performance-tools/benchmark-scripts && \
	( \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt && \
	python3 usage_graph_plot.py --dir $(RESULTS_DIR)  && \
	deactivate \
	)
