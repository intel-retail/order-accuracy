.PHONY: build-all build-benchmark run-benchmark start-performance-tools stop-performance-tools

build-all: build-benchmark

build-benchmark:
	echo "Building benchmark container HTTPS_PROXY=${HTTPS_PROXY} HTTP_PROXY=${HTTP_PROXY}"
	log_dir=./results docker compose build

run-benchmark:
	log_dir=./results docker compose up --build

start-performance-tools:
	log_dir=./results docker compose up -d

stop-performance-tools:
	log_dir=./results docker compose down