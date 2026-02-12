.PHONY: help start stop logs build clean status

# Default values
STATIONS ?= 1
COMPOSE_FILE = docker-compose.unified.yaml
MULTI_COMPOSE = docker-compose.multi-station.yaml

help: ## Show this help message
	@echo "Order Accuracy System - Multi-Station Support"
	@echo ""
	@echo "Usage:"
	@echo "  make start              # Start single station (default)"
	@echo "  make start STATIONS=4   # Start 4 stations"
	@echo "  make stop               # Stop all services"
	@echo "  make logs STATION=1     # View logs for station 1"
	@echo "  make status             # Show status of all services"
	@echo "  make build              # Build service images"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

start: ## Start order accuracy system (use STATIONS=N for multi-station)
ifeq ($(STATIONS),1)
	@echo "Starting single station mode..."
	docker compose -f $(COMPOSE_FILE) --profile ovms up -d
	@echo ""
	@echo "✓ Single station started"
	@echo "  - Order Accuracy API: http://localhost:8000"
	@echo "  - Gradio UI: http://localhost:7860"
	@echo ""
	@echo "View logs: make logs"
else
	@echo "Starting $(STATIONS) stations..."
	@# Start shared services first
	docker compose -f $(COMPOSE_FILE) up -d minio ovms-vlm semantic-service gradio-ui
	@sleep 3
	@# Start stations using the launcher script
	./launch_multi_station.sh --stations $(STATIONS)
	@echo ""
	@echo "✓ $(STATIONS) stations started"
	@$(MAKE) --no-print-directory status
endif

stop: ## Stop all services
	@echo "Stopping all services..."
	@# Stop multi-station containers
	@docker ps -a --filter "name=oa_service_station_" -q | xargs -r docker stop 2>/dev/null || true
	@docker ps -a --filter "name=oa_frame_selector_station_" -q | xargs -r docker rm 2>/dev/null || true
	@# Stop unified services
	docker compose -f $(COMPOSE_FILE) down
	@echo "✓ All services stopped"

logs: ## View logs (use STATION=N to specify station, default=1)
ifndef STATION
	docker logs -f oa_service
else
	docker logs -f oa_service_station_$(STATION)
endif

logs-fs: ## View frame-selector logs (use STATION=N)
ifndef STATION
	docker logs -f oa_frame_selector
else
	docker logs -f oa_frame_selector_station_$(STATION)
endif

results: ## View results summary for all stations
	python3 view_results.py

results-station: ## View detailed results for specific station (use STATION=N)
ifndef STATION
	@echo "Please specify STATION number: make results-station STATION=1"
else
	python3 view_results.py --station station_$(STATION)
endif

results-live: ## Live monitoring of all station results
	python3 view_results.py --live

status: ## Show status of all running services
	@echo "Service Status:"
	@echo "==============="
	@docker ps --filter "name=oa_" --filter "name=ovms" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20
	@echo ""
	@echo "Station Endpoints:"
	@docker ps --filter "name=oa_service_station_" --format "  - Station {{.Names}}: http://localhost:{{.Ports}}" | sed 's/.*station_//' | sed 's/:.*//' | sed 's/0.0.0.0://' | sed 's/->8000\/tcp//'

build: ## Build service images
	@echo "Building services..."
	docker compose -f $(COMPOSE_FILE) build order-accuracy frame-selector
	@echo "✓ Build complete"

clean: ## Remove all containers and images
	@echo "Cleaning up..."
	docker compose -f $(COMPOSE_FILE) down -v --rmi local
	@docker ps -a --filter "name=oa_" -q | xargs -r docker rm -f 2>/dev/null || true
	@echo "✓ Cleanup complete"

restart: stop start ## Restart all services

# Quick access commands
station-1: ## Start station 1 logs
	make logs STATION=1

station-2: ## Start station 2 logs
	make logs STATION=2

station-3: ## Start station 3 logs
	make logs STATION=3

station-4: ## Start station 4 logs
	make logs STATION=4

# Docker compose shortcuts
ps: ## Show running containers
	docker compose -f $(COMPOSE_FILE) ps

exec: ## Execute command in order-accuracy container (usage: make exec CMD="ls -la")
	docker exec -it oa_service $(CMD)

# Development helpers
dev-build: ## Build and restart single station
	make build
	make restart

test: ## Run a test video through the system
	@echo "Uploading test video..."
	@curl -X POST http://localhost:8000/upload-video \
		-F "file=@uploads/384-651-925.mp4" \
		-F "video_id=test_video"
	@echo ""
	@echo "✓ Video uploaded. Check Gradio UI at http://localhost:7860"
