# Dine-In Order Accuracy Benchmark - Docker Setup

This directory contains a Dockerized setup for the Dine-In Order Accuracy Benchmark application with both a Gradio UI and REST API, integrated with OVMS (OpenVINO Model Server).

## Architecture

The setup includes three main components:

1. **dine-in**: 
   - **Gradio UI** (Port 7861): Interactive web interface for order validation
   - **REST API** (Port 8083): FastAPI endpoints for programmatic access
2. **ovms-vlm**: OpenVINO Model Server for Vision-Language Model inference (Port 8002)
3. **semantic-service**: AI-powered semantic item matching service (Optional)

## API Endpoints

The application provides the following REST endpoints:

### Core Endpoints
- `POST /api/validate` - Validate single plate image against order
- `POST /api/validate/batch` - Validate multiple plates in one request  
- `GET /api/validate/{validation_id}` - Retrieve validation result by ID
- `GET /api/validate` - List all validation results
- `DELETE /api/validate/{validation_id}` - Delete validation result
- `GET /health` - Health check endpoint

### Documentation
- **Swagger UI**: http://localhost:8083/docs
- **API Documentation**: See [API.md](API.md) for detailed examples

## Quick Start

- Docker and Docker Compose installed
- GPU with Intel drivers (for OVMS GPU acceleration)
- Models directory set up at `../models/` (relative to this directory)
- Semantic comparison service image built

### Build Semantic Service (if not already built)

```bash
# Navigate to the semantic-comparison-service directory and build
cd ../semantic-comparison-service
docker build -t semantic-comparison-service:latest .
```

### Setup OVMS Models (if not already done)

```bash
# Navigate to the ovms-service directory and run setup
cd ../ovms-service
./setup_models.sh
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- GPU with Intel drivers (for OVMS GPU acceleration)
- Models directory set up at `../models/` (relative to this directory)

### 1. Build and Start All Services

```bash
docker compose up --build -d
```

### 2. Access the Application

**Gradio Web UI:**
```
http://localhost:7861
```

**REST API:**
```
http://localhost:8083
```

**API Documentation (Swagger UI):**
```
http://localhost:8083/docs
```

**OVMS Service:**
```
http://localhost:8002
```

### 3. Test the API

Run the included test script:
```bash
python test_api.py
```

Or use cURL:
```bash
# Health check
curl http://localhost:8083/health

# Validate a plate
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/image_01_mcd_combo.png" \
  -F 'order={"order_id":"order_123","items":[{"name":"Big Mac","quantity":1}]}'
```

### 4. Check Service Health

```bash
# Check all containers
docker compose ps

# View logs
docker compose logs -f dine-in
docker compose logs -f ovms-vlm
```

### 5. Stop Services

```bash
docker compose down
```

## Service Details

### Dine-In Application
- **Port**: 7860
- **Type**: Gradio web interface
- **Purpose**: Staff-triggered plate validation UI

### OVMS VLM
- **Port**: 8001 (mapped to internal 8000)
- **Type**: OpenVINO Model Server with GPU support
- **Purpose**: Vision-Language Model inference

### Semantic Service
- **Port**: 8080 (REST API), 9090 (Prometheus metrics)
- **Type**: FastAPI service
- **Purpose**: AI-powered semantic item matching

## Environment Variables

The following environment variables are configured in docker-compose.yml:

### Dine-In Service
- `SEMANTIC_SERVICE_ENDPOINT`: http://semantic-service:8080
- `OVMS_ENDPOINT`: http://ovms-vlm:8000

### OVMS Service
- `OV_CACHE_DIR`: /tmp/ov_cache (for model compilation caching)

### Semantic Service
- `VLM_BACKEND`: ovms
- `OVMS_ENDPOINT`: http://ovms-vlm:8000
- `CACHE_ENABLED`: true
- `LOG_LEVEL`: INFO

## Troubleshooting

### OVMS fails to start
- Ensure GPU drivers are properly installed
- Check that models are present in `../models/` directory
- Verify `../models/config.json` exists and is valid

### Semantic service is unhealthy
- Check OVMS is running and healthy first
- View logs: `docker compose logs semantic-service`
- Verify network connectivity between services

### Dine-In app can't connect to services
- Ensure all services are healthy: `docker compose ps`
- Check network configuration in docker-compose.yml
- Verify service dependencies are met

### Port conflicts
- Check if ports 7860, 8001, 8080, or 9090 are already in use
- Modify port mappings in docker-compose.yml if needed

## Development

### Rebuild Single Service

```bash
# Rebuild dine-in only
docker compose build dine-in
docker compose up -d dine-in

# View live logs
docker compose logs -f dine-in
```

### Update Application Code

After modifying app.py:
```bash
docker compose build dine-in
docker compose restart dine-in
```

## Files

- `Dockerfile`: Container definition for the dine-in application
- `docker-compose.yml`: Multi-container orchestration
- `requirements.txt`: Python dependencies
- `app.py`: Gradio application code
- `images/`: Plate images for validation
- `orders/orders.json`: Order manifests

## Network

All services communicate over a dedicated bridge network `dinein-net` for isolation and security.

## Volume Mounts

- `../models:/models:ro` - OVMS models (read-only)
- `../ovms-service/cache:/tmp/ov_cache` - OVMS compilation cache
- `../config:/app/config:ro` - Configuration files (read-only)
