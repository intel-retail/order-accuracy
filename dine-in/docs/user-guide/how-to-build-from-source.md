# How to Build from Source

Instructions for building the Dine-In Order Accuracy application from source code.

## Prerequisites

- Git
- Docker Engine 24.0+
- Docker Compose 2.20+
- Python 3.10+ (for local development)

## Clone Repository

```bash
git clone <repository-url>
cd order-accuracy/dine-in
```

## Project Structure

```
dine-in/
├── configs/
│   ├── orders.json       # Test order manifests
│   └── inventory.json    # Known menu items
├── images/               # Test plate images
├── src/
│   ├── app.py           # Gradio UI application
│   ├── api.py           # FastAPI endpoints
│   ├── config.py        # Configuration management
│   └── services/
│       ├── vlm_client.py         # VLM inference client
│       ├── semantic_client.py    # Semantic matching client
│       ├── validation_service.py # Validation logic
│       └── benchmark_service.py  # Benchmark orchestration
├── docs/                 # Documentation
├── results/              # Benchmark outputs
├── docker-compose.yml    # Container orchestration
├── Dockerfile           # Application image
├── Makefile             # Build automation
└── requirements.txt     # Python dependencies
```

## Build Docker Images

### Using Make

```bash
make build
```

### Using Docker Compose Directly

```bash
docker compose build
```

### Build with No Cache

```bash
docker compose build --no-cache
```

## Build Dependencies

The application builds from `requirements.txt`:

```
gradio>=4.0.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pillow>=10.0.0
pydantic>=2.0.0
httpx[http2]>=0.25.0
aiofiles>=23.0.0
psutil>=5.9.0
git+https://github.com/sumanaga/performance-tools.git@unique-id-support#subdirectory=benchmark-scripts
```

## Local Development (Without Docker)

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API Server

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8083 --reload
```

### Run Gradio UI

```bash
cd src
python app.py
```

## Environment Variables

Set these for local development:

```bash
export OVMS_ENDPOINT="http://localhost:8000"
export SEMANTIC_SERVICE_ENDPOINT="http://localhost:8080"
export API_TIMEOUT=300
export LOG_LEVEL=DEBUG
```

## Running Tests

### Syntax Check

```bash
python -m py_compile src/api.py
python -m py_compile src/services/vlm_client.py
```

### Type Checking (Optional)

```bash
pip install mypy
mypy src/ --ignore-missing-imports
```

## Building Semantic Service

The semantic service is a separate image:

```bash
cd ../semantic-comparison-service
docker build -t semantic-comparison-service:latest .
```

## Building Metrics Collector

```bash
cd ../metrics-collector
docker build -t metrics-collector:latest .
```

## Verify Build

```bash
# List built images
docker images | grep -E 'dine-in|semantic|metrics'

# Expected output:
# dine-in-dine-in              latest    ...
# semantic-comparison-service  latest    ...
# metrics-collector           latest    ...
```

## Troubleshooting Build Issues

### Git Dependency Fails

If the vlm_metrics_logger fails to install:

```bash
# Ensure git is installed
apt-get install git

# Or install without the git dependency for testing
pip install gradio fastapi uvicorn httpx pillow pydantic psutil
```

### HTTP/2 Missing

If you see "h2 package not installed":

```bash
# Ensure httpx[http2] is installed
pip install httpx[http2]
```

### Docker Build Timeout

Increase timeout or use cache:

```bash
# Use cached layers
docker compose build

# If cache is stale
docker compose build --no-cache
```
