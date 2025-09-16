# Order Accuracy VLM App

A modular application for processing grocery checkout videos using object detection pipelines and Vision Language Models (VLM) to identify items and extract bill information.

## Installation

### Prerequisites
- Python 3.9+
- ffmpeg (for video processing)
- Access to external services (pipeline server, RabbitMQ, VLM service)

### Setup with uv

1. **Install uv** (if not present):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Restart shell so that uv is on PATH

   if not work try with : sudo snap install astral-uv --classic
   ```

2. **Install dependencies**:
   ```bash
   cd order-accuracy-vlm
   uv sync
   ```

3. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg
   
   # Or with uv for Python packages
   uv add moviepy  # Optional for enhanced video processing
   ```

## Setting Required Environment Variables

### Before running the application, you need to set several environment variables:

1. Registry Configuration: The application uses registry URL and tag to pull the required images.

```bash
export REGISTRY_URL=intel   
export TAG=1.2.0   
```

2. Required credentials for some services: Following variables MUST be set on your current shell before running the setup script:

 
```bash
#MinIO credentials (object storage)
export MINIO_ROOT_USER=<your-minio-username>
export MINIO_ROOT_PASSWORD=<your-minio-password>

#PostgreSQL credentials (database)
export POSTGRES_USER=<your-postgres-username>
export POSTGRES_PASSWORD=<your-postgres-password>

#RabbitMQ credentials (message broker)
export RABBITMQ_USER=<your-rabbitmq-username>
export RABBITMQ_PASSWORD=<your-rabbitmq-password>
```

3. Setting environment variables for customizing model selection:
```bash
export OD_MODEL_NAME="yolov8l-worldv2"
export VCLIP_MODEL="openai/clip-vit-base-patch32"
export ENABLED_WHISPER_MODELS=true
export VLM_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"  # or any other supported VLM model on CPU
export ENABLED_WHISPER_MODELS="tiny.en,small.en,medium.en" 
```
Go to https://huggingface.co/settings/tokens to get your token.
```bash
export GATED_MODEL=true
export HUGGINGFACE_TOKEN=
```

4. Add/Update no_proxy with HOST_IP of your machine 
```bash
no_proxy=x.x.x.x(HOST_IP)
```

## Run/Stop the Application(By default application is running on CPU device)

```bash

make update-submodules (This command only needs to be run the first time during setup.)

make build (To Build the application)

make run-demo (To Run the application)

make down (To Stop the application)
```

The web interface will be available at `http://<HOST_IP>:7860`

## To Run Application on GPU device

```bash

export ENABLE_VLM_GPU=true

```


## Configuration

All configuration is centralized in `config.py`.

## Architecture

The application is now organized into modular components for better maintainability and extensibility:

### Core Modules

- **`config.py`** - Configuration settings and logging setup
- **`video_utils.py`** - Video processing utilities (duration validation, upload)
- **`pipeline.py`** - Pipeline management (trigger, monitoring)
- **`messaging.py`** - RabbitMQ/MQTT messaging for metadata collection
- **`vlm.py`** - Vision Language Model integration
- **`interface.py`** - Gradio web interface components
- **`main.py`** - Application entry point and orchestration
- **`app.py`** - Backward compatibility wrapper

### Data Flow

1. **Video Upload** → User uploads video via Gradio interface
2. **Validation** → Check video duration and format
3. **Upload Service** → Send video to processing service
4. **Pipeline Trigger** → Start object detection pipeline
5. **Metadata Collection** → Collect frame metadata via RabbitMQ
6. **Frame Selection** → Select top frames with most detected objects
7. **VLM Analysis** → Analyze selected frames to identify grocery items
8. **Results Display** → Show detected items and bill information



## Usage

1. **Upload Video**: Select a grocery checkout video
2. **Run Agent**: Click "Run Agent" to start processing
3. **Monitor Progress**: Watch status updates in real-time
4. **View Results**: See detected items and bill information in JSON format

## Module Details

### Configuration (`config.py`)
- API endpoints and URLs
- RabbitMQ connection settings  
- Model configurations
- Application settings
- Logging setup

### Video Processing (`video_utils.py`)
- Video duration extraction (moviepy/ffprobe)
- Video validation against size limits
- Video upload to processing service

### Pipeline Management (`pipeline.py`)
- Pipeline trigger with video parameters
- Pipeline status monitoring and polling
- Error handling and timeout management

### Messaging (`messaging.py`)
- RabbitMQ connection management
- MQTT topic subscription
- Frame metadata collection and parsing
- Frame ranking by object count

### VLM Integration (`vlm.py`)
- Payload construction for VLM API
- Frame URL generation
- Response parsing and JSON extraction
- Error handling for VLM calls

### User Interface (`interface.py`)
- Gradio web interface setup
- Video upload handling
- Progress tracking and status updates
- Results display

## External Dependencies

The application requires several external services (using VSS_IP environment variable, defaults to 10.223.24.242):

- **Video Upload Service**: `http://${VSS_IP}:12345/manager/videos`
- **Pipeline Service**: `http://${VSS_IP}:8090/pipelines/...`
- **Frame Storage**: `http://${VSS_IP}:12345/datastore/video-summary/...`
- **VLM Service**: `http://${VSS_IP}:9766/v1/chat/completions`
- **RabbitMQ**: `${VSS_IP}` (MQTT messaging)

## Troubleshooting

### Video Processing Issues

### RabbitMQ Connection Issues
- **Connection refused**: Check RabbitMQ host and credentials
- **No messages received**: Verify topic configuration and publisher status
- **Authentication failed**: Check RABBITMQ_USER and RABBITMQ_PASSWORD

### Pipeline Issues
- **Pipeline trigger fails**: Check pipeline service availability
- **Pipeline timeout**: Check pipeline service logs and increase timeout
- **Invalid video format**: Ensure video is in supported format

### VLM Issues
- **API timeout**: Check VLM service availability and increase timeout
- **Invalid response**: Check VLM service health and model status
- **JSON parsing fails**: Review VLM prompt and response format

## Development

### Adding New Features

1. **New video processing**: Extend `video_utils.py`
2. **New pipeline types**: Extend `pipeline.py`
3. **New messaging formats**: Extend `messaging.py`
4. **New VLM models**: Update `vlm.py` and `config.py`
5. **UI improvements**: Modify `interface.py`

### Testing

Run tests for individual modules:
```bash
uv run python -m pytest tests/  # When tests are added
```

### Logging

Logs are written to:
- Console output (INFO level)
- `order_accuracy_app.log` file
- Structured format with timestamps

## License

This project is developed for retail automation purposes.
