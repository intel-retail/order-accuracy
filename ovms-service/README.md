# OVMS Service for Order Accuracy

This directory contains the OpenVINO Model Server (OVMS) configuration and model export scripts for the order accuracy VLM backend.

## Directory Structure

```
ovms-service/
├── export_model.py          # Script to export HuggingFace models to OpenVINO format
├── export_requirements.txt  # Python dependencies for model export
├── models_vlm/              # OVMS model repository
│   ├── config.json          # OVMS configuration
│   └── Qwen/                # Model directory (created after export)
│       └── Qwen2-VL-2B-Instruct/
└── README.md                # This file
```

## Model Setup

### Prerequisites

1. **Python environment** with model export dependencies:
   ```bash
   pip install -r export_requirements.txt
   ```

2. **Disk space**: ~5GB for Qwen2-VL-2B-Instruct model (int4 quantization)

### Export Model

The model needs to be exported once before running OVMS:

```bash
cd ovms-service

# Export Qwen2-VL-2B-Instruct with int4 quantization for GPU
python export_model.py text_generation \
  --source_model Qwen/Qwen2-VL-2B-Instruct \
  --weight-format int4 \
  --target_device GPU \
  --model_repository_path models_vlm
```

This will:
- Download the model from HuggingFace
- Convert to OpenVINO IR format
- Apply int4 quantization for reduced memory usage
- Save to `models_vlm/Qwen/Qwen2-VL-2B-Instruct/`

### Alternative: Copy Pre-exported Model

If you already have the model exported in another location:

```bash
# Copy from existing ovms-vlm installation
cp -r /home/intel/jsaini/ovms-vlm/models_vlm/Qwen ./models_vlm/
```

## Running OVMS

The OVMS service is integrated into the main docker-compose. To start it:

```bash
# From order-accuracy root directory
cd ..

# Start with OVMS backend
docker-compose --profile ovms up -d

# Check OVMS health
curl http://localhost:8001/v1/config

# Check model status
curl http://localhost:8001/v1/models
```

## Configuration

### OVMS Model Configuration

The `models_vlm/config.json` file configures the OVMS model server:

```json
{
    "model_config_list": [
        {
            "config": {
                "name": "Qwen/Qwen2-VL-2B-Instruct",
                "base_path": "Qwen/Qwen2-VL-2B-Instruct",
                "plugin_config": {
                    "NUM_STREAMS": "1",
                    "CACHE_DIR": ""
                }
            }
        }
    ],
    "monitoring": {
        "metrics": {
            "enable": true,
            "metrics_list": ["ovms_streams"]
        }
    }
}
```

### Docker Compose Integration

The OVMS service is defined in `../docker-compose.yaml`:

```yaml
ovms-vlm:
  image: openvino/model_server:latest-gpu
  profiles: ["ovms"]
  volumes:
    - ./ovms-service/models_vlm:/models:ro
  ports:
    - "8001:8001"
  environment:
    - LOG_LEVEL=INFO
  devices:
    - /dev/dri:/dev/dri
```

## Usage in Application

When `VLM_BACKEND=ovms` is set in application configuration:

1. **Application Service** connects to OVMS via HTTP
2. **Endpoint**: http://ovms-vlm:8000/v3/chat/completions
3. **Model**: Qwen/Qwen2-VL-2B-Instruct
4. **API**: OpenAI-compatible chat completions

See [../QUICK_START_BACKEND_SWITCH.md](../QUICK_START_BACKEND_SWITCH.md) for backend switching guide.

## Troubleshooting

### Model not found error
```bash
# Ensure model is exported
ls models_vlm/Qwen/Qwen2-VL-2B-Instruct/

# Check OVMS logs
docker-compose logs ovms-vlm
```

### Out of memory
```bash
# Use int4 quantization (default in export script)
# Reduce cache_size in config if needed
```

### Permission errors
```bash
# Ensure models directory is readable
chmod -R 755 models_vlm/
```

## Performance

- **Model Size**: ~2GB (int4 quantization)
- **Inference Device**: Intel Arc iGPU
- **Latency**: ~2-3s per image (1024x1024)
- **Memory**: ~4GB total OVMS footprint
- **Batching**: Supported for multiple requests

## References

- [OVMS Documentation](https://github.com/openvinotoolkit/model_server)
- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
