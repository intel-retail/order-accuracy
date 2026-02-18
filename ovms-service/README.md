# OVMS Service for Order Accuracy

This directory contains the OpenVINO Model Server (OVMS) configuration and model export scripts for the order accuracy VLM backend.

## Directory Structure

```
ovms-service/
├── export_model.py          # Script to export HuggingFace models to OpenVINO format
├── export_requirements.txt  # Python dependencies for model export
├── models/                  # OVMS model repository
│   ├── config.json          # OVMS configuration
│   └── Qwen/                # Model directory (created after export)
│       └── Qwen2.5-VL-7B-Instruct-ov-int8/
│           ├── graph.pbtxt  # MediaPipe graph configuration (critical)
│           └── openvino_*   # OpenVINO model files
└── README.md                # This file
```

## Model Setup

### Prerequisites

1. **Python environment** with model export dependencies:
   ```bash
   pip install -r export_requirements.txt
   ```

2. **Disk space**: ~8GB for Qwen2.5-VL-7B-Instruct-ov-int8 model (int8 quantization)

### Export Model

The model needs to be exported once before running OVMS:

```bash
cd ovms-service

# Export Qwen2.5-VL-7B-Instruct with int8 quantization for GPU
python export_model.py text_generation \
  --source_model Qwen/Qwen2.5-VL-7B-Instruct \
  --weight-format int8 \
  --target_device GPU \
  --model_repository_path models \
  --cache_size 32 \
  --max_num_seqs 1 \
  --enable_prefix_caching
```

This will:
- Download the model from HuggingFace
- Convert to OpenVINO IR format
- Apply int8 quantization for optimal quality/performance balance
- Save to `models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/`
- Generate graph.pbtxt for MediaPipe configuration

### Alternative: Use Setup Script

Use the automated setup script that handles export or copying automatically:

```bash
cd ovms-service
./setup_models.sh
```

This script will:
- Check if models already exist in `models/` directory
- Offer to copy from existing installations if found
- Otherwise, automatically export from HuggingFace

## Running OVMS

The OVMS service is integrated into the main docker-compose. To start it:

```bash
# From order-accuracy root directory
cd ..

# Start with OVMS backend
docker-compose --profile ovms up -d

# Check OVMS health
curl http://localhost:8002/v1/config

# Check model status  
curl http://localhost:8002/v1/models

# Verify model is AVAILABLE
curl http://localhost:8002/v1/config | jq '."Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"'
```

## Configuration

### OVMS Model Configuration

The `models/config.json` file configures the OVMS model server:

```json
{
    "model_config_list": [
        {
            "config": {
                "name": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8",
                "base_path": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"
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

### MediaPipe Graph Configuration

The `models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/graph.pbtxt` file configures the MediaPipe execution graph. **This file is critical** and must be in protobuf text format (not JSON).

Key parameters for optimal performance:

```protobuf
node: {
  calculator: "ModelAPISessionCalculator"
  output_side_packet: "SESSION:session"
  node_options: {
    [type.googleapis.com / mediapipe.ModelAPISidePacketCalculatorOptions]: {
      servable_name: "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"
      servable_version: "0"
      base_path: "/models"
    }
  }
  node_options: {
    [type.googleapis.com /mediapipe.LLMNodeOptions]: {
      max_num_seqs: 1              # Single request processing
      cache_size: 32               # 32GB KV cache for inventory prompts
      block_size: 32
      max_num_batched_tokens: 256
      enable_prefix_caching: true  # Cache repeated inventory lists
      dynamic_split_fuse: true
      plugin_config: {
        key: "NUM_STREAMS"
        value: "1"                # Dedicated GPU resources
      }
      plugin_config: {
        key: "CACHE_DIR"
        value: "/models/cache"
      }
    }
  }
}
```

### Docker Compose Integration

The OVMS service is defined in `../docker-compose.yaml`:

```yaml
ovms-vlm:
  image: openvino/model_server:2025.4.1-gpu
  container_name: dinein_ovms_vlm
  volumes:
    - ../ovms-service/models:/models:ro
  ports:
    - "8002:8000"  # External:Internal
  environment:
    - LOG_LEVEL=INFO
  devices:
    - /dev/dri:/dev/dri
  command: >
    --config_path /models/config.json
    --port 8000
```

## Usage in Application

When `VLM_BACKEND=ovms` is set in application configuration:

1. **Application Service** connects to OVMS via HTTP
2. **Endpoint**: http://ovms-vlm:8000/v3/chat/completions (internal)
3. **External Port**: http://localhost:8002 (from host)
4. **Model**: Qwen/Qwen2.5-VL-7B-Instruct-ov-int8
5. **API**: OpenAI-compatible chat completions

See [../QUICK_START_BACKEND_SWITCH.md](../QUICK_START_BACKEND_SWITCH.md) for backend switching guide.

## Troubleshooting

### Model not found error
```bash
# Ensure model is exported
ls models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/

# Check for required files
ls models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/graph.pbtxt
ls models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/openvino_*.{xml,bin}

# Check OVMS logs
docker logs dinein_ovms_vlm
```

### Out of memory
```bash
# Model uses int8 quantization (~7.8GB)
# Reduce cache_size in graph.pbtxt if needed (default: 32GB)
# Ensure sufficient system memory (16GB+ recommended)
```

### Permission errors
```bash
# Ensure models directory is readable
chmod -R 755 models/
```

### OVMS parsing errors
```bash
# If you see "Error parsing text-format mediapipe.CalculatorGraphConfig"
# The graph.pbtxt MUST be in protobuf text format, NOT JSON

# Verify graph.pbtxt format
head -5 models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/graph.pbtxt
# Should show: input_stream: "..." (NOT {"input_stream": ...})

# Check OVMS model status
curl http://localhost:8002/v1/config | jq
# Should show "state": "AVAILABLE"
```

## Performance

- **Model Size**: ~7.8GB (int8 quantization)
- **Inference Device**: Intel Meteor Lake iGPU
- **Latency**: ~5-15s per image with inventory-aware prompts (depends on prompt length)
- **Memory**: ~8-12GB total OVMS footprint (model + KV cache)
- **Configuration**: Optimized for single-request processing (max_num_seqs=1)
- **Cache**: 32GB KV cache with prefix caching enabled for repeated inventory lists

## References

- [OVMS Documentation](https://github.com/openvinotoolkit/model_server)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
