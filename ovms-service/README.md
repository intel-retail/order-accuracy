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

1. **Disk space**: ~8 GB for Qwen2.5-VL-7B-Instruct-ov-int8 model (int8 quantization)

### Export Model

The model is exported by running `setup_models.sh` from the **repo root** — this is the only supported export path. The script downloads `export_model.py` and its dependencies on demand, so no manual `pip install` step is needed:

```bash
# From repo root — run for take-away (default) or dine-in
bash ovms-service/setup_models.sh --app take-away
# or
bash ovms-service/setup_models.sh --app dine-in
```

This will:
- Download `export_model.py` and install its dependencies automatically
- Download the model from HuggingFace
- Convert to OpenVINO IR format with int8 quantization
- Save to `ovms-service/models/Qwen/Qwen2.5-VL-7B-Instruct/`
- Generate `graph.pbtxt` for OVMS configuration

> **ℹ Low-RAM systems:** Set `export CACHE_SIZE=2` before running `setup_models.sh` if you are on a 16 GB system. For first-time export, a 48–64 GB host is recommended to avoid OOM. See [Tuning the KV Cache Size](#tuning-the-kv-cache-size).

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
      cache_size: 4                # 4 GB KV cache (adequate for max_num_seqs=1; raise via CACHE_SIZE env var for higher concurrency)
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

See [Tuning the KV Cache Size](#tuning-the-kv-cache-size) for the full sizing guide.

```bash
# Quick fix: lower CACHE_SIZE before re-running setup_models.sh (run from repo root)
export CACHE_SIZE=2
bash ovms-service/setup_models.sh --app take-away

# Or edit graph.pbtxt directly (no re-export needed, run from repo root)
# Find: cache_size: <N>
# Change to a value that keeps model (~8 GB) + cache_size ≤ available VRAM
sed -i 's/cache_size: [0-9]*/cache_size: 2/' \
    ovms-service/models/Qwen/Qwen2.5-VL-7B-Instruct/graph.pbtxt
docker restart oa_ovms_vlm
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

- **Model Size**: ~7.8 GB (int8 quantization)
- **Inference Device**: Intel Arc GPU (GPU) or CPU fallback
- **Latency**: ~5–15 s per image on GPU; ~60–120 s on CPU
- **Memory**: model (~8 GB VRAM) + KV cache (default 4 GB) ≈ 12 GB VRAM total
- **Configuration**: Optimized for single-station use (max_num_seqs=4, prefix caching enabled)

---

## Tuning the KV Cache Size

The `cache_size` parameter in `graph.pbtxt` controls how much memory OVMS pre-allocates for the KV (key-value attention) cache. Choosing the right value depends on your GPU VRAM and system RAM.

### How memory is used

| Component | Where | Approximate size |
|---|---|---|
| INT8 model weights | GPU VRAM (or system RAM on CPU) | ~8 GB |
| KV cache (`cache_size`) | GPU VRAM (discrete GPU) | configurable |
| KV cache (`cache_size`) | **System RAM** (integrated iGPU, WCL/MTL) | configurable |
| OVMS process overhead | System RAM | ~1–2 GB |

> **⚠ Integrated GPU warning (Wildcat Lake / Meteor Lake):** On platforms with an integrated Intel GPU (iGPU), the KV cache is allocated from **system RAM** — not dedicated VRAM. A `cache_size=32` on a 32 GB system will consume all available RAM and cause OVMS to crash. Always use a small `cache_size` on iGPU platforms.

### Recommended values by platform

| Platform | VRAM | Recommended `cache_size` | Total VRAM used | Notes |
|---|---|---|---|---|
| Intel Arc A770 16 GB | 16 GB | **4–6 GB** | ~12–14 GB | Default; leaves headroom for OS |
| Intel Arc A770 8 GB | 8 GB | **0** (dynamic) | ~8 GB + dynamic | Model alone fills VRAM; use dynamic |
| Intel Arc A380 6 GB | 6 GB | **0** (dynamic) | ~6 GB | Run model on CPU instead |
| Intel iGPU / WCL / MTL (32 GB system RAM) | shared | **2–4 GB** | uses system RAM | Keep total ≤ 24 GB system RAM |
| Intel iGPU / WCL / MTL (16 GB system RAM) | shared | **1–2 GB** | uses system RAM | Keep total ≤ 12 GB system RAM |
| CPU only | N/A | **2–4 GB** | system RAM | Slower inference; cache from RAM |

> `cache_size: 0` enables **dynamic allocation** — OVMS grows the cache as needed up to available memory. This avoids OOM at startup but may consume all available VRAM/RAM under load. Use for unknown or constrained hardware.

### How to change `cache_size`

**Option A — Before export** (recommended, bakes the value into `graph.pbtxt`):
```bash
# Set CACHE_SIZE env var before running setup_models.sh (run from repo root)
export CACHE_SIZE=2          # e.g. 2 GB for a 16 GB iGPU system
bash ovms-service/setup_models.sh --app take-away
```

**Option B — After export** (no re-export needed, edit `graph.pbtxt` directly):
```bash
# Run from repo root
# Edit graph.pbtxt
sed -i 's/cache_size: [0-9]*/cache_size: 2/' \
    ovms-service/models/Qwen/Qwen2.5-VL-7B-Instruct/graph.pbtxt

# Verify the change
grep cache_size ovms-service/models/Qwen/Qwen2.5-VL-7B-Instruct/graph.pbtxt

# Restart OVMS to pick up the new value
docker restart oa_ovms_vlm

# Confirm the model reloads as AVAILABLE
curl -sf http://localhost:8002/v1/config | grep -o '"state":"[^"]*"'
```

**Option C — Persistent via `.env`** (survives re-runs of `setup_models.sh`):
```bash
# Add CACHE_SIZE to your app .env file
echo "CACHE_SIZE=2" >> take-away/.env

# Re-run setup to apply
bash ovms-service/setup_models.sh --app take-away
```

### How to pick the right value

Run this helper to get a recommendation for your system:
```bash
python3 - << 'EOF'
import subprocess, re

# Total system RAM
with open('/proc/meminfo') as f:
    mem = int(re.search(r'MemTotal:\s+(\d+)', f.read()).group(1)) // 1024 // 1024

model_gb = 8   # INT8 Qwen2.5-VL-7B
overhead_gb = 2

print(f"System RAM  : {mem} GB")
print()

# Detect discrete vs integrated GPU via lspci memory (more reliable than /dev/dri)
has_discrete = False
try:
    lspci_out = subprocess.check_output(['lspci', '-v'], text=True, stderr=subprocess.DEVNULL)
    # Intel Arc dGPUs report large BAR memory regions (>= 8 GB); iGPUs do not
    for line in lspci_out.splitlines():
        if 'Arc' in line or 'Display' in line or 'VGA' in line:
            import re as _re
            bars = _re.findall(r'Memory.*\[size=(\d+)([MG])\]', lspci_out)
            for size, unit in bars:
                gb = int(size) if unit == 'G' else int(size) // 1024
                if gb >= 8:
                    has_discrete = True
except Exception:
    pass

if has_discrete:
    # Assume Arc A770 16 GB as most common discrete target
    available_vram = 16 - model_gb - overhead_gb
    rec = max(1, min(available_vram, 6))
    print(f"Recommended cache_size (discrete GPU, ~16 GB VRAM): {rec} GB")
else:
    # iGPU or CPU: KV cache comes from system RAM
    budget = mem - model_gb - overhead_gb - 4   # leave 4 GB for OS
    rec = max(1, min(budget // 4, 4))
    print(f"Recommended cache_size (iGPU/CPU, {mem} GB RAM): {rec} GB")
    if mem < 16:
        print("  ⚠ Very low RAM — consider cache_size=1 or cache_size=0 (dynamic)")
EOF
```

### Effect on performance

| `cache_size` | Behaviour |
|---|---|
| Too small (< 1 GB) | Long prompts or concurrent requests get terminated early |
| **4 GB (default)** | **Handles up to 4 simultaneous requests with ~4 K token context** |
| 8+ GB | Better for long menu/inventory prompts or higher concurrency |
| 0 (dynamic) | Grows as needed; safest on unknown hardware; may consume all RAM under load |

Monitor actual cache usage in the OVMS logs:
```bash
docker logs oa_ovms_vlm 2>&1 | grep "Cache usage"
# Example: Cache usage 23.9%  →  cache_size is well-sized
# Example: Cache usage 95%+   →  increase cache_size or reduce max_num_seqs
```

## References

- [OVMS Documentation](https://github.com/openvinotoolkit/model_server)
- [Qwen2.5-VL Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
