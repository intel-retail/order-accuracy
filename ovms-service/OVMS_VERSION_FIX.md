# OVMS Model Structure Fix - Quick Guide

## Problem
OVMS error: `No version found for model in path: /models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8`

## Cause
The model directory is **missing the `graph.pbtxt` file** which is required for VLM/LLM models in OVMS MediaPipe mode. 

Without `graph.pbtxt`, OVMS falls back to standard model loading which requires versioned directories (model_name/1/). 

**With `graph.pbtxt`**: Model files can be directly in the model directory ✅  
**Without `graph.pbtxt`**: OVMS requires version subdirectory (model_name/1/) ❌

## Solution

### CORRECT Solution: Copy graph.pbtxt file
```bash
# On the working machine, copy graph.pbtxt
scp /home/intel/jsaini/order-accuracy/ovms-service/models_vlm/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/graph.pbtxt \
    user@other-machine:/home/intel/sainijit/order-accuracy/ovms-service/models_vlm/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/

# Or manually copy this content to graph.pbtxt:
```

**graph.pbtxt content:**
```protobuf
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }
  node_options: {
      [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
          pipeline_type: VLM_CB,
          models_path: "./",
          plugin_config: '{}',
          enable_prefix_caching: false,
          cache_size: 10,
          max_num_seqs: 8,
          device: "GPU",
      }
  }
  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler",
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_wait_time: -1
      }
    }
  }
}
```

Then restart OVMS:
```bash
cd /home/intel/sainijit/order-accuracy
docker compose --profile ovms restart ovms-vlm
```

### Alternative: Use Version Directory (Not Recommended for VLM)
```bash
cd /home/intel/sainijit/order-accuracy/ovms-service
./fix_model_structure.sh
```

### Option 2: Manual Fix
```bash
# Navigate to model directory
cd /home/intel/sainijit/order-accuracy/ovms-service/models_vlm/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8

# Create version directory
mkdir -p 1

# Move all model files to version directory
mv *.xml *.bin *.json *.txt *.jinja 1/

# Verify structure
ls -la 1/ | head -10
```

### Option 3: Fix Running Container
```bash
# Enter the container
docker exec -it oa_ovms_vlm /bin/sh

# Navigate to model directory
cd /models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8

# Create version directory
mkdir -p 1

# Move files
mv *.xml *.bin *.json *.txt *.jinja 1/

# Exit container
exit

# Restart OVMS
docker compose --profile ovms restart ovms-vlm
```

## Verification

### 1. Check Directory Structure
```bash
docker exec oa_ovms_vlm ls -la /models/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8/1/ | head -10
```

Expected output:
```
openvino_language_model.xml
openvino_language_model.bin
openvino_tokenizer.xml
openvino_tokenizer.bin
config.json
...
```

### 2. Check OVMS Logs
```bash
docker logs oa_ovms_vlm --tail 50 | grep -E "Loading|loaded|ERROR|ready|version 1"
```

Success indicators:
```
[modelmanager][info] Loading model: Qwen/Qwen2.5-VL-7B-Instruct-ov-int8, version: 1
[modelmanager][info] Model Qwen/Qwen2.5-VL-7B-Instruct-ov-int8, version: 1 loaded
```

### 3. Check Model Status
```bash
curl http://localhost:8001/v1/config | jq
```

Should show the model with version "1" listed.

## Important Notes

1. **Always use version directories** - OVMS will NOT load models without them
2. **Version must be numeric** - Use "1", "2", "3", etc. (not "v1" or "latest")
3. **All model files go inside** - XML, BIN, JSON, config files all go in version directory
4. **Config.json uses base_path** - No need to include version in config, OVMS finds it automatically:
   ```json
   {
     "name": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8",
     "base_path": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"  ← OVMS looks for /1/ subdirectory
   }
   ```

## After Fix

```bash
# Restart OVMS
cd /home/intel/sainijit/order-accuracy
docker compose --profile ovms restart ovms-vlm

# Wait for model to load (30-60 seconds)
sleep 30

# Verify no more warnings
docker logs oa_ovms_vlm --tail 20 | grep -i "version found"

# If you see "No version found", the fix didn't work - verify directory structure again
# If no output, the fix worked!
```
