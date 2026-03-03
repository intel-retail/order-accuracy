#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${SCRIPT_DIR}/models"

###############################################
# HARD CODED MODEL REGISTRY
###############################################
declare -A MODEL_SOURCES
MODEL_SOURCES["Qwen2.5-VL-7B-Instruct-ov-int8"]="Qwen/Qwen2.5-VL-7B-Instruct"

POTENTIAL_SOURCE_DIRS=(
    "${HOME}/ovms-vlm/models"
    "/opt/ovms/models"
    "${PROJECT_ROOT}/../ovms-vlm/models"
)

###############################################
echo "=========================================="
echo "OVMS Model Setup for Order Accuracy"
echo "=========================================="
echo ""

###############################################
check_model() {
    local model_path="$1"

    echo "Debug: Checking model at ${model_path}"
    
    if [ ! -d "${model_path}" ]; then
        echo "  Directory not found"
        # Check if model was created with source name instead
        local source_path="${MODELS_DIR}/Qwen/Qwen2.5-VL-7B-Instruct"
        if [ -d "${source_path}" ]; then
            echo "  Found model at source path: ${source_path}"
            echo "  Moving to expected path..."
            mv "${source_path}" "${model_path}"
        else
            return 1
        fi
    fi
    
    echo "  Contents of ${model_path}:"
    ls -la "${model_path}" 2>/dev/null || echo "  (empty or inaccessible)"

    if [ -f "${model_path}/graph.pbtxt" ] &&
       [ -f "${model_path}/openvino_language_model.xml" ] &&
       [ -f "${model_path}/openvino_language_model.bin" ]; then
        echo "  ✓ All required files found"
        return 0
    else
        echo "  ✗ Missing required files:"
        [ ! -f "${model_path}/graph.pbtxt" ] && echo "    - graph.pbtxt"
        [ ! -f "${model_path}/openvino_language_model.xml" ] && echo "    - openvino_language_model.xml" 
        [ ! -f "${model_path}/openvino_language_model.bin" ] && echo "    - openvino_language_model.bin"
        return 1
    fi
}

###############################################
ask_user_model() {
    local model_name="$1"
    read -p "Do you want to setup ${model_name}? (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

###############################################
setup_python_env() {

    if [ ! -f "${SCRIPT_DIR}/export_model.py" ]; then
        echo "[1/3] Downloading OVMS export tools..."

        EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models"

        curl -fsSL "${EXPORT_BASE_URL}/export_model.py" -o "${SCRIPT_DIR}/export_model.py"
        curl -fsSL "${EXPORT_BASE_URL}/requirements.txt" -o "${SCRIPT_DIR}/export_requirements.txt"
        echo "  ✓ Export tools downloaded"
    else
        echo "[1/3] Export tools already present"
    fi

    if [ ! -d "${SCRIPT_DIR}/venv" ] || [ ! -f "${SCRIPT_DIR}/venv/bin/pip" ]; then
        echo "[2/3] Creating Python virtual environment..."
        python3 -m venv "${SCRIPT_DIR}/venv" --clear
        echo "  ✓ Virtual environment created"
    else
        echo "[2/3] Virtual environment already exists"
    fi

    source "${SCRIPT_DIR}/venv/bin/activate"

    echo "[3/3] Installing Python dependencies (this may take a minute)..."
    pip install -q --upgrade pip
    pip install -q -r "${SCRIPT_DIR}/export_requirements.txt"
    echo "  ✓ Dependencies installed"
}

###############################################
export_model() {

    local MODEL_NAME="$1"
    local SOURCE_MODEL="$2"

    echo ""
    echo "Exporting ${MODEL_NAME}"
    echo ""

    python "${SCRIPT_DIR}/export_model.py" text_generation \
      --source_model "${SOURCE_MODEL}" \
      --weight-format int8 \
      --pipeline_type VLM_CB \
      --target_device GPU \
      --cache_size 32 \
      --max_num_seqs 1 \
      --enable_prefix_caching \
      --config_file_path "${MODELS_DIR}/config.json" \
      --model_repository_path "${MODELS_DIR}" \
      --model_name "${MODEL_NAME}"
}

###############################################
mkdir -p "${MODELS_DIR}"

echo "Setting up Python environment..."
echo ""
setup_python_env
echo ""
echo "✓ Python environment ready"

###############################################
# MAIN MODEL LOOP
###############################################

for MODEL_NAME in "${!MODEL_SOURCES[@]}"; do

    SOURCE_MODEL="${MODEL_SOURCES[$MODEL_NAME]}"
    TARGET_PATH="${MODELS_DIR}/Qwen/${MODEL_NAME}"

    echo ""
    echo "------------------------------------------"
    echo "Model: ${MODEL_NAME}"
    echo "------------------------------------------"

    if ! ask_user_model "${MODEL_NAME}"; then
        echo "Skipped ${MODEL_NAME}"
        continue
    fi

    ###########################################
    # Check if already exists locally
    ###########################################
    if check_model "${TARGET_PATH}"; then
        echo "✓ Model already exists locally"
        continue
    fi

    ###########################################
    # Check copy from external sources
    ###########################################
    for SOURCE_DIR in "${POTENTIAL_SOURCE_DIRS[@]}"; do
        if [ -d "${SOURCE_DIR}/Qwen/${MODEL_NAME}" ]; then

            if check_model "${SOURCE_DIR}/Qwen/${MODEL_NAME}"; then
                echo "Copying model from ${SOURCE_DIR}"

                mkdir -p "${MODELS_DIR}/Qwen"
                cp -r "${SOURCE_DIR}/Qwen/${MODEL_NAME}" "${MODELS_DIR}/Qwen/"

                echo "✓ Copied ${MODEL_NAME}"
                continue 2
            fi
        fi
    done

    ###########################################
    # Ask before downloading/export
    ###########################################
    echo ""
    echo "Model not found locally."
    echo "Will download and export from HuggingFace."
    echo ""

    read -p "Continue export for ${MODEL_NAME}? (y/N): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipped export for ${MODEL_NAME}"
        continue
    fi

    export_model "${MODEL_NAME}" "${SOURCE_MODEL}"

    ###########################################
    # Verify
    ###########################################
    if check_model "${TARGET_PATH}"; then
        echo "✓ Export successful for ${MODEL_NAME}"
    else
        echo "✗ Export failed for ${MODEL_NAME}"
        exit 1
    fi

done

###############################################
# APPLY graph_options.json TO graph.pbtxt
###############################################
# This allows testers to tune OVMS parameters by editing
# graph_options.json instead of manually editing graph.pbtxt.
###############################################

apply_graph_config() {
    local GRAPH_OPTIONS_FILE="${MODELS_DIR}/graph_options.json"
    local MODEL_NAME="$1"
    local GRAPH_FILE="${MODELS_DIR}/Qwen/${MODEL_NAME}/graph.pbtxt"

    if [ ! -f "${GRAPH_OPTIONS_FILE}" ]; then
        echo "  No graph_options.json found, keeping existing graph.pbtxt"
        return 0
    fi

    echo ""
    echo "------------------------------------------"
    echo "Applying graph_options.json to graph.pbtxt"
    echo "------------------------------------------"

    python3 - "${GRAPH_OPTIONS_FILE}" "${GRAPH_FILE}" << 'PYEOF'
import json, sys

graph_options_file = sys.argv[1]
graph_file = sys.argv[2]

opts = json.load(open(graph_options_file))

# Build plugin_config string with escaped quotes for protobuf
plugin_cfg = opts.get('plugin_config', {})
plugin_str = json.dumps(plugin_cfg).replace('"', '\\"') if plugin_cfg else '{}'

# Boolean fields (protobuf uses lowercase true/false)
prefix_caching = 'true' if opts.get('enable_prefix_caching', False) else 'false'
dynamic_split = 'true' if opts.get('dynamic_split_fuse', False) else 'false'

max_num_seqs = opts.get('max_num_seqs', 4)
cache_size = opts.get('cache_size', 10)
max_num_batched_tokens = opts.get('max_num_batched_tokens', 4096)
device = opts.get('device', 'GPU')

graph = f'''input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node {{
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"

  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"

  input_side_packet: "LLM_NODE_RESOURCES:llm"

  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"

  input_stream_info {{
    tag_index: "LOOPBACK:0"
    back_edge: true
  }}

  node_options {{
    [type.googleapis.com/mediapipe.LLMCalculatorOptions] {{

      pipeline_type: VLM_CB
      models_path: "./"

      plugin_config: "{plugin_str}"

      enable_prefix_caching: {prefix_caching}
      cache_size: {cache_size}

      max_num_seqs: {max_num_seqs}
      dynamic_split_fuse: {dynamic_split}
      max_num_batched_tokens: {max_num_batched_tokens}

      device: "{device}"
    }}
  }}
  input_stream_handler {{
    input_stream_handler: "SyncSetInputStreamHandler"

    options {{
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {{
        sync_set {{
          tag_index: "LOOPBACK:0"
        }}
      }}
    }}
  }}
}}
'''

with open(graph_file, 'w') as f:
    f.write(graph)

print('  Values applied:')
print(f'    max_num_seqs:          {max_num_seqs}')
print(f'    cache_size:            {cache_size}')
print(f'    enable_prefix_caching: {prefix_caching}')
print(f'    dynamic_split_fuse:    {dynamic_split}')
print(f'    max_num_batched_tokens:{max_num_batched_tokens}')
print(f'    device:                {device}')
PYEOF

    if [ $? -eq 0 ]; then
        echo "  ✓ graph.pbtxt updated from graph_options.json"
    else
        echo "  ✗ Failed to apply graph_options"
        return 1
    fi
}

# Apply graph_options for each model
for MODEL_NAME in "${!MODEL_SOURCES[@]}"; do
    TARGET_PATH="${MODELS_DIR}/Qwen/${MODEL_NAME}"
    if [ -d "${TARGET_PATH}" ]; then
        apply_graph_config "${MODEL_NAME}"
    fi
done

###############################################
echo ""
echo "=========================================="
echo "✓ All Model Setup Complete!"
echo "=========================================="

###############################################
# EASYOCR MODEL DOWNLOAD
# EasyOCR models are used by the frame_pipeline for OCR-based order detection.
# Pre-downloading avoids a 60-90s delay on first container start.
###############################################
TAKEAWAY_DIR="$(dirname "${SCRIPT_DIR}")/take-away"
EASYOCR_DIR="${TAKEAWAY_DIR}/models/easyocr"

echo ""
echo "=========================================="
echo "EasyOCR Model Setup"
echo "=========================================="
echo "Target: ${EASYOCR_DIR}"
echo ""

if [ -f "${EASYOCR_DIR}/craft_mlt_25k.pth" ] && [ -f "${EASYOCR_DIR}/english_g2.pth" ]; then
    echo "✓ EasyOCR models already present, skipping download."
else
    read -p "Download EasyOCR models (~200MB)? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "${EASYOCR_DIR}"
        echo "Downloading EasyOCR models to ${EASYOCR_DIR} ..."
        echo "  (detection model ~90MB + recognition model ~100MB)"
        EASYOCR_DIR="${EASYOCR_DIR}" python3 -c "
import sys, os
easyocr_dir = os.environ['EASYOCR_DIR']
try:
    import easyocr
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'easyocr'])
    import easyocr
easyocr.Reader(['en'], gpu=False, verbose=True, model_storage_directory=easyocr_dir)
print('Done.')
"
        if [ -f "${EASYOCR_DIR}/craft_mlt_25k.pth" ] && [ -f "${EASYOCR_DIR}/english_g2.pth" ]; then
            echo "✓ EasyOCR models downloaded to ${EASYOCR_DIR}"
        else
            echo "✗ EasyOCR download may have failed — check ${EASYOCR_DIR}"
        fi
    else
        echo "Skipped EasyOCR download."
    fi
fi
