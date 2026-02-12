#!/bin/bash

#
# OVMS Model Setup Script for Order Accuracy
# This script sets up the OVMS model files needed for the VLM backend
# It will automatically export models if not found
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
SOURCE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

# Optional: Check common locations for pre-exported models
POTENTIAL_SOURCE_DIRS=(
    "${HOME}/ovms-vlm/models"
    "/opt/ovms/models"
    "${PROJECT_ROOT}/../ovms-vlm/models"
)

echo "=========================================="
echo "OVMS Model Setup for Order Accuracy"
echo "=========================================="
echo ""

# Function to check if model is properly set up (including graph.pbtxt)
check_model() {
    local model_path="$1"
    if [ -f "${model_path}/graph.pbtxt" ] && \
       [ -f "${model_path}/openvino_language_model.xml" ] && \
       [ -f "${model_path}/openvino_language_model.bin" ]; then
        return 0
    else
        return 1
    fi
}

# Check if models already exist in order-accuracy
if check_model "${MODELS_DIR}/Qwen/${MODEL_NAME}"; then
    echo "✓ Model already exists and is properly configured"
    echo "  Location: ${MODELS_DIR}/Qwen/${MODEL_NAME}"
    echo "  Size: $(du -sh ${MODELS_DIR}/Qwen/${MODEL_NAME} | cut -f1)"
    echo ""
    echo "✓ Setup complete! You can now start OVMS."
    exit 0
fi

# Check if we can copy from existing installations
for SOURCE_DIR in "${POTENTIAL_SOURCE_DIRS[@]}"; do
    if [ -d "${SOURCE_DIR}/Qwen/${MODEL_NAME}" ]; then
        echo "✓ Found existing model at ${SOURCE_DIR}"
        echo ""
        
        # Check if model is properly configured
        if check_model "${SOURCE_DIR}/Qwen/${MODEL_NAME}"; then
            echo "  → Copying ${MODEL_NAME} model..."
            mkdir -p "${MODELS_DIR}/Qwen"
            cp -r "${SOURCE_DIR}/Qwen/${MODEL_NAME}" "${MODELS_DIR}/Qwen/"
            echo "  ✓ Model copied successfully"
            
            echo ""
            echo "✓ Model setup complete!"
            echo ""
            echo "Model location: ${MODELS_DIR}/Qwen/${MODEL_NAME}"
            echo "Model size: $(du -sh ${MODELS_DIR}/Qwen/${MODEL_NAME} | cut -f1)"
            echo ""
            exit 0
        fi
    fi
done

echo "No pre-exported models found in standard locations."
echo "Searched: ${POTENTIAL_SOURCE_DIRS[@]}"

echo ""
echo "Will export from HuggingFace..."
echo ""
echo "This will:"
echo "  1. Download ${SOURCE_MODEL} (~7GB)"
echo "  2. Convert to OpenVINO format with INT8 quantization"
echo "  3. Optimize for single-request processing (max_num_seqs=1)"
echo "  4. Configure 32GB KV cache with prefix caching"
echo "  5. Create graph.pbtxt for OVMS MediaPipe mode"
echo ""
read -p "Continue with automatic export? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled. You can manually export with:"
    echo "  cd ${SCRIPT_DIR}"
    echo "  python export_model.py text_generation \\"
    echo "    --source_model ${SOURCE_MODEL} \\"
    echo "    --weight-format int8 \\"
    echo "    --pipeline_type VLM_CB \\"
    echo "    --target_device GPU \\"
    echo "    --cache_size 32 \\"
    echo "    --max_num_seqs 1 \\"
    echo "    --enable_prefix_caching \\"
    echo "    --config_file_path ${MODELS_DIR}/config.json \\"
    echo "    --model_repository_path ${MODELS_DIR}"
    exit 1
fi

# Setup Python environment
echo "Setting up Python environment..."

# Ensure we have the export script and requirements
if [ ! -f "${SCRIPT_DIR}/export_model.py" ] || [ ! -s "${SCRIPT_DIR}/export_model.py" ]; then
    echo "  → Downloading OVMS export tools from GitHub..."
    
    EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models"
    
    curl -fsSL "${EXPORT_BASE_URL}/export_model.py" -o "${SCRIPT_DIR}/export_model.py"
    curl -fsSL "${EXPORT_BASE_URL}/requirements.txt" -o "${SCRIPT_DIR}/export_requirements.txt"
    
    if [ -f "${SCRIPT_DIR}/export_model.py" ] && [ -s "${SCRIPT_DIR}/export_model.py" ]; then
        echo "  ✓ Export tools downloaded from OpenVINO Model Server (release 2025.4)"
    else
        echo "  ✗ Failed to download export tools"
        echo ""
        echo "Please download manually:"
        echo "  curl -o export_model.py ${EXPORT_BASE_URL}/export_model.py"
        echo "  curl -o export_requirements.txt ${EXPORT_BASE_URL}/requirements.txt"
        echo ""
        exit 1
    fi
fi

# Check if we can create a venv
if ! python3 -m venv --help &>/dev/null; then
    echo "  ⚠ Python venv module not available"
    echo ""
    echo "Installing python3-venv package..."
    echo "You may be prompted for sudo password."
    echo ""
    
    # Try to install python3-venv
    if command -v apt &>/dev/null; then
        sudo apt update && sudo apt install -y python3-venv
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3-virtualenv
    else
        echo "✗ Could not install python3-venv automatically"
        echo ""
        echo "Please install it manually:"
        echo "  Ubuntu/Debian: sudo apt install python3-venv"
        echo "  RHEL/Fedora:   sudo dnf install python3-virtualenv"
        echo ""
        exit 1
    fi
fi

if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "  → Creating virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/venv"
fi

echo "  → Activating virtual environment..."
source "${SCRIPT_DIR}/venv/bin/activate"

echo "  → Installing dependencies (this may take a few minutes)..."
pip install -q --upgrade pip
pip install -q -r "${SCRIPT_DIR}/export_requirements.txt"

echo "  ✓ Python environment ready"
echo ""

# Create models directory
mkdir -p "${MODELS_DIR}"

# Export model
echo ""
echo "Exporting model (this may take 30-60 minutes)..."
echo "Model: ${SOURCE_MODEL}"
echo "Target: ${MODELS_DIR}"
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
  --model_repository_path "${MODELS_DIR}"

# Verify export
if check_model "${MODELS_DIR}/Qwen/${MODEL_NAME}"; then
    echo ""
    echo "✓ Model export successful!"
    echo ""
    echo "Model location: ${MODELS_DIR}/Qwen/${MODEL_NAME}"
    echo "Model size: $(du -sh ${MODELS_DIR}/Qwen/${MODEL_NAME} | cut -f1)"
    echo ""
else
    echo ""
    echo "✗ Model export may have failed - graph.pbtxt or model files missing"
    echo "Expected location: ${MODELS_DIR}/Qwen/${MODEL_NAME}"
    echo "Checking for model files..."
    ls -la "${MODELS_DIR}/Qwen/${MODEL_NAME}" 2>/dev/null || echo "Directory not found"
    echo ""
    echo "Check the output above for errors"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the OVMS service:"
echo "   cd ${PROJECT_ROOT}"
echo "   docker compose up -d ovms-vlm"
echo ""
echo "2. Wait for model to load (30-60 seconds):"
echo "   watch -n 2 'docker logs dinein_ovms_vlm --tail 5'"
echo ""
echo "3. Verify OVMS is healthy:"
echo "   curl http://localhost:8002/v1/config"
echo ""
echo "4. Check model status:"
echo "   curl http://localhost:8002/v1/models"
echo ""
echo "5. Start the dine-in application:"
echo "   docker compose up -d dinein_app"
echo ""

