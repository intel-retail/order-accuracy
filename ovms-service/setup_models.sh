#!/bin/bash

#
# OVMS Model Setup Script for Order Accuracy
# This script sets up the OVMS model files needed for the VLM backend
# It will automatically export models if not found
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${PROJECT_ROOT}/models"
SOURCE_MODELS_DIR="/home/intel/jsaini/ovms-vlm/models_vlm"
MODEL_NAME="Qwen2.5-VL-7B-Instruct"
SOURCE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

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

# Check if we can copy from existing ovms-vlm installation
if [ -d "${SOURCE_MODELS_DIR}/Qwen" ]; then
    echo "✓ Found existing OVMS models at ${SOURCE_MODELS_DIR}"
    echo ""
    
    # Check for Qwen2.5-VL-7B-Instruct (current model)
    if check_model "${SOURCE_MODELS_DIR}/Qwen/${MODEL_NAME}"; then
        echo "  → Copying ${MODEL_NAME} model..."
        mkdir -p "${MODELS_DIR}/Qwen"
        cp -r "${SOURCE_MODELS_DIR}/Qwen/${MODEL_NAME}" "${MODELS_DIR}/Qwen/"
        echo "  ✓ Model copied successfully"
        
        echo ""
        echo "✓ Model setup complete!"
        echo ""
        echo "Model location: ${MODELS_DIR}/Qwen/${MODEL_NAME}"
        echo "Model size: $(du -sh ${MODELS_DIR}/Qwen/${MODEL_NAME} | cut -f1)"
        echo ""
        exit 0
    else
        echo "  ⚠ Model ${MODEL_NAME} not found in ${SOURCE_MODELS_DIR}"
        echo "  Will proceed with automatic export..."
        echo ""
    fi
fi

# No existing models found - export from HuggingFace
echo "No pre-exported models found. Will export from HuggingFace..."
echo ""
echo "This will:"
echo "  1. Download ${SOURCE_MODEL} (~7GB)"
echo "  2. Convert to OpenVINO format with INT8 quantization"
echo "  3. Create graph.pbtxt for OVMS MediaPipe mode"
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
    echo "    --cache_size 10 \\"
    echo "    --config_file_path models_vlm/config.json \\"
    echo "    --model_repository_path models_vlm"
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
  --cache_size 10 \
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
echo "   cd $(dirname ${SCRIPT_DIR})"
echo "   docker compose --profile ovms up -d"
echo ""
echo "2. Wait for model to load (30-60 seconds):"
echo "   watch -n 2 'docker logs oa_ovms_vlm --tail 5'"
echo ""
echo "3. Verify OVMS is healthy:"
echo "   curl http://localhost:8001/v1/config"
echo ""
echo "4. Check model status:"
echo "   curl http://localhost:8001/v1/models"
echo ""
echo "5. Start parallel pipeline with OVMS backend:"
echo "   docker compose --profile parallel --profile ovms up -d"
echo ""

