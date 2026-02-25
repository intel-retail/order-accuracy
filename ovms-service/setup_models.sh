#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${SCRIPT_DIR}/models"

###############################################
# HARD CODED MODEL REGISTRY
###############################################
declare -A MODEL_SOURCES
MODEL_SOURCES["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"

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

    if [ -f "${model_path}/graph.pbtxt" ] &&
       [ -f "${model_path}/openvino_language_model.xml" ] &&
       [ -f "${model_path}/openvino_language_model.bin" ]; then
        return 0
    else
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
        echo "Downloading OVMS export tools..."

        EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/4/demos/common/export_models"

        curl -fsSL "${EXPORT_BASE_URL}/export_model.py" -o "${SCRIPT_DIR}/export_model.py"
        curl -fsSL "${EXPORT_BASE_URL}/requirements.txt" -o "${SCRIPT_DIR}/export_requirements.txt"
    fi

    if [ ! -d "${SCRIPT_DIR}/venv" ]; then
        python3 -m venv "${SCRIPT_DIR}/venv"
    fi

    source "${SCRIPT_DIR}/venv/bin/activate"

    pip install -q --upgrade pip
    pip install -q -r "${SCRIPT_DIR}/export_requirements.txt"
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
      --model_repository_path "${MODELS_DIR}"
}

###############################################
mkdir -p "${MODELS_DIR}"

setup_python_env

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
