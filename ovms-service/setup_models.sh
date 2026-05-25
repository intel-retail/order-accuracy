#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODELS_DIR="${SCRIPT_DIR}/models"

###############################################
# ARGUMENT PARSING
# --app take-away  (default)
# --app dine-in
###############################################
APP="take-away"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)
            APP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--app take-away|dine-in]"
            exit 1
            ;;
    esac
done

if [ "${APP}" != "take-away" ] && [ "${APP}" != "dine-in" ]; then
    echo "Error: --app must be 'take-away' or 'dine-in' (got: ${APP})"
    exit 1
fi

###############################################
# HARD CODED MODEL REGISTRY
# key = model_name passed to --model_name (also used as path under MODELS_DIR)
# value = HuggingFace source passed to --source_model
###############################################
declare -A MODEL_SOURCES
MODEL_SOURCES["Qwen/Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct"

POTENTIAL_SOURCE_DIRS=(
    "${HOME}/ovms-vlm/models"
    "/opt/ovms/models"
    "${PROJECT_ROOT}/../ovms-vlm/models"
)

###############################################
# LOAD CONFIG FROM APP-SPECIFIC .env
###############################################
ENV_FILE="${PROJECT_ROOT}/${APP}/.env"
echo "App: ${APP}  →  reading config from ${ENV_FILE}"
if [ -f "${ENV_FILE}" ]; then
    OVMS_MODEL_NAME_ENV=$(grep -E '^OVMS_MODEL_NAME=' "${ENV_FILE}" | head -1 | cut -d'=' -f2- | tr -d '"\r')
fi
# Fall back to the hard-coded source model if .env is missing or unset
OVMS_MODEL_NAME_ENV="${OVMS_MODEL_NAME_ENV:-Qwen/Qwen2.5-VL-7B-Instruct}"

# Read TARGET_DEVICE: shell env var takes precedence over take-away/.env, default GPU
_TARGET_DEVICE_FILE=$(grep -E '^TARGET_DEVICE=' "${ENV_FILE}" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"\r')
TARGET_DEVICE_ENV="${TARGET_DEVICE:-${_TARGET_DEVICE_FILE:-GPU}}"

# Read VLM_PRECISION: shell env var takes precedence over take-away/.env, default int8
_VLM_PRECISION_FILE=$(grep -E '^VLM_PRECISION=' "${ENV_FILE}" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"\r')
VLM_PRECISION_ENV="${VLM_PRECISION:-${_VLM_PRECISION_FILE:-int8}}"

# Read CACHE_SIZE: KV cache size in GB for OVMS.  Default is 4 GB which is
# adequate for a single-station app (max_num_seqs=4).  On GPU the KV cache
# occupies VRAM; Intel Arc A770 has 8–16 GB VRAM and the INT8 model already
# consumes ~8 GB, so values above 8 will overflow to system RAM and can cause
# OOM on 32 GB platforms (ITEP-91499).  Users with more VRAM/RAM can raise
# this via `export CACHE_SIZE=8` before running setup_models.sh.
_CACHE_SIZE_FILE=$(grep -E '^CACHE_SIZE=' "${ENV_FILE}" 2>/dev/null | head -1 | cut -d'=' -f2- | tr -d '"\r')
CACHE_SIZE_ENV="${CACHE_SIZE:-${_CACHE_SIZE_FILE:-4}}"

###############################################
echo "=========================================="
echo "OVMS Model Setup for Order Accuracy"
echo "=========================================="
echo ""

###############################################
# Validate that the OpenVINO IR XML files in a model directory are non-empty
# and have a valid XML header.  A system running out of memory during export
# (e.g. WCL / MTL platforms with 32 GB RAM) can write a partial, corrupt XML
# file that passes a file-existence check but fails when OpenVINO tries to
# load it (ITEP-90866 / ITEP-91499).
###############################################
validate_model_xml() {
    local model_path="$1"
    local found_valid=0

    for xml_file in "${model_path}"/*.xml; do
        [ -e "${xml_file}" ] || continue

        if [ ! -s "${xml_file}" ]; then
            echo "  ✗ Model XML file is empty: ${xml_file}"
            return 1
        fi

        # OpenVINO IR XML files must start with the '<?xml' processing instruction.
        # A truncated/corrupt export will typically start with binary garbage or
        # be missing this header entirely.
        local header
        header=$(head -c 5 "${xml_file}" 2>/dev/null || true)
        if [ "${header}" != "<?xml" ]; then
            echo "  ✗ Model XML appears corrupt (invalid header in ${xml_file##*/})"
            echo "    This often happens when the export ran out of memory and wrote an"
            echo "    incomplete file. Re-run setup_models.sh on a system with at least"
            echo "    64 GB RAM, or delete ${model_path} and retry."
            return 1
        fi

        found_valid=1
    done

    if [ "${found_valid}" -eq 0 ]; then
        return 1
    fi

    return 0
}

###############################################
# Warn when available system RAM is below the threshold needed for a safe
# model export.  The Qwen2.5-VL-7B INT8 quantisation step can temporarily
# require >32 GB of RAM; running it on a 32 GB platform can cause partial /
# corrupt writes without an obvious error (ITEP-90866 / ITEP-91499).
###############################################
check_memory_for_export() {
    local recommended_gb=64
    local total_gb=0

    if [ -f /proc/meminfo ]; then
        total_gb=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
    fi

    if [ "${total_gb}" -gt 0 ] && [ "${total_gb}" -lt "${recommended_gb}" ]; then
        echo ""
        echo "  ⚠ WARNING: System RAM is ${total_gb} GB (${recommended_gb} GB recommended for model export)."
        echo "    On platforms with less than ${recommended_gb} GB RAM (e.g. Wildcat Lake / Meteor Lake"
        echo "    with 32 GB RAM), the INT8 quantisation step may exhaust memory and produce"
        echo "    corrupt model files that cannot be loaded by OVMS."
        echo "    If export fails or OVMS reports 'Unable to read the model', re-run"
        echo "    setup_models.sh on a system with at least ${recommended_gb} GB RAM."
        echo ""
    fi
}

###############################################
check_model() {
    local model_path="$1"

    if [ ! -d "${model_path}" ]; then
        return 1
    fi

    # A successfully exported OVMS model always has:
    #   - graph.pbtxt  (created by export_model.py for all VLM/LLM models)
    #   - at least one .xml file (OpenVINO IR; name varies by model architecture)
    if [ -f "${model_path}/graph.pbtxt" ] && \
       ls "${model_path}"/*.xml > /dev/null 2>&1; then
        # Also validate the XML files are not corrupt (ITEP-90866)
        if validate_model_xml "${model_path}"; then
            echo "  ✓ Model found at ${model_path}"
            return 0
        else
            echo "  ✗ Model at ${model_path} has corrupt XML files — will re-export"
            return 1
        fi
    else
        echo "  ✗ Model not ready at ${model_path} (missing graph.pbtxt or .xml files)"
        return 1
    fi
}

###############################################
# Migrate a model from a legacy path to the expected target path.
# Pre-v2026.0.0 releases exported to Qwen2.5-VL-7B-Instruct-ov-int8;
# current releases use Qwen/Qwen2.5-VL-7B-Instruct.
# If the new path doesn't exist but an old one does, copy it so that
# users don't need to re-export the full 7B model from scratch.
###############################################
migrate_legacy_model() {
    local target_path="$1"

    if [ -d "${target_path}" ]; then
        return 0  # Already at the new location, nothing to do
    fi

    local legacy_paths=(
        "${MODELS_DIR}/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"
        "${MODELS_DIR}/Qwen2.5-VL-7B-Instruct-ov-int8"
    )

    for legacy_path in "${legacy_paths[@]}"; do
        if check_model "${legacy_path}"; then
            echo "  Legacy model found at ${legacy_path}"
            echo "  Migrating to new path: ${target_path} ..."
            mkdir -p "$(dirname "${target_path}")"
            cp -r "${legacy_path}" "${target_path}"
            echo "  ✓ Migration complete (original preserved at ${legacy_path})"
            return 0
        fi
    done

    return 1
}

###############################################
# Replace the host MODELS_DIR path with the container path /models in graph.pbtxt.
# export_model.py always writes the absolute host path, but OVMS runs inside a
# container where the models directory is mounted at /models.
###############################################
patch_graph_pbtxt_paths() {
    local model_name="$1"
    local graph_file="${MODELS_DIR}/${model_name}/graph.pbtxt"

    if [ ! -f "${graph_file}" ]; then
        return 0
    fi

    if grep -qF "${MODELS_DIR}" "${graph_file}"; then
        sed -i "s|${MODELS_DIR}|/models|g" "${graph_file}"
        echo "  ✓ graph.pbtxt paths patched (host path → /models)"
    fi
}

###############################################
# Update device in graph.pbtxt when TARGET_DEVICE doesn't match.
# The device field is just a text string — edit it directly with sed.
# No need to re-run export_model.py (that would require HuggingFace source).
###############################################
update_graph_pbtxt_device() {
    local model_name="$1"
    local graph_file="${MODELS_DIR}/${model_name}/graph.pbtxt"

    if [ ! -f "${graph_file}" ]; then
        return 0
    fi

    # Always patch container paths first (handles models that existed before this fix)
    patch_graph_pbtxt_paths "${model_name}"

    local current_device
    current_device=$(grep -oP '(?<=device: ")[^"]+' "${graph_file}" || true)

    if [ "${current_device}" = "${TARGET_DEVICE_ENV}" ]; then
        echo "  ✓ graph.pbtxt device already set to ${TARGET_DEVICE_ENV}"
        return 0
    fi

    echo "  Device mismatch: graph.pbtxt has \"${current_device}\", TARGET_DEVICE is \"${TARGET_DEVICE_ENV}\""
    sed -i "s|device: \"${current_device}\"|device: \"${TARGET_DEVICE_ENV}\"|g" "${graph_file}"
    echo "  ✓ graph.pbtxt device updated: ${current_device} → ${TARGET_DEVICE_ENV}"
}

###############################################
ask_user_model() {
    local model_name="$1"
    echo "Setting up ${model_name}..."
    return 0
}

###############################################
setup_python_env() {

    if [ ! -f "${SCRIPT_DIR}/export_model.py" ]; then
        echo "[1/3] Downloading OVMS export tools..."

        EXPORT_BASE_URL="https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2026/0/demos/common/export_models"

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
    echo "Exporting ${MODEL_NAME} (device: ${TARGET_DEVICE_ENV}, precision: ${VLM_PRECISION_ENV}, cache_size: ${CACHE_SIZE_ENV} GB)"
    echo ""

    # Build optional --target_device argument; CPU is the default so omit it
    local target_device_args=()
    if [ "${TARGET_DEVICE_ENV}" != "CPU" ]; then
        target_device_args=(--target_device "${TARGET_DEVICE_ENV}")
    fi

    python "${SCRIPT_DIR}/export_model.py" text_generation \
      --source_model "${SOURCE_MODEL}" \
      --weight-format "${VLM_PRECISION_ENV}" \
      --pipeline_type VLM_CB \
      "${target_device_args[@]}" \
      --cache_size "${CACHE_SIZE_ENV}" \
      --max_num_seqs 4 \
      --max_num_batched_tokens 8192 \
      --enable_prefix_caching True \
      --config_file_path "${MODELS_DIR}/config.json" \
      --model_repository_path "${MODELS_DIR}" \
      --model_name "${MODEL_NAME}"
}

###############################################
mkdir -p "${MODELS_DIR}"

_PYTHON_ENV_READY=0
ensure_python_env() {
    if [ "${_PYTHON_ENV_READY}" -eq 0 ]; then
        echo "Setting up Python environment..."
        echo ""
        setup_python_env
        echo ""
        echo "✓ Python environment ready"
        _PYTHON_ENV_READY=1
    fi
}

###############################################
# MAIN MODEL LOOP
###############################################

for MODEL_NAME in "${!MODEL_SOURCES[@]}"; do

    SOURCE_MODEL="${MODEL_SOURCES[$MODEL_NAME]}"
    TARGET_PATH="${MODELS_DIR}/${MODEL_NAME}"

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
        update_graph_pbtxt_device "${MODEL_NAME}"
        continue
    fi

    ###########################################
    # Migrate from legacy model path (pre-v2026.0.0)
    ###########################################
    if migrate_legacy_model "${TARGET_PATH}"; then
        echo "✓ Migrated from legacy path"
        update_graph_pbtxt_device "${MODEL_NAME}"
        continue
    fi

    ###########################################
    # Check copy from external sources
    ###########################################
    for SOURCE_DIR in "${POTENTIAL_SOURCE_DIRS[@]}"; do
        if [ -d "${SOURCE_DIR}/${MODEL_NAME}" ]; then

            if check_model "${SOURCE_DIR}/${MODEL_NAME}"; then
                echo "Copying model from ${SOURCE_DIR}"

                mkdir -p "$(dirname "${MODELS_DIR}/${MODEL_NAME}")"
                cp -r "${SOURCE_DIR}/${MODEL_NAME}" "$(dirname "${MODELS_DIR}/${MODEL_NAME}")/"

                echo "✓ Copied ${MODEL_NAME}"
                update_graph_pbtxt_device "${MODEL_NAME}"
                continue 2
            fi
        fi
    done

    ###########################################
    # Download/export automatically
    ###########################################
    echo ""
    echo "Model not found locally."
    echo "Downloading and exporting from HuggingFace..."
    echo ""

    # Warn if system RAM is below recommended threshold (ITEP-91499)
    check_memory_for_export

    ensure_python_env
    export_model "${MODEL_NAME}" "${SOURCE_MODEL}"

    ###########################################
    # Verify and patch container paths
    ###########################################
    if check_model "${TARGET_PATH}"; then
        patch_graph_pbtxt_paths "${MODEL_NAME}"
        echo "✓ Export successful for ${MODEL_NAME}"
    else
        echo "✗ Export failed for ${MODEL_NAME}"
        exit 1
    fi

done

###############################################
# GENERATE OVMS config.json (mediapipe_config_list)
# export_model.py generates a model_config_list config which does NOT support
# the /v3/chat/completions endpoint. OVMS LLM/VLM serving requires a
# mediapipe_config_list entry (pointing at the graph.pbtxt). The model name
# is read from OVMS_MODEL_NAME in take-away/.env so it always matches what
# the OA service requests (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").
###############################################
generate_ovms_config() {
    echo ""
    echo "------------------------------------------"
    echo "Generating OVMS config.json "
    echo "------------------------------------------"
    echo "  OVMS_MODEL_NAME: ${OVMS_MODEL_NAME_ENV}"

    local config_entries=""
    local first=1

    for MODEL_NAME in "${!MODEL_SOURCES[@]}"; do
        local TARGET_PATH="${MODELS_DIR}/${MODEL_NAME}"

        if [ ! -d "${TARGET_PATH}" ]; then
            echo "  Skipping ${MODEL_NAME} (not found)"
            continue
        fi

        if [ $first -ne 1 ]; then
            config_entries+=","
        fi
        config_entries+="
        {
            \"name\": \"${OVMS_MODEL_NAME_ENV}\",
            \"base_path\": \"${MODEL_NAME}\"
        }"
        first=0
        echo "  + ${OVMS_MODEL_NAME_ENV}  →  ${MODEL_NAME}"
    done

    cat > "${MODELS_DIR}/config.json" << EOF
{
    "model_config_list": [],
    "mediapipe_config_list": [${config_entries}
    ]
}
EOF

    echo "  ✓ config.json written to ${MODELS_DIR}/config.json"
}

generate_ovms_config

###############################################
echo ""
echo "=========================================="
echo "✓ All Model Setup Complete!"
echo "=========================================="

###############################################
# SHARED VENV FOR EASYOCR + YOLO DOWNLOADS
# Ubuntu 24.04 (PEP 668) blocks pip install on system Python.
# A single shared venv is used to download both EasyOCR and YOLO
# model weights to disk. The containers never use this venv —
# they have their own Python environments.
###############################################
TAKEAWAY_DIR="$(dirname "${SCRIPT_DIR}")/take-away"
EASYOCR_DIR="${TAKEAWAY_DIR}/models/easyocr"
YOLO_MODEL_DIR="${TAKEAWAY_DIR}/models"
YOLO_DATASETS_DIR="${TAKEAWAY_DIR}/datasets"
YOLO_PT="${YOLO_MODEL_DIR}/yolo11n.pt"
YOLO_FP32_DIR="${YOLO_MODEL_DIR}/yolo11n_openvino_model"
YOLO_INT8_DIR="${YOLO_MODEL_DIR}/yolo11n_int8_openvino_model"
MODEL_TOOLS_VENV="${SCRIPT_DIR}/model-tools-venv"

_easyocr_present() {
    [ -f "${EASYOCR_DIR}/craft_mlt_25k.pth" ] && [ -f "${EASYOCR_DIR}/english_g2.pth" ]
}

_yolo_all_present() {
    [ -f "${YOLO_PT}" ] && [ -d "${YOLO_FP32_DIR}" ] && [ -d "${YOLO_INT8_DIR}" ]
}

# Only create the venv if at least one model set is missing
if _easyocr_present && _yolo_all_present; then
    echo ""
    echo "✓ EasyOCR and YOLO models already present, skipping."
else
    echo ""
    echo "Setting up shared Python environment for model downloads..."
    if [ ! -d "${MODEL_TOOLS_VENV}" ] || [ ! -f "${MODEL_TOOLS_VENV}/bin/pip" ]; then
        python3 -m venv "${MODEL_TOOLS_VENV}" --clear
    fi
    source "${MODEL_TOOLS_VENV}/bin/activate"
    pip install -q --upgrade pip
    # CPU-only torch (mirrors the frame-selector Dockerfile)
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -q ultralytics openvino easyocr
    echo "  ✓ Dependencies installed"
fi

###############################################
# EASYOCR MODEL DOWNLOAD
###############################################
echo ""
echo "=========================================="
echo "EasyOCR Model Setup"
echo "=========================================="
echo "Target: ${EASYOCR_DIR}"
echo ""

if _easyocr_present; then
    echo "✓ EasyOCR models already present, skipping download."
else
    mkdir -p "${EASYOCR_DIR}"
    echo "Downloading EasyOCR models to ${EASYOCR_DIR} ..."
    echo "  (detection model ~90MB + recognition model ~100MB)"

    EASYOCR_DIR="${EASYOCR_DIR}" python3 -c "
import os
import easyocr
easyocr_dir = os.environ['EASYOCR_DIR']
easyocr.Reader(['en'], gpu=False, verbose=True, model_storage_directory=easyocr_dir)
print('Done.')
"

    if _easyocr_present; then
        echo "✓ EasyOCR models downloaded to ${EASYOCR_DIR}"
    else
        echo "✗ EasyOCR download may have failed — check ${EASYOCR_DIR}"
    fi
fi

###############################################
# YOLO MODEL DOWNLOAD AND QUANTIZATION
# YOLO11n is used by the frame-selector service for frame quality scoring.
# Three artifacts are pre-built here so the container starts instantly:
#   take-away/models/yolo11n.pt                    — base PyTorch weights
#   take-away/models/yolo11n_openvino_model/        — FP32 OpenVINO IR
#   take-away/models/yolo11n_int8_openvino_model/   — INT8 quantized OpenVINO IR
# docker-compose mounts take-away/models/ → /app/models inside the container.
###############################################
echo ""
echo "=========================================="
echo "YOLO Model Setup (Frame Selector)"
echo "=========================================="
echo "Target: ${YOLO_MODEL_DIR}"
echo ""

if _yolo_all_present; then
    echo "✓ YOLO models already present, skipping."
else
    echo "One or more YOLO model artifacts are missing:"
    [ ! -f "${YOLO_PT}" ]       && echo "  - yolo11n.pt"
    [ ! -d "${YOLO_FP32_DIR}" ] && echo "  - yolo11n_openvino_model/  (FP32 OpenVINO)"
    [ ! -d "${YOLO_INT8_DIR}" ] && echo "  - yolo11n_int8_openvino_model/  (INT8 OpenVINO)"
    echo ""
    echo "Downloading and quantizing YOLO models..."

    mkdir -p "${YOLO_MODEL_DIR}" "${YOLO_DATASETS_DIR}"

        echo "[1/2] Downloading yolo11n.pt and exporting OpenVINO models..."
        YOLO_MODEL_DIR="${YOLO_MODEL_DIR}" YOLO_DATASETS_DIR="${YOLO_DATASETS_DIR}" \
        python3 - << 'PYEOF'
import os, sys
from pathlib import Path
from ultralytics import YOLO

model_dir    = Path(os.environ["YOLO_MODEL_DIR"])
datasets_dir = Path(os.environ["YOLO_DATASETS_DIR"])

# Tell ultralytics where to store datasets (needed for INT8 calibration)
os.environ["YOLO_DATASETS_DIR"] = str(datasets_dir)

yolo_pt   = model_dir / "yolo11n.pt"
fp32_dir  = model_dir / "yolo11n_openvino_model"
int8_dir  = model_dir / "yolo11n_int8_openvino_model"

# ── Step 1: Download base weights ────────────────────────────────────────────
if not yolo_pt.exists():
    print("  Downloading yolo11n.pt ...")
    orig = os.getcwd()
    os.chdir(str(model_dir))
    YOLO("yolo11n.pt")   # ultralytics downloads to CWD when the file is absent
    os.chdir(orig)
    print(f"  ✓ Downloaded: {yolo_pt}")
else:
    print(f"  yolo11n.pt already exists: {yolo_pt}")

# ── Step 2: FP32 OpenVINO export ─────────────────────────────────────────────
if not fp32_dir.exists():
    print("  Exporting to OpenVINO FP32 ...")
    orig = os.getcwd()
    os.chdir(str(model_dir))
    YOLO(str(yolo_pt)).export(format="openvino", half=False)
    os.chdir(orig)
    print(f"  ✓ FP32 export: {fp32_dir}")
else:
    print(f"  FP32 model already exists: {fp32_dir}")

# ── Step 3: INT8 quantization ────────────────────────────────────────────────
if not int8_dir.exists():
    print("  Quantizing to OpenVINO INT8 (downloads COCO128 ~7 MB if needed) ...")
    orig = os.getcwd()
    os.chdir(str(model_dir))
    YOLO(str(yolo_pt)).export(format="openvino", int8=True, data="coco128.yaml")
    # ultralytics exports INT8 to "yolo11n_openvino_model/" in CWD;
    # rename it so it doesn't overwrite the FP32 export.
    default_out = Path("yolo11n_openvino_model")
    if default_out.exists() and not int8_dir.exists():
        default_out.rename(int8_dir.name)
    os.chdir(orig)
    print(f"  ✓ INT8 quantization: {int8_dir}")
else:
    print(f"  INT8 model already exists: {int8_dir}")

print("YOLO export complete.")
PYEOF

        echo "[2/2] Verifying YOLO artifacts..."
        _ok=1
        if [ -f "${YOLO_PT}" ]; then
            echo "  ✓ yolo11n.pt"
        else
            echo "  ✗ yolo11n.pt missing"
            _ok=0
        fi
        if [ -d "${YOLO_FP32_DIR}" ]; then
            echo "  ✓ yolo11n_openvino_model/"
        else
            echo "  ✗ yolo11n_openvino_model/ missing"
            _ok=0
        fi
        if [ -d "${YOLO_INT8_DIR}" ]; then
            echo "  ✓ yolo11n_int8_openvino_model/"
        else
            echo "  ✗ yolo11n_int8_openvino_model/ missing"
            _ok=0
        fi

        if [ "${_ok}" -eq 1 ]; then
            echo "✓ YOLO models ready"
        else
            echo "✗ Some YOLO artifacts are missing — check ${YOLO_MODEL_DIR}"
        fi
fi

# Deactivate the shared venv if it was activated
deactivate 2>/dev/null || true

echo ""
echo "=========================================="
echo "✓ All Setup Complete!"
echo "=========================================="
