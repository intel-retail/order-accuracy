#!/bin/bash
set -euo pipefail

MODEL_NAME="$1"
PRECISION="$2"
OUT_DIR="$3"

mkdir -p "$OUT_DIR"

echo "[INFO] Exporting OpenVINO VLM"
echo "  Model: $MODEL_NAME"
echo "  Precision: $PRECISION"
echo "  Output: $OUT_DIR"

export HF_HOME=/model/hf-cache
CLEAN_PRECISION=$(echo "$PRECISION" | tr -d '"')

EXPORT_CMD=(
    optimum-cli export openvino
    --trust-remote-code
    --model "$MODEL_NAME"
    "$OUT_DIR"
    --weight-format "$CLEAN_PRECISION"
)

echo "[INFO] Running: ${EXPORT_CMD[*]}"
if ! "${EXPORT_CMD[@]}"; then
    echo "[ERROR] Model export failed. Removing partial artifacts at $OUT_DIR"
    rm -rf "$OUT_DIR"
    exit 1
fi

echo "[INFO] Model exported successfully to $OUT_DIR"
