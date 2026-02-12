#!/bin/bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/model}"

ACTUAL_MODEL="${MODEL_NAME##*/}"
TARGET_DIR="$MODELS_DIR/${ACTUAL_MODEL}-ov-${PRECISION}"
MODEL_XML="$TARGET_DIR/openvino_language_model.xml"

if [[ -f "$MODEL_XML" ]]; then
  echo "[INFO] VLM already present: $MODEL_XML"
  exit 0
fi

echo "[INFO] Exporting VLM model: $MODEL_NAME ($PRECISION)"

bash /scripts/compress_model.sh "$MODEL_NAME" "$PRECISION" "$TARGET_DIR"
