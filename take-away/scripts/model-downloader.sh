#!/bin/bash
set -euo pipefail

APP_CONFIG="${APP_CONFIG:-/config/application.yaml}"
MODELS_DIR="${MODELS_DIR:-/model}"

echo "[INFO] Using app config: $APP_CONFIG"
mkdir -p "$MODELS_DIR"

if [[ ! -f "$APP_CONFIG" ]]; then
  echo "[ERROR] application.yaml not found"
  exit 1
fi

MODEL_NAME=$(yq '.vlm.model' "$APP_CONFIG" | tr -d '"')
PRECISION=$(yq '.vlm.precision // "int8"' "$APP_CONFIG" | tr -d '"')

if [[ "$MODEL_NAME" == "null" || -z "$MODEL_NAME" ]]; then
  echo "[ERROR] vlm.model not defined in application.yaml"
  exit 1
fi

export MODEL_NAME
export PRECISION
export MODELS_DIR

echo "[INFO] VLM model: $MODEL_NAME"
echo "[INFO] Precision: $PRECISION"

bash /scripts/model-handler.sh
