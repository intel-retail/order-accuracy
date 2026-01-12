#!/bin/bash
set -e

MODEL_XML="/model/Qwen2.5-VL-7B-Instruct-ov-int8/openvino_language_model.xml"
HF_CACHE="/hf-cache"

echo "[INFO] OA Entrypoint started"

if [[ ! -f "$MODEL_XML" ]]; then
  echo "[INFO] Model not found — downloading"
  bash /scripts/model-downloader.sh
else
  echo "[INFO] Model already available"

fi

exec uvicorn main:app --host 0.0.0.0 --port 8000
