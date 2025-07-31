#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#


# Source the export_yolo_model function from /workspace/downloadAndQuantizeModel.sh
source /workspace/downloadAndQuantizeModel.sh

modelPrecisionFP16INT8="FP16-INT8"
modelPrecisionFP32INT8="FP32-INT8"
modelPrecisionFP32="FP32"
REFRESH_MODE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --refresh)
            echo "running model downloader in refresh mode"
            REFRESH_MODE=1
            ;;
        *)
            echo "Invalid flag: $1" >&2
            exit 1
            ;;
    esac
    shift
done

MODEL_NAME="yolo11n"
MODEL_TYPE="yolo_v11"

# Debugging output
echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "REFRESH_MODE: $REFRESH_MODE"

MODEL_EXEC_PATH="$(dirname "$(readlink -f "$0")")"
# Allow override of modelDir via environment variable (for container flexibility)
modelDir="${MODELS_DIR:-$(dirname "$MODEL_EXEC_PATH")/models}"
mkdir -p "$modelDir"
cd "$modelDir" || { echo "Failure to cd to $modelDir"; exit 1; }

# Print the actual modelDir for debugging
echo "[DEBUG] Downloading models to: $modelDir"

if [ "$REFRESH_MODE" -eq 1 ]; then
    # cleaned up all downloaded files so it will re-download all files again
    echo "In refresh mode, clean the existing downloaded models if any..."
    (
        cd "$MODEL_EXEC_PATH"/.. || echo "failed to cd to $MODEL_EXEC_PATH/.."
        make clean-models
    )
fi

pipelineZooModel="https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/"


### Run custom downloader section below:
# Call export_yolo_model after Python conversion (if needed)
export_yolo_model



echo "###################### Model downloading has been completed successfully #########################"
