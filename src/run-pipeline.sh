#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

set -eo pipefail

# Defaults
PRE_PROCESS="${PRE_PROCESS:-""}" # "" or "pre-process-backend=vaapi-surface-sharing" etc.
DEVICE="${DEVICE:-CPU}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RTSP_SERVER="${RTSP_SERVER:-rtsp://localhost:8554}"
RTSP_PATH="${RTSP_PATH:-stream}"
RENDER_MODE="${RENDER_MODE:-0}"
RTSP_OUTPUT="${RTSP_OUTPUT:-0}"
PIPELINE_ID="${PIPELINE_ID:-1}"
cid="${cid:-0}"  # Set externally or defaults to 0

# Input validation
if [ -z "$inputsrc" ]; then
  echo "❌ Error: inputsrc not specified."
  exit 1
fi

# Choose OUTPUT sink
if [ "$RENDER_MODE" == "1" ]; then
    OUTPUT="gvawatermark ! videoconvert ! fpsdisplaysink video-sink=autovideosink text-overlay=false signal-fps-measurements=true"
elif [ "$RTSP_OUTPUT" == "1" ]; then
    OUTPUT="gvawatermark ! x264enc ! video/x-h264,profile=baseline ! rtspclientsink location=$RTSP_SERVER/$RTSP_PATH protocols=tcp timeout=0"
else
    OUTPUT="fpsdisplaysink video-sink=fakesink signal-fps-measurements=true"
fi



MODEL_PATH="/home/pipeline-server/models/object_detection/yolo11n/INT8/yolo11n.xml"
SHARED_MODEL_ID="shared_yolo_model"

echo "🚀 Running YOLOv11s pipeline on $DEVICE with batch size = $BATCH_SIZE"

# Function to run the pipeline
run_pipeline() {
  local pipeline_id=$1

  echo "▶️ Starting pipeline ID $pipeline_id"
  
  GST_DEBUG="GST_TRACER:7" GST_TRACERS='latency_tracer(flags=pipeline)' \
  gst-launch-1.0 --verbose \
  filesrc location="$inputsrc" ! \
  qtdemux ! $DECODE !\
  queue ! \
  gvadetect batch-size=$BATCH_SIZE \
      model-instance-id=odmodel \
      name=detection \
      model=$MODEL_PATH \
      threshold=0.5 \
      device=$DEVICE \
      inference-interval=3\
      ${PRE_PROCESS:+$PRE_PROCESS} ${DETECTION_OPTIONS:+$DETECTION_OPTIONS} \
  ! gvametaconvert \
  ! tee name=t \
      t. ! queue ! $OUTPUT \
      t. ! queue ! gvametapublish name=destination file-format=json-lines file-path="/tmp/results/r${cid}.jsonl" ! fakesink sync=false async=false \
  2>&1 | tee "/tmp/results/gst-launch_${cid}.log" \
  | (stdbuf -oL sed -n -e 's/^.*current: //p' | stdbuf -oL cut -d , -f 1 > "/tmp/results/pipeline${cid}.log")

}
# Run based on pipeline ID
case "$PIPELINE_ID" in
  1|2)
    run_pipeline "$PIPELINE_ID"
    ;;
  *)
    echo "❌ Invalid PIPELINE_ID: $PIPELINE_ID"
    exit 1
    ;;
esac
