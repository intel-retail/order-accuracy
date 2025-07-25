#!/bin/bash
#
# Copyright (C) 2025 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#
set -eo pipefail

PRE_PROCESS="${PRE_PROCESS:=""}" #""|pre-process-backend=vaapi-surface-sharing|pre-process-backend=vaapi-surface-sharing pre-process-config=VAAPI_FAST_SCALE_LOAD_FACTOR=1

if [ "$RENDER_MODE" == "1" ]; then
    OUTPUT="gvawatermark ! videoconvert ! fpsdisplaysink video-sink=autovideosink text-overlay=false signal-fps-measurements=true"
elif [ "$RTSP_OUTPUT" == "1" ]; then
    OUTPUT="gvawatermark ! x264enc ! video/x-h264,profile=baseline ! rtspclientsink location=$RTSP_SERVER/$RTSP_PATH protocols=tcp timeout=0"
else
    OUTPUT="fpsdisplaysink video-sink=fakesink signal-fps-measurements=true"
fi

echo "Run run yolo11s pipeline on $DEVICE with batch size = $BATCH_SIZE"
MODEL_PATH="/home/pipeline-server/models/object_detection/yolo11n/INT8/yolo11n.xml"
SHARED_MODEL_ID="shared_yolo_model"

if [ "$PIPELINE_ID" = "1" ]; then
    gstLaunchCmd="gst-launch-1.0 --verbose \
        filesrc location=$inputsrc ! \
        qtdemux ! \
        h264parse ! \
        avdec_h264 ! \
        queue leaky=downstream max-size-buffers=2 ! \
        gvadetect model-instance-id=$SHARED_MODEL_ID \
            name=detection \
            batch-size=$BATCH_SIZE \
            model=$MODEL_PATH \
            threshold=0.5 \
            device=$DEVICE \
            $PRE_PROCESS $DETECTION_OPTIONS ! \
        gvametaconvert format=json ! \
        tee name=t \
            t. ! queue leaky=downstream ! gvametapublish method=file file-path=/tmp/results/r\$cid-pipeline1.jsonl ! gvafpscounter ! fakesink sync=false async=false \
            t. ! queue leaky=downstream ! $OUTPUT"

elif [ "$PIPELINE_ID" = "2" ]; then
    gstLaunchCmd="gst-launch-1.0 --verbose \
        filesrc location=$inputsrc ! \
        qtdemux ! \
        h264parse ! \
        avdec_h264 ! \
        queue leaky=downstream max-size-buffers=2 ! \
        gvadetect model-instance-id=$SHARED_MODEL_ID \
            name=detection \
            batch-size=$BATCH_SIZE \
            model=$MODEL_PATH \
            threshold=0.5 \
            device=$DEVICE \
            $PRE_PROCESS $DETECTION_OPTIONS ! \
        gvametaconvert format=json ! \
        tee name=t \
            t. ! queue leaky=downstream ! gvametapublish method=file file-path=/tmp/results/r\$cid-pipeline2.jsonl ! gvafpscounter ! fakesink sync=false async=false \
            t. ! queue leaky=downstream ! $OUTPUT"

else
    echo "Invalid PIPELINE_ID: $PIPELINE_ID"
    exit 1
fi

# Actually execute the pipeline command
echo "Running command: $gstLaunchCmd"
eval "$gstLaunchCmd"