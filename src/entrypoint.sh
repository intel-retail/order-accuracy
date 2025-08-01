#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

checkBatchSize() {
	if [ "$BATCH_SIZE" -lt 0 ]
	then
		echo "Invalid: BATCH_SIZE should be >= 0: $BATCH_SIZE"
		exit 1
	elif [ "$BATCH_SIZE" -gt 1024 ]
	then
		echo "Invalid: BATCH_SIZE should be <= 1024: $BATCH_SIZE"
		exit 1
	fi
	echo "Ok, BATCH_SIZE = $BATCH_SIZE"
}

cid_count="${cid_count:=0}"
CONTAINER_NAME="${CONTAINER_NAME:=gst}"
stream_density_mount="${stream_density_mount:=}"
stream_density_params="${stream_density_params:=}"
cl_cache_dir="${cl_cache_dir:=$HOME/.cl-cache}"
BATCH_SIZE="${BATCH_SIZE:=0}"
DEVICE="${DEVICE:="CPU"}" #GPU|CPU|MULTI:GPU,CPU

show_help() {
	echo "usage: \"--pipeline_script_choice\" requires an argument run-pipeline.sh"
}






if [ "$PIPELINE_SCRIPT" != "run-pipeline.sh" ]

then
	echo "Error on your input: $PIPELINE_SCRIPT"
	show_help
	exit
fi

echo "Run gst pipeline profile $PIPELINE_SCRIPT"
cd /home/pipeline-server || exit

rmDocker=--rm
if [ -n "$DEBUG" ]
then
	# when there is non-empty DEBUG env, the output of app outputs to the console for easily debugging
	rmDocker=
fi

echo "OCR_RECLASSIFY_INTERVAL=$OCR_RECLASSIFY_INTERVAL  BARCODE_RECLASSIFY_INTERVAL=$BARCODE_RECLASSIFY_INTERVAL"

echo "$rmDocker"
bash_cmd="/home/pipeline-server/$PIPELINE_SCRIPT"
chmod +x "$bash_cmd"
inputsrc="$INPUTSRC"


# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
CONTAINER_NAME="${CONTAINER_NAME//\"/}" # Ensure to remove all double quotes from CONTAINER_NAME
cid="${cid}"_${CONTAINER_NAME}
echo "CONTAINER_NAME: ${CONTAINER_NAME}"
echo "cid: $cid"



cl_cache_dir="/home/pipeline-server/.cl-cache" \
DISPLAY="$DISPLAY" \
RESULT_DIR="/tmp/result" \
DEVICE="$DEVICE" \
BATCH_SIZE="$BATCH_SIZE" \
PRE_PROCESS="$PRE_PROCESS" \
BARCODE_RECLASSIFY_INTERVAL="$BARCODE_INTERVAL" \
OCR_RECLASSIFY_INTERVAL="$OCR_INTERVAL" \
OCR_DEVICE="$OCR_DEVICE" \
LOG_LEVEL="$LOG_LEVEL" \
GST_DEBUG="$GST_DEBUG" \
cid="$cid" \
inputsrc="/home/pipeline-server/sample-media/qsr-usecase.mp4" \
OCR_RECLASSIFY_INTERVAL="$OCR_RECLASSIFY_INTERVAL" \
BARCODE_RECLASSIFY_INTERVAL="$BARCODE_RECLASSIFY_INTERVAL" \
# Add this at the beginning of run-pipeline.sh
if [ ! -f "$inputsrc" ]; then
    echo "Error: Input file not found: $inputsrc"
    exit 1
fi

create_pipeline_files() {
    local cid=$1
    local base_path="/tmp/results"
    
    # Array of file specifications [prefix:extension]
    local file_specs=(
        "r:jsonl"
        "gst-launch_:log"
        "pipeline:log"
    )
    
    for spec in "${file_specs[@]}"; do
        IFS=':' read -r prefix ext <<< "$spec"
        local file_path="${base_path}/${prefix}${cid}.${ext}"
        touch "$file_path"
        chown 1000:1000 "$file_path"
    done
}
# Run pipeline 1 in background
(
	export DEVICE="${DEVICE}"
	export inputsrc="${inputsrc}"
	export BATCH_SIZE="${BATCH_SIZE}"
    export PIPELINE_ID=1
    export cid="${cid}_overhead_view"
	echo "Pipeline ${pipeline_id} cid value: $cid"
    create_pipeline_files "$cid"
    "$bash_cmd"
) &

# Run pipeline 2 in background
(
    export PIPELINE_ID=2
	export DEVICE="${DEVICE}"
	export inputsrc="${inputsrc}"
	export BATCH_SIZE="${BATCH_SIZE}"
    export cid="${cid}_side_view"
	echo "Pipeline ${pipeline_id} cid value: $cid"
    create_pipeline_files "$cid"
    "$bash_cmd"
) &

wait