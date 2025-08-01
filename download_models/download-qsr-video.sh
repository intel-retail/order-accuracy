#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

CONFIG_FILE="qsr_videos_download.txt"
OUTPUT_DIR="/sample-videos"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi
# Print proxy settings for debugging
echo "Using HTTP_PROXY: ${HTTP_PROXY}"
echo "Using HTTPS_PROXY: ${HTTPS_PROXY}"
# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Read the video entries from config file
while read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    [[ $line =~ ^#.*$ ]] && continue
    [[ -z $line ]] && continue
    
    # Read name and URL from the line
    read -r video_name video_url <<< "$line"
    
    echo "Downloading video: $video_name from $video_url"
    wget --no-check-certificate \
        -e use_proxy=yes \
        -e http_proxy=$HTTP_PROXY \
        -e https_proxy=$HTTPS_PROXY \
        -O "$OUTPUT_DIR/$video_name" "$video_url"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $video_name"
    else
        echo "Failed to download $video_name"
    fi
done < "$CONFIG_FILE"