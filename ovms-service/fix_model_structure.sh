#!/bin/bash
# Fix OVMS Model Directory Structure
# OVMS requires models to be in versioned directories: model_name/1/
# This script moves model files into the required structure

set -e

MODEL_DIR="models_vlm/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"

echo "Fixing OVMS model directory structure..."

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if version directory already exists
if [ -d "$MODEL_DIR/1" ]; then
    echo "Version directory already exists. Checking if it's populated..."
    if [ -f "$MODEL_DIR/1/openvino_language_model.xml" ]; then
        echo "✓ Model already correctly structured in $MODEL_DIR/1/"
        exit 0
    else
        echo "Version directory exists but is empty or incomplete."
    fi
fi

# Check if model files are in the root directory
if [ ! -f "$MODEL_DIR/openvino_language_model.xml" ]; then
    echo "Error: Model files not found in $MODEL_DIR"
    echo "Expected files like openvino_language_model.xml"
    exit 1
fi

echo "Creating version directory: $MODEL_DIR/1/"
mkdir -p "$MODEL_DIR/1"

echo "Moving model files to version directory..."
# Move all files except subdirectories to version 1
find "$MODEL_DIR" -maxdepth 1 -type f -exec mv {} "$MODEL_DIR/1/" \;

echo "✓ Model structure fixed!"
echo "Structure is now: $MODEL_DIR/1/<model_files>"
echo ""
echo "Verify with:"
echo "  ls -la $MODEL_DIR/1/ | head -10"
echo ""
echo "Now restart OVMS:"
echo "  docker compose --profile ovms restart ovms-vlm"
