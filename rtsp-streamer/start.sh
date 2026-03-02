#!/bin/sh
set -eu

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
MEDIAMTX_BIN=${MEDIAMTX_BIN:-/opt/rtsp-streamer/mediamtx}

# Loop configuration
# LOOP_COUNT: number of times to play the video (default -1 = infinite)
#   -1 = infinite loop
#    1 = play once
#    2 = play twice with optional blank frames between
LOOP_COUNT=${LOOP_COUNT:--1}

# BLANK_DURATION: seconds of black frames between loops (default 0 = no blanks)
# Only used when LOOP_COUNT > 1
BLANK_DURATION=${BLANK_DURATION:-0}

if [ ! -d "$MEDIA_DIR" ]; then
  echo "Media directory $MEDIA_DIR does not exist" >&2
  exit 1
fi

if [ ! -x "$MEDIAMTX_BIN" ]; then
  echo "mediamtx binary $MEDIAMTX_BIN not found or not executable" >&2
  exit 1
fi

set -- "$MEDIA_DIR"/*.mp4
if [ ! -e "$1" ]; then
  echo "No .mp4 files found in $MEDIA_DIR" >&2
  exit 1
fi

"$MEDIAMTX_BIN" >/tmp/mediamtx.log 2>&1 &
mediamtx_pid=$!
pids="$mediamtx_pid"

# Wait for RTSP server to accept connections
retry=50
while ! nc -z 127.0.0.1 "$RTSP_PORT" >/dev/null 2>&1; do
  retry=$((retry - 1))
  if [ "$retry" -le 0 ]; then
    echo "RTSP server failed to start on port $RTSP_PORT" >&2
    kill "$mediamtx_pid"
    wait "$mediamtx_pid" 2>/dev/null || true
    exit 1
  fi
  sleep 0.2
done

# Function to generate black video with same properties as source
generate_black_video() {
  src_file=$1
  black_file=$2
  duration=$3
  
  # Get video properties from source
  resolution=$("$FFMPEG_BIN" -i "$src_file" 2>&1 | grep -oP '\d{3,4}x\d{3,4}' | head -1)
  fps=$("$FFMPEG_BIN" -i "$src_file" 2>&1 | grep -oP '\d+(\.\d+)? fps' | head -1 | grep -oP '[\d.]+')
  
  # Default to 1920x1080 @ 30fps if detection fails
  resolution=${resolution:-1920x1080}
  fps=${fps:-30}
  
  echo "Generating ${duration}s black video at ${resolution} @ ${fps}fps"
  
  "$FFMPEG_BIN" -hide_banner -loglevel error \
    -f lavfi -i "color=c=black:s=${resolution}:r=${fps}:d=${duration}" \
    -c:v libx264 -preset ultrafast -tune stillimage \
    -pix_fmt yuv420p \
    -y "$black_file"
}

# Function to create concatenated video for limited loops with blanks
create_looped_video() {
  src_file=$1
  output_file=$2
  loop_count=$3
  blank_dur=$4
  
  tmpdir=$(mktemp -d)
  concat_list="$tmpdir/concat.txt"
  black_file="$tmpdir/black.mp4"
  
  # Generate black video if needed
  if [ "$blank_dur" -gt 0 ]; then
    generate_black_video "$src_file" "$black_file" "$blank_dur"
  fi
  
  # Build concat list
  i=1
  while [ "$i" -le "$loop_count" ]; do
    echo "file '$src_file'" >> "$concat_list"
    # Add black frames between loops (not after the last one)
    if [ "$blank_dur" -gt 0 ] && [ "$i" -lt "$loop_count" ]; then
      echo "file '$black_file'" >> "$concat_list"
    fi
    i=$((i + 1))
  done
  
  echo "Concat list:"
  cat "$concat_list"
  
  # Create concatenated video
  "$FFMPEG_BIN" -hide_banner -loglevel info \
    -f concat -safe 0 -i "$concat_list" \
    -c copy \
    -y "$output_file"
  
  # Cleanup temp files (keep black_file path for later cleanup)
  rm -f "$concat_list"
  rm -f "$black_file"
  rmdir "$tmpdir" 2>/dev/null || true
}

pids="$pids"
for file in "$@"; do
  [ -f "$file" ] || continue
  filename=$(basename "$file")
  stream_name=${filename%.*}
  
  echo "Starting RTSP stream $stream_name from $file (loop_count=$LOOP_COUNT, blank_duration=${BLANK_DURATION}s)"
  
  # Determine streaming mode
  if [ "$LOOP_COUNT" -eq -1 ]; then
    # Infinite loop mode (original behavior)
    stream_loop_arg="-1"
    input_file="$file"
  elif [ "$LOOP_COUNT" -eq 1 ]; then
    # Single play, no loop
    stream_loop_arg="0"
    input_file="$file"
  else
    # Limited loops with optional blank frames
    if [ "$BLANK_DURATION" -gt 0 ]; then
      # Create concatenated video with blank frames
      looped_file="/tmp/${stream_name}_looped.mp4"
      echo "Creating looped video with ${BLANK_DURATION}s blank frames between ${LOOP_COUNT} loops..."
      create_looped_video "$file" "$looped_file" "$LOOP_COUNT" "$BLANK_DURATION"
      stream_loop_arg="0"  # Play the concatenated file once
      input_file="$looped_file"
    else
      # Simple loop without blanks
      stream_loop_arg="$((LOOP_COUNT - 1))"
      input_file="$file"
    fi
  fi
  
  # -re throttles playback to real-time
  # -stream_loop controls looping behavior
  # -c copy copies audio/video streams without re-encoding
  # -f rtsp publishes to MediaMTX via RTSP protocol
  "$FFMPEG_BIN" \
    -hide_banner \
    -loglevel info \
    -re \
    -stream_loop "$stream_loop_arg" \
    -i "$input_file" \
    -c copy \
    -rtsp_transport tcp \
    -f rtsp \
    "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
  pid=$!
  pids="$pids $pid"
  sleep 0.2
done

cleanup() {
  echo "Stopping RTSP streams"
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
}

trap 'cleanup' INT TERM

status=0
for pid in $pids; do
  if ! wait "$pid"; then
    status=$?
  fi
done

exit $status
