#!/bin/sh
set -eu

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
MEDIAMTX_BIN=${MEDIAMTX_BIN:-/opt/rtsp-streamer/mediamtx}

# Number of streams to create (one per worker/station)
NUM_STREAMS=${WORKERS:-2}

# Loop configuration
# LOOP_COUNT: number of times to play the video
#   -1 = infinite loop
#    1 = play once
#    2 = play twice, etc.
LOOP_COUNT=${LOOP_COUNT:--1}

# LOOP_WARMUP: seconds of black frames BEFORE each loop iteration (except first)
# This gives the pipeline time to "reset" and ensures the first order in each loop
# gets proper frames instead of transition frames at the loop boundary.
LOOP_WARMUP=${LOOP_WARMUP:-5}

# Source video file (use first .mp4 found, or specified by RTSP_STREAM_NAME)
SOURCE_VIDEO=${RTSP_STREAM_NAME:-}

if [ ! -d "$MEDIA_DIR" ]; then
  echo "Media directory $MEDIA_DIR does not exist" >&2
  exit 1
fi

if [ ! -x "$MEDIAMTX_BIN" ]; then
  echo "mediamtx binary $MEDIAMTX_BIN not found or not executable" >&2
  exit 1
fi

# Find the source video file
if [ -n "$SOURCE_VIDEO" ] && [ -f "$MEDIA_DIR/${SOURCE_VIDEO}.mp4" ]; then
  source_file="$MEDIA_DIR/${SOURCE_VIDEO}.mp4"
else
  # Use first .mp4 file found
  set -- "$MEDIA_DIR"/*.mp4
  if [ ! -e "$1" ]; then
    echo "No .mp4 files found in $MEDIA_DIR" >&2
    exit 1
  fi
  source_file="$1"
fi

echo "Source video: $source_file"
echo "Number of streams to create: $NUM_STREAMS"

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

echo "RTSP server ready on port $RTSP_PORT"

# ==============================================================================
# Pipeline Synchronization - Two-Phase Commit (2PC) Pattern
# ==============================================================================
# Fault-tolerant synchronization between RTSP streamer and pipeline workers.
# This prevents the race condition where video starts before pipelines connect,
# causing the first order (e.g., order 384) to be missed.
#
# Two-Phase Commit Protocol:
#   Phase 1 - PREPARE: Pipeline signals ready at /sync/prepare/station_N
#             (after GStreamer connected AND OCR model warmed up)
#   Phase 2 - COMMIT:  Streamer signals at /sync/commit/station_N
#             (after streams are active and producing frames)
#
# Flow:
#   1. Streamer waits for all PREPARE signals (workers ready)
#   2. Streamer starts RTSP streams
#   3. Streamer sends COMMIT signals (workers can begin processing)
#   4. Workers unblock and start consuming video frames
#
# Modes:
#   SYNC_MODE=signal - Full 2PC with PREPARE/COMMIT handshake
#   SYNC_MODE=delay  - Use simple STARTUP_DELAY timer (legacy)
#   SYNC_MODE=none   - Start immediately (for debugging)
# ==============================================================================

SYNC_MODE=${SYNC_MODE:-signal}
SYNC_DIR=${SYNC_DIR:-/sync/ready}
OCR_READY_DIR=${OCR_READY_DIR:-/sync/ocr_ready}
PREPARE_DIR=${PREPARE_DIR:-/sync/prepare}
COMMIT_DIR=${COMMIT_DIR:-/sync/commit}
SYNC_TIMEOUT=${SYNC_TIMEOUT:-180}  # Max seconds to wait for all workers (increased for multi-worker CPU contention)
STARTUP_DELAY=${STARTUP_DELAY:-10}  # Fallback delay if sync mode is 'delay'

wait_for_ready_signals() {
  local num_workers=$1
  local timeout=$2
  local sync_dir=$3
  
  echo "Waiting for $num_workers pipeline(s) to signal ready..."
  echo "Sync directory: $sync_dir"
  echo "Timeout: ${timeout}s"
  
  # NOTE: Cleanup moved to init_sync_dirs() to avoid race condition
  # where we delete ready files workers already created
  mkdir -p "$sync_dir"
  
  local start_time=$(date +%s)
  local ready_count=0
  
  while [ "$ready_count" -lt "$num_workers" ]; do
    # Count ready files
    ready_count=0
    for i in $(seq 1 "$num_workers"); do
      if [ -f "$sync_dir/station_$i" ]; then
        ready_count=$((ready_count + 1))
      fi
    done
    
    # Check timeout
    local elapsed=$(($(date +%s) - start_time))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "WARNING: Sync timeout after ${elapsed}s - only $ready_count/$num_workers pipelines ready"
      echo "Proceeding anyway (some frames may be missed)"
      return 1
    fi
    
    # Log progress periodically
    if [ $((elapsed % 5)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
      echo "Waiting for pipelines: $ready_count/$num_workers ready (${elapsed}s elapsed)"
    fi
    
    sleep 0.5
  done
  
  local elapsed=$(($(date +%s) - start_time))
  echo "All $num_workers pipeline(s) ready after ${elapsed}s - starting video streams"
  return 0
}

# Function to wait for OCR ready signals and restart streams
wait_for_ocr_ready_and_restart() {
  local num_workers=$1
  local timeout=$2
  local ocr_ready_dir=$3
  
  echo ""
  echo "=== Phase 2: Waiting for OCR warmup completion ==="
  echo "Watching for OCR ready signals in: $ocr_ready_dir"
  echo "This ensures order 384 (first order) is not missed due to OCR loading time"
  
  # NOTE: Cleanup moved to init_sync_dirs() to avoid race condition
  mkdir -p "$ocr_ready_dir"
  
  local start_time=$(date +%s)
  local ready_count=0
  
  while [ "$ready_count" -lt "$num_workers" ]; do
    # Count ready files
    ready_count=0
    for i in $(seq 1 "$num_workers"); do
      if [ -f "$ocr_ready_dir/station_$i" ]; then
        ready_count=$((ready_count + 1))
      fi
    done
    
    # Check timeout
    local elapsed=$(($(date +%s) - start_time))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "WARNING: OCR ready timeout after ${elapsed}s - only $ready_count/$num_workers OCR ready"
      echo "Proceeding without restart (first order may be missed)"
      return 1
    fi
    
    # Log progress periodically
    if [ $((elapsed % 10)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
      echo "Waiting for OCR: $ready_count/$num_workers ready (${elapsed}s elapsed)"
    fi
    
    sleep 1
  done
  
  local elapsed=$(($(date +%s) - start_time))
  echo ""
  echo "=== All $num_workers OCR pipeline(s) warmed up after ${elapsed}s ==="
  echo "Starting streams now - order 384 will be captured!"
  return 0
}

# Wait for PREPARE signals from all stations (2PC Phase 1)
# This is the modern replacement for wait_for_ocr_ready_and_restart
wait_for_prepare_signals() {
  local num_workers=$1
  local timeout=$2
  local prepare_dir=$3
  
  echo ""
  echo "=== 2PC Phase 1: Waiting for PREPARE signals ==="
  echo "Directory: $prepare_dir"
  echo "Workers expected: $num_workers"
  
  # NOTE: Do NOT clear PREPARE files here!
  # Workers may have already signaled PREPARE while we were waiting for READY signals.
  # The READY/PREPARE/COMMIT directories are cleared at container startup via init_sync_dirs().
  mkdir -p "$prepare_dir"
  mkdir -p "$COMMIT_DIR"
  
  local start_time=$(date +%s)
  local ready_count=0
  
  while [ "$ready_count" -lt "$num_workers" ]; do
    # Count PREPARE files
    ready_count=0
    for i in $(seq 1 "$num_workers"); do
      if [ -f "$prepare_dir/station_$i" ]; then
        ready_count=$((ready_count + 1))
      fi
    done
    
    # Check timeout
    local elapsed=$(($(date +%s) - start_time))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "WARNING: 2PC PREPARE timeout after ${elapsed}s - only $ready_count/$num_workers ready"
      echo "Proceeding anyway (some workers may not be synchronized)"
      return 1
    fi
    
    # Log progress periodically
    if [ $((elapsed % 10)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
      echo "[2PC-PREPARE] Waiting: $ready_count/$num_workers ready (${elapsed}s elapsed)"
    fi
    
    sleep 0.5
  done
  
  local elapsed=$(($(date +%s) - start_time))
  echo ""
  echo "=== 2PC Phase 1 COMPLETE: All $num_workers workers PREPARED after ${elapsed}s ==="
  return 0
}

# Send COMMIT signals to all stations (2PC Phase 2)
# This tells workers they can start consuming video frames
send_commit_signals() {
  local num_workers=$1
  local commit_dir=$2
  
  echo ""
  echo "=== 2PC Phase 2: Sending COMMIT signals ==="
  echo "Directory: $commit_dir"
  
  mkdir -p "$commit_dir"
  
  local commit_time=$(date +%s)
  
  for i in $(seq 1 "$num_workers"); do
    # Create COMMIT file with timestamp
    echo "$commit_time" > "$commit_dir/station_$i"
    echo "[2PC-COMMIT] Signaled station_$i"
  done
  
  echo ""
  echo "=== 2PC Phase 2 COMPLETE: All $num_workers COMMIT signals sent ==="
  echo "=== Workers can now begin processing video frames ==="
}

# Initialize sync directories - cleanup any stale files from previous runs
# CRITICAL: Must run BEFORE oa_service starts, or we'll delete files workers create
# Uses a marker file to avoid re-init on container restart (which would delete worker's files)
init_sync_dirs() {
  local init_marker="/sync/.rtsp_init_done"
  
  # Skip if already initialized (container restart scenario)
  if [ -f "$init_marker" ]; then
    echo "Sync directories already initialized (container restart detected)"
    return 0
  fi
  
  echo "Creating sync directories (volume is clean from docker compose down -v)..."
  
  # Only create directories - do NOT delete files!
  # Workers may have already written ready files before this runs.
  # The volume is guaranteed clean by docker compose down -v between iterations.
  mkdir -p "$SYNC_DIR"
  mkdir -p "$PREPARE_DIR"
  mkdir -p "$COMMIT_DIR"
  mkdir -p "$OCR_READY_DIR"
  
  # Mark as initialized to prevent re-cleaning on restart
  touch "$init_marker"
  
  echo "Sync directories initialized"
}

# Initialize sync dirs IMMEDIATELY (before RTSP server starts, before workers signal)
init_sync_dirs

case "$SYNC_MODE" in
  signal)
    # Two-Phase Commit synchronization protocol
    echo ""
    echo "========================================================"
    echo "  2PC SYNCHRONIZATION PROTOCOL"
    echo "========================================================"
    
    # Legacy Phase 1: Wait for pipelines to connect (quick)
    # This ensures GStreamer pipelines have connected to RTSP
    wait_for_ready_signals "$NUM_STREAMS" "$SYNC_TIMEOUT" "$SYNC_DIR"
    
    # 2PC Phase 1: Wait for PREPARE signals
    # Workers signal PREPARE after OCR warmup is complete
    # This replaces the old OCR ready wait
    # First try new 2PC PREPARE signals, fall back to legacy OCR_READY
    if ! wait_for_prepare_signals "$NUM_STREAMS" "$SYNC_TIMEOUT" "$PREPARE_DIR"; then
      echo "Trying legacy OCR ready signals..."
      wait_for_ocr_ready_and_restart "$NUM_STREAMS" "$SYNC_TIMEOUT" "$OCR_READY_DIR"
    fi
    ;;
  delay)
    if [ "$STARTUP_DELAY" -gt 0 ]; then
      echo "Using legacy delay mode: waiting ${STARTUP_DELAY}s for pipelines..."
      sleep "$STARTUP_DELAY"
      echo "Delay complete, starting streams"
    fi
    ;;
  none)
    echo "Sync disabled - starting streams immediately"
    ;;
  *)
    echo "Unknown SYNC_MODE: $SYNC_MODE - using 'signal' mode"
    wait_for_ready_signals "$NUM_STREAMS" "$SYNC_TIMEOUT" "$SYNC_DIR"
    wait_for_ocr_ready_and_restart "$NUM_STREAMS" "$SYNC_TIMEOUT" "$OCR_READY_DIR"
    ;;
esac

# Get video properties for black frame generation
get_video_props() {
  video_file="$1"
  
  # Default values
  VIDEO_WIDTH=1920
  VIDEO_HEIGHT=1080
  VIDEO_FPS=30
  
  # Try to extract actual values using ffprobe if available
  if command -v ffprobe >/dev/null 2>&1; then
    width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$video_file" 2>/dev/null)
    height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$video_file" 2>/dev/null)
    fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$video_file" 2>/dev/null)
    
    if [ -n "$width" ] && [ "$width" -gt 0 ] 2>/dev/null; then
      VIDEO_WIDTH="$width"
    fi
    if [ -n "$height" ] && [ "$height" -gt 0 ] 2>/dev/null; then
      VIDEO_HEIGHT="$height"
    fi
    if [ -n "$fps" ]; then
      # fps might be a fraction like "30/1", extract numerator
      VIDEO_FPS=$(echo "$fps" | cut -d'/' -f1)
    fi
  else
    # Fallback to parsing ffmpeg output
    props=$("$FFMPEG_BIN" -i "$video_file" 2>&1 | grep -E "Video:" | head -1)
    
    # Extract resolution (looking for pattern like "1920x1080" or "640x480")
    if echo "$props" | grep -qoE ", [0-9]+x[0-9]+"; then
      res=$(echo "$props" | sed -n 's/.*[^0-9]\([0-9]\{3,4\}x[0-9]\{3,4\}\).*/\1/p')
      if [ -n "$res" ]; then
        w=$(echo "$res" | cut -dx -f1)
        h=$(echo "$res" | cut -dx -f2)
        if [ "$w" -gt 100 ] 2>/dev/null && [ "$h" -gt 100 ] 2>/dev/null; then
          VIDEO_WIDTH="$w"
          VIDEO_HEIGHT="$h"
        fi
      fi
    fi
  fi
  
  echo "Video properties: ${VIDEO_WIDTH}x${VIDEO_HEIGHT} @ ${VIDEO_FPS}fps"
}

# Create a looped video file with black warmup segments between loops
# This keeps the stream continuous (no disconnects) while adding warmup periods
create_looped_video() {
  video_file="$1"
  loop_count="$2"
  warmup_sec="$3"
  output_file="$4"
  
  echo "Creating looped video with ${warmup_sec}s warmup between ${loop_count} loops..."
  
  # Get video properties
  get_video_props "$video_file"
  
  # Create concat list file
  concat_list="/tmp/concat_list_$$.txt"
  
  # Use different variable name to avoid conflict with outer loop's 'i'
  loop_idx=1
  while [ $loop_idx -le $loop_count ]; do
    # Add black warmup BEFORE each loop (except the first one)
    if [ $loop_idx -gt 1 ] && [ "$warmup_sec" -gt 0 ]; then
      echo "file '/tmp/black_warmup.mp4'" >> "$concat_list"
    fi
    echo "file '$video_file'" >> "$concat_list"
    loop_idx=$((loop_idx + 1))
  done
  
  # Generate black warmup video if needed
  if [ "$warmup_sec" -gt 0 ] && [ "$loop_count" -gt 1 ]; then
    echo "Generating ${warmup_sec}s black warmup segment..."
    
    # Get codec info from source video for better compatibility
    video_codec=$("$FFMPEG_BIN" -i "$video_file" 2>&1 | grep -E "Stream.*Video" | grep -oE "h264|hevc|h265|mpeg4" | head -1)
    video_codec=${video_codec:-h264}
    
    "$FFMPEG_BIN" -y \
      -f lavfi \
      -i "color=c=black:s=${VIDEO_WIDTH}x${VIDEO_HEIGHT}:r=${VIDEO_FPS}:d=${warmup_sec}" \
      -f lavfi \
      -i "anullsrc=r=48000:cl=stereo" \
      -t "$warmup_sec" \
      -c:v libx264 -preset ultrafast -tune zerolatency \
      -profile:v baseline -level 3.0 \
      -c:a aac -b:a 128k \
      -pix_fmt yuv420p \
      /tmp/black_warmup.mp4 2>/dev/null || \
    "$FFMPEG_BIN" -y \
      -f lavfi \
      -i "color=c=black:s=${VIDEO_WIDTH}x${VIDEO_HEIGHT}:r=${VIDEO_FPS}:d=${warmup_sec}" \
      -t "$warmup_sec" \
      -c:v libx264 -preset ultrafast -tune zerolatency \
      -pix_fmt yuv420p \
      /tmp/black_warmup.mp4 2>/dev/null
    
    if [ ! -f /tmp/black_warmup.mp4 ]; then
      echo "WARNING: Failed to create black warmup segment, will proceed without warmup"
      rm -f "$concat_list"
      return 1
    fi
    echo "Black warmup segment created"
  fi
  
  # Concatenate using concat demuxer
  # Try -c copy first (fast), fall back to re-encoding if it fails
  echo "Concatenating videos..."
  if ! "$FFMPEG_BIN" -y \
    -f concat \
    -safe 0 \
    -i "$concat_list" \
    -c copy \
    "$output_file" 2>/dev/null; then
    echo "Fast concat failed, trying with re-encoding..."
    if ! "$FFMPEG_BIN" -y \
      -f concat \
      -safe 0 \
      -i "$concat_list" \
      -c:v libx264 -preset fast -crf 18 \
      -c:a aac -b:a 128k \
      "$output_file" 2>/dev/null; then
      echo "WARNING: concat failed, will proceed without warmup"
      rm -f "$concat_list"
      return 1
    fi
  fi
  
  rm -f "$concat_list"
  
  if [ ! -f "$output_file" ]; then
    echo "WARNING: Failed to create looped video, will proceed without warmup"
    return 1
  fi
  
  echo "Looped video created: $output_file"
  return 0
}

# Function to stream video with loop warmup support
stream_video() {
  stream_name="$1"
  
  echo "Starting RTSP stream: $stream_name (LOOP_COUNT=$LOOP_COUNT, LOOP_WARMUP=${LOOP_WARMUP}s)"
  
  if [ "$LOOP_COUNT" = "-1" ]; then
    # Infinite loop mode - use simple -stream_loop
    # Note: No warmup in infinite mode (would require complex filter)
    # LOW-LATENCY FLAGS for faster client connection
    echo "Using infinite loop mode (no warmup, low-latency)"
    "$FFMPEG_BIN" \
      -hide_banner \
      -loglevel warning \
      -fflags nobuffer \
      -flags low_delay \
      -re \
      -ss 0 \
      -stream_loop -1 \
      -i "$source_file" \
      -c copy \
      -rtsp_transport tcp \
      -flush_packets 1 \
      -f rtsp \
      "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
  elif [ "$LOOP_COUNT" = "1" ]; then
    # Single play - force seek to beginning with -ss 0
    # LOW-LATENCY FLAGS + VIDEO LEADER:
    # - Add 30 seconds of black frames at start to give GStreamer time to connect
    # - This accounts for: initial connection (~2s) + potential restart (~2s) + OCR warmup (~10s) + buffer
    # - This ensures order 384 (at second 2) is captured even with pipeline restart
    # - fflags nobuffer: No output buffering
    # - flags low_delay: Enable low-delay mode
    # - flush_packets 1: Flush packets immediately
    echo "Using single play mode with 30s leader (low-latency)"
    
    # Get video dimensions for black frame generation
    video_dim=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$source_file" 2>/dev/null)
    video_width=$(echo "$video_dim" | cut -d',' -f1)
    video_height=$(echo "$video_dim" | cut -d',' -f2)
    video_width=${video_width:-1280}
    video_height=${video_height:-720}
    
    "$FFMPEG_BIN" \
      -hide_banner \
      -loglevel warning \
      -f lavfi -i "color=c=black:s=${video_width}x${video_height}:r=30:d=30" \
      -re \
      -ss 0 \
      -i "$source_file" \
      -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[outv]" \
      -map "[outv]" \
      -c:v libx264 -preset ultrafast -tune zerolatency \
      -fflags nobuffer \
      -flags low_delay \
      -rtsp_transport tcp \
      -flush_packets 1 \
      -f rtsp \
      "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
  else
    # Finite loop with warmup - try to create pre-concatenated video
    looped_file="/tmp/looped_video_${stream_name}.mp4"
    
    if create_looped_video "$source_file" "$LOOP_COUNT" "$LOOP_WARMUP" "$looped_file" && [ -f "$looped_file" ]; then
      echo "Streaming pre-looped video with warmup"
      "$FFMPEG_BIN" \
        -hide_banner \
        -loglevel warning \
        -re \
        -i "$looped_file" \
        -c copy \
        -rtsp_transport tcp \
        -f rtsp \
        "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
    else
      # Fallback to simple -stream_loop without warmup
      echo "Falling back to simple loop mode (no warmup)"
      stream_loop_val=$((LOOP_COUNT - 1))
      "$FFMPEG_BIN" \
        -hide_banner \
        -loglevel warning \
        -re \
        -stream_loop $stream_loop_val \
        -i "$source_file" \
        -c copy \
        -rtsp_transport tcp \
        -f rtsp \
        "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
    fi
  fi
}

# Track stream PIDs globally for restart capability
stream_pids=""

# Function to start all video streams
start_all_streams() {
  local reason=${1:-"initial"}
  echo "Starting all $NUM_STREAMS RTSP streams ($reason)..."
  
  stream_pids=""
  
  if [ -n "${RTSP_STREAMS:-}" ]; then
    # Use custom stream names from RTSP_STREAMS
    stream_names=$(echo "$RTSP_STREAMS" | tr ',' ' ')
    i=0
    for stream_name in $stream_names; do
      i=$((i + 1))
      if [ $i -gt $NUM_STREAMS ]; then
        break
      fi
      stream_video "$stream_name"
      pid=$!
      stream_pids="$stream_pids $pid"
      pids="$pids $pid"
      sleep 0.1
    done
  else
    # Create station_N streams
    i=1
    while [ $i -le $NUM_STREAMS ]; do
      stream_name="station_${i}"
      stream_video "$stream_name"
      pid=$!
      stream_pids="$stream_pids $pid"
      pids="$pids $pid"
      i=$((i + 1))
      sleep 0.1
    done
  fi
}

# Create streams for each station
# If RTSP_STREAMS is set (comma-separated), use those names
# Otherwise create station_1, station_2, etc.
start_all_streams "initial"

# Signal to workers that streams are now available
# This file tells GStreamer pipelines it's safe to connect
touch /sync/streams_started
echo "=== Signaled /sync/streams_started - pipelines can now connect ==="

# ========================================================================
# 2PC Phase 2: Send COMMIT signals to all workers
# ========================================================================
# This tells workers that streams are ready and they can start consuming
# video frames. Workers are blocked waiting for COMMIT after PREPARE.
if [ "$SYNC_MODE" = "signal" ]; then
  # Brief delay to ensure ffmpeg has started outputting frames
  # This prevents workers from trying to read before frames are available
  sleep 1
  send_commit_signals "$NUM_STREAMS" "$COMMIT_DIR"
fi
# ========================================================================

echo "All $NUM_STREAMS RTSP streams started successfully"
echo "Available streams:"
i=1
while [ $i -le $NUM_STREAMS ]; do
  if [ -n "${RTSP_STREAMS:-}" ]; then
    stream_name=$(echo "$RTSP_STREAMS" | cut -d',' -f$i)
  else
    stream_name="station_${i}"
  fi
  echo "  - rtsp://rtsp-streamer:${RTSP_PORT}/${stream_name}"
  i=$((i + 1))
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
