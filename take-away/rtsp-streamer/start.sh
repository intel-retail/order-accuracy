#!/bin/sh
set -eu

MEDIA_DIR=${MEDIA_DIR:-/media}
RTSP_PORT=${RTSP_PORT:-8554}
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
MEDIAMTX_BIN=${MEDIAMTX_BIN:-/opt/rtsp-streamer/mediamtx}

# Number of streams to create (one per worker/station)
NUM_STREAMS=${WORKERS:-2}

# Stream loop configuration
# -1 = loop infinitely, 0 = no loop (stream ends when video ends), N = loop N times
STREAM_LOOP=${STREAM_LOOP:-0}

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

# Startup delay to allow GStreamer pipelines to connect
# This ensures pipelines are ready before video starts playing
STARTUP_DELAY=${STARTUP_DELAY:-0}
if [ "$STARTUP_DELAY" -gt 0 ]; then
  echo "Waiting ${STARTUP_DELAY}s for pipelines to connect..."
  sleep "$STARTUP_DELAY"
  echo "Delay complete, starting streams"
fi

# Create streams for each station
# If RTSP_STREAMS is set (comma-separated), use those names
# Otherwise create station_1, station_2, etc.
if [ -n "${RTSP_STREAMS:-}" ]; then
  # Use custom stream names from RTSP_STREAMS
  stream_names=$(echo "$RTSP_STREAMS" | tr ',' ' ')
  i=0
  for stream_name in $stream_names; do
    i=$((i + 1))
    if [ $i -gt $NUM_STREAMS ]; then
      break
    fi
    echo "Starting RTSP stream: $stream_name (from $source_file)"
    "$FFMPEG_BIN" \
      -hide_banner \
      -loglevel warning \
      -re \
      -stream_loop $STREAM_LOOP \
      -i "$source_file" \
      -c copy \
      -rtsp_transport tcp \
      -f rtsp \
      "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
    pid=$!
    pids="$pids $pid"
    # Minimal delay between stream starts (was 0.5s, caused timing desync)
    sleep 0.1
  done
else
  # Create station_N streams - start all with minimal delay for synchronized playback
  i=1
  while [ $i -le $NUM_STREAMS ]; do
    stream_name="station_${i}"
    echo "Starting RTSP stream: $stream_name (from $source_file)"
    "$FFMPEG_BIN" \
      -hide_banner \
      -loglevel warning \
      -re \
      -stream_loop $STREAM_LOOP \
      -i "$source_file" \
      -c copy \
      -rtsp_transport tcp \
      -f rtsp \
      "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
    pid=$!
    pids="$pids $pid"
    i=$((i + 1))
    # Minimal delay between stream starts (was 0.5s, caused timing desync)
    sleep 0.1
  done
fi

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
