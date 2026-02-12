# RTSP Streamer Integration

The RTSP streamer is now integrated with the parallel-pipeline system.

## Architecture

```
storage/videos/          ← Video files (*.mp4)
       ↓
rtsp-streamer           ← Converts files to RTSP streams
       ↓
rtsp://rtsp-streamer:8554/video1
rtsp://rtsp-streamer:8554/video2
       ↓
parallel-pipeline       ← Consumes RTSP streams
```

## Configuration

### 1. Video Files Location
- **Path**: `order-accuracy/storage/videos/`
- **Files**: `video1.mp4`, `video2.mp4` (or any `.mp4` files)
- **Current setup**: Symlinks to `384-651-925.mp4`

### 2. RTSP Streams
The rtsp-streamer automatically creates streams for each `.mp4` file:
- `video1.mp4` → `rtsp://rtsp-streamer:8554/video1`
- `video2.mp4` → `rtsp://rtsp-streamer:8554/video2`

### 3. Parallel Pipeline Config
Located in: `parallel-pipeline/config/system_config.yaml`

```yaml
rtsp_urls:
  - rtsp://rtsp-streamer:8554/video1  # Station 1
  - rtsp://rtsp-streamer:8554/video2  # Station 2
```

## How It Works

1. **rtsp-streamer** container:
   - Mounts `storage/videos` to `/media`
   - Starts MediaMTX RTSP server on port 8554
   - Uses ffmpeg to stream each `.mp4` file in a loop (`-stream_loop -1`)
   - Stream name = filename without extension

2. **parallel-pipeline** container:
   - Reads RTSP URLs from `system_config.yaml`
   - Each station worker connects to an RTSP stream
   - GStreamer pipeline: `rtspsrc` → `decodebin` → `gvapython` → processing

3. **Continuous Playback**:
   - Videos loop infinitely (no more "exit code 0" restarts)
   - Pipeline stays stable and running
   - Frame processing happens continuously

## Starting the System

### Start with RTSP streams:
```bash
cd /home/intel/sainijit/order-accuracy
docker compose --profile parallel up -d
```

This starts:
- ✅ MinIO (object storage)
- ✅ OVMS VLM (inference server)
- ✅ **rtsp-streamer** (RTSP server)
- ✅ **parallel-pipeline** (processing pipeline)

### Check RTSP streams are running:
```bash
# Check rtsp-streamer logs
docker logs oa_rtsp_streamer

# Test RTSP stream with ffplay (if installed)
ffplay rtsp://localhost:8554/video1

# Or check with VLC
vlc rtsp://localhost:8554/video1
```

### Monitor parallel-pipeline:
```bash
docker logs oa_parallel_pipeline -f
```

## Adding More Videos

To add more videos:

1. Copy `.mp4` files to `storage/videos/`:
   ```bash
   cp my_new_video.mp4 storage/videos/video3.mp4
   ```

2. Restart rtsp-streamer:
   ```bash
   docker compose restart rtsp-streamer
   ```

3. Update `parallel-pipeline/config/system_config.yaml`:
   ```yaml
   rtsp_urls:
     - rtsp://rtsp-streamer:8554/video1
     - rtsp://rtsp-streamer:8554/video2
     - rtsp://rtsp-streamer:8554/video3  # New stream
   ```

4. Restart parallel-pipeline:
   ```bash
   docker compose restart parallel-pipeline
   ```

## Ports

- **8554**: RTSP server port (rtsp-streamer)
  - Access: `rtsp://localhost:8554/<stream_name>`

## Benefits of RTSP Streams

✅ **Continuous playback** - Videos loop infinitely, no restarts
✅ **Real-world simulation** - Mimics live camera feeds
✅ **Multiple consumers** - Same stream can be used by multiple clients
✅ **Low latency** - TCP-based RTSP with `-re` (real-time) throttling
✅ **No re-encoding** - Uses `-c copy` for efficient streaming

## Troubleshooting

### No streams available
```bash
# Check if video files exist
ls -lh storage/videos/

# Check rtsp-streamer logs
docker logs oa_rtsp_streamer

# Verify MediaMTX is running
docker exec oa_rtsp_streamer nc -z 127.0.0.1 8554
```

### Pipeline not connecting
```bash
# Check network connectivity
docker exec oa_parallel_pipeline ping rtsp-streamer

# Check RTSP URLs in config
docker exec oa_parallel_pipeline cat /app/config/system_config.yaml | grep rtsp

# Verify GStreamer pipeline
docker logs oa_parallel_pipeline | grep "Pipeline command"
```

### Videos not looping
The `-stream_loop -1` flag in `start.sh` ensures infinite looping. If videos stop:
```bash
# Check ffmpeg processes
docker exec oa_rtsp_streamer ps aux | grep ffmpeg

# Restart rtsp-streamer
docker compose restart rtsp-streamer
```
