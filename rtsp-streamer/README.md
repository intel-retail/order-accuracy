# RTSP Streamer Integration

## Overview

The RTSP streamer service converts MP4 video files into continuous RTSP streams using MediaMTX and FFmpeg. This allows the parallel-pipeline to consume video data as if it were coming from real cameras.

## Architecture

```
┌─────────────────────┐
│  Video Files        │
│  (storage/videos/)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RTSP Streamer      │
│  - MediaMTX Server  │
│  - FFmpeg Streams   │
│  Port: 8554         │
└──────────┬──────────┘
           │
           ▼ rtsp://rtsp-streamer:8554/video1
┌─────────────────────┐
│  Parallel Pipeline  │
│  - Station Workers  │
│  - GStreamer        │
└─────────────────────┘
```

## Features

✅ **Automatic Stream Creation** - Converts all .mp4 files in `/media` to RTSP streams  
✅ **Infinite Loop** - Streams loop continuously (`-stream_loop -1`)  
✅ **Real-time Playback** - Maintains original video timing (`-re`)  
✅ **No Re-encoding** - Uses copy codec for efficiency (`-c copy`)  
✅ **TCP Transport** - Reliable RTSP over TCP  
✅ **Health Checks** - Monitors RTSP server availability  

## Stream Naming

Streams are named after the video filename without extension:

| File              | Stream URL                              |
|-------------------|-----------------------------------------|
| video1.mp4        | rtsp://rtsp-streamer:8554/video1        |
| video2.mp4        | rtsp://rtsp-streamer:8554/video2        |
| 384-651-925.mp4   | rtsp://rtsp-streamer:8554/384-651-925   |

## Configuration

### Docker Compose

The RTSP streamer is integrated into docker-compose.yaml:

```yaml
rtsp-streamer:
  build: ./rtsp-streamer
  container_name: oa_rtsp_streamer
  volumes:
    - ./storage/videos:/media:ro
  environment:
    - MEDIA_DIR=/media
    - RTSP_PORT=8554
  ports:
    - "8554:8554"
  networks:
    - order-accuracy-net
  profiles:
    - parallel
```

### Parallel Pipeline Configuration

Update `parallel-pipeline/config/system_config.yaml`:

```yaml
rtsp_urls:
  - rtsp://rtsp-streamer:8554/video1  # Station 1
  - rtsp://rtsp-streamer:8554/video2  # Station 2
```

## Usage

### Start All Services

```bash
cd /home/intel/sainijit/order-accuracy

# Build and start with RTSP streamer
docker compose --profile parallel --profile ovms up --build -d

# Check status
docker compose ps
```

### Start RTSP Streamer Only

```bash
# Build
docker compose build rtsp-streamer

# Start
docker compose --profile parallel up rtsp-streamer -d

# View logs
docker logs oa_rtsp_streamer -f
```

### Test RTSP Streams

```bash
# List available streams
ffprobe rtsp://localhost:8554/video1

# Play stream with FFmpeg
ffplay rtsp://localhost:8554/video1

# Play stream with VLC
vlc rtsp://localhost:8554/video2
```

## Troubleshooting

### Check RTSP Server Health

```bash
# Check if port 8554 is listening
docker exec oa_rtsp_streamer nc -z 127.0.0.1 8554

# Check MediaMTX logs
docker exec oa_rtsp_streamer cat /tmp/mediamtx.log
```

### Check Active Streams

```bash
# View FFmpeg processes
docker exec oa_rtsp_streamer ps aux | grep ffmpeg

# Count active streams
docker exec oa_rtsp_streamer pgrep ffmpeg | wc -l
```

### Common Issues

#### No .mp4 files found
```
No .mp4 files found in /media
```
**Solution**: Ensure video files exist in `storage/videos/`

#### RTSP server failed to start
```
RTSP server failed to start on port 8554
```
**Solution**: Check if port 8554 is already in use

#### Connection timeout
```
rtspsrc location=rtsp://rtsp-streamer:8554/video1 ! ... timeout
```
**Solution**: 
1. Check RTSP streamer is running: `docker ps | grep rtsp`
2. Check network connectivity: `docker exec oa_parallel_pipeline ping rtsp-streamer`
3. Verify stream exists: `docker logs oa_rtsp_streamer | grep "Starting RTSP"`

## Benefits Over File-based Sources

| Feature               | File Source        | RTSP Stream       |
|-----------------------|-------------------|-------------------|
| Continuous playback   | ❌ Restarts needed | ✅ Infinite loop   |
| Realistic simulation  | ❌ No              | ✅ Like real camera |
| Frame timing          | ❌ Fast            | ✅ Real-time       |
| Multi-station sharing | ❌ File locking    | ✅ Multiple clients |
| Network simulation    | ❌ No              | ✅ TCP/RTSP        |

## Performance

- **Latency**: ~200ms (configurable via `latency=` parameter)
- **CPU Usage**: ~2-5% per stream (no re-encoding)
- **Memory**: ~50MB per stream
- **Network**: Minimal (containers share bridge network)

## Advanced Configuration

### Adjust Stream Quality

Edit `rtsp-streamer/start.sh` to add encoding parameters:

```bash
"$FFMPEG_BIN" \
  -re -stream_loop -1 \
  -i "$file" \
  -c:v libx264 -preset ultrafast -b:v 2M \  # Add encoding
  -c:a aac -b:a 128k \
  -rtsp_transport tcp \
  -f rtsp \
  "rtsp://127.0.0.1:${RTSP_PORT}/${stream_name}" &
```

### Change RTSP Port

```yaml
environment:
  - RTSP_PORT=8555  # Custom port
ports:
  - "8555:8555"
```

### Add Authentication

Edit `mediamtx.yml` to enable authentication (see MediaMTX documentation).

## References

- [MediaMTX Documentation](https://github.com/bluenviron/mediamtx)
- [FFmpeg RTSP Output](https://ffmpeg.org/ffmpeg-formats.html#rtsp)
- [GStreamer RTSP Source](https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html)
