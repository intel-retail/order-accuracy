# frame_pipeline.py
"""
Frame Pipeline - Queue-Based Order Frame Collection with Timestamp Routing
==========================================================================

Design
------
Every GStreamer frame is captured as a FrameMeta(frame_id, capture_ts_ms, jpeg_bytes)
and added to a rolling PRE-BUFFER (deque, never uploaded until OCR fires).

A _frame_ts_map keeps frame_id → capture_ts_ms for the most recent
PRE_BUFFER_FRAMES frames so that when an async OCR result arrives
(carrying the frame_id that was submitted), we can pinpoint exactly
when the order slip became visible.

Order State Machine
-------------------
  IDLE          frames → pre-buffer only
  COLLECTING(X) frames → upload queue for X   (also kept in pre-buffer)

Transitions driven by OCR confirmation (OCR_CONFIRM_COUNT consecutive hits):

  IDLE → COLLECTING(X)  at ocr_fid F:
      Flush ALL pre-buffer frames (no previous order context).

  COLLECTING(X) → COLLECTING(Y)  at ocr_fid F:
      Stop uploading to X + write EOS(X).
      Look up capture_ts_ms of frame F from _frame_ts_map.
      Flush only pre-buffer frames where capture_ts_ms >= ts_of(F)
      → exactly the frames that appeared after the slip changed.

  No OCR for ORDER_IDLE_TIMEOUT_SEC → write EOS, go IDLE, clear pre-buffer.

Upload Worker
-------------
A single daemon thread drains a queue.Queue of
  ("frame",  order_id, frame_id, capture_ts_ms, jpeg_bytes)
  ("eos",    order_id)
and writes to MinIO with a per-upload hard timeout.

Tuneable env-vars
-----------------
  PRE_BUFFER_FRAMES      int  30    rolling frame window
  OCR_SAMPLE_INTERVAL    int   2    submit 1 frame every N to OCR (async mode only)
  OCR_CONFIRM_COUNT      int   3    consecutive same-id hits to confirm
  ORDER_IDLE_TIMEOUT_SEC int   8    seconds with no OCR → finalise
  MINIO_UPLOAD_TIMEOUT   int  10    per-upload wall-clock timeout (s)
  UPLOAD_QUEUE_MAXSIZE   int 500    queue depth (oldest dropped on overflow)
  FRAME_TS_MAP_MAXSIZE   int 200    max entries kept in frame→timestamp map
  STATION_ID             str        injected by docker-compose
  
  OCR_SEQUENTIAL_MODE    bool true  when true, frame waits for OCR result before proceeding
  OCR_SEQUENTIAL_TIMEOUT float 5.0  max seconds to wait per frame in sequential mode
"""

import os
import io
import sys
import time
import queue
import threading
import atexit
import traceback
import uuid
from collections import deque, OrderedDict
from typing import Optional, NamedTuple

import cv2
import numpy as np
from minio import Minio
from config_loader import load_config

# ==========================================================
# FRAME METADATA
# ==========================================================

class FrameMeta(NamedTuple):
    """
    Metadata stored alongside every frame in the pre-buffer.

    frame_id       : monotonic counter assigned by process_frame()
    capture_ts_ms  : wall-clock time (ms since epoch) when the frame
                     was extracted from GStreamer — used to slice frames
                     across order boundaries when OCR result arrives late
    jpeg_bytes     : JPEG-encoded image bytes (ready to upload)
    """
    frame_id:      int
    capture_ts_ms: int
    jpeg_bytes:    bytes


# ==========================================================
# CONFIG
# ==========================================================

cfg           = load_config()
MINIO_CFG     = cfg["minio"]
BUCKETS       = cfg["buckets"]

FRAMES_BUCKET   = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]

try:
    from gstgva import VideoFrame
except Exception:
    VideoFrame = object

MINIO_ENDPOINT = MINIO_CFG["endpoint"]
HAND_LABELS    = {"hand", "person"}

STATION_ID  = os.environ.get("STATION_ID",  "station_unknown")
PIPELINE_ID = os.environ.get("PIPELINE_ID", f"{STATION_ID}_{uuid.uuid4().hex[:8]}")

# Tuneable parameters
PRE_BUFFER_FRAMES      = int(os.environ.get("PRE_BUFFER_FRAMES",      "300"))  # Rolling frame window (300 @ 10fps = 30s)
OCR_SAMPLE_INTERVAL    = int(os.environ.get("OCR_SAMPLE_INTERVAL",     "1"))
ORDER_IDLE_TIMEOUT_SEC = int(os.environ.get("ORDER_IDLE_TIMEOUT_SEC",   "8"))
MINIO_UPLOAD_TIMEOUT   = int(os.environ.get("MINIO_UPLOAD_TIMEOUT",    "10"))
UPLOAD_QUEUE_MAXSIZE   = int(os.environ.get("UPLOAD_QUEUE_MAXSIZE",   "500"))
FRAME_TS_MAP_MAXSIZE   = int(os.environ.get("FRAME_TS_MAP_MAXSIZE",   "200"))

# OCR warmup: number of frames to wait for OCR process to become ready
# Before OCR is ready, frames are buffered but orders are not confirmed
# This prevents losing the first order (e.g., 384) while OCR models load (~30s)
OCR_WARMUP_FRAMES      = int(os.environ.get("OCR_WARMUP_FRAMES",       "2"))

# Sequential OCR mode: when enabled, frame processing BLOCKS until OCR returns
# Default: True (sequential mode) for accuracy. Set to false for high-throughput async mode.
OCR_SEQUENTIAL_MODE    = os.environ.get("OCR_SEQUENTIAL_MODE", "true").lower() in ("true", "1", "yes")
# Increased timeout to 30s to handle slow OCR when queue buffer holds many frames
OCR_SEQUENTIAL_TIMEOUT = float(os.environ.get("OCR_SEQUENTIAL_TIMEOUT", "30.0"))  # max wait per frame (seconds)

# In sequential mode, reduce voting to 1 for immediate response (like old simple version)
# In async mode, keep default of 3 for noise rejection
_default_confirm = "1" if OCR_SEQUENTIAL_MODE else "3"
OCR_CONFIRM_COUNT      = int(os.environ.get("OCR_CONFIRM_COUNT", _default_confirm))

# ==========================================================
# YOLO-GATED OCR FLOW
# ---------------------------------------------------------
# YOLO detects objects on the table every frame as a cheap gating signal.
#   • Table empty  → skip OCR entirely
#   • Table busy (objects detected) for > TABLE_BUSY_THRESHOLD_SEC → trigger OCR once
#   • Once an order_id is confirmed → apply ORDER_LOCK_SEC lock so that the
#     hand leaving the frame (table briefly empty) does NOT prematurely send EOS
#   • Table empty + lock expired → write EOS, hand off to frame-selector / VLM
#   • OCR is NEVER submitted on every frame — only once per retry window until
#     an order is confirmed, eliminating the per-frame OCR bottleneck.
#
# Model: INT8 OpenVINO YOLO11n — same file written by setup_models.sh into the
#        shared models volume (${MODELS_DIR}:/models in order-accuracy container).
# Falls back gracefully to "assume objects present" when model is missing.
# ==========================================================

YOLO_MODEL_PATH          = os.environ.get("YOLO_MODEL_PATH",
                               "/models/yolo11n_int8_openvino_model")
YOLO_TABLE_CONF          = float(os.environ.get("YOLO_TABLE_CONF",          "0.25"))
TABLE_BUSY_THRESHOLD_SEC  = float(os.environ.get("TABLE_BUSY_THRESHOLD_SEC",  "1.5"))
TABLE_EMPTY_THRESHOLD_SEC = float(os.environ.get("TABLE_EMPTY_THRESHOLD_SEC", "2.0"))
ORDER_LOCK_SEC            = float(os.environ.get("ORDER_LOCK_SEC",            "5.0"))
OCR_RETRY_INTERVAL_SEC    = float(os.environ.get("OCR_RETRY_INTERVAL_SEC",    "2.0"))
# After the table looks empty, keep the order open this long so a worker pause
# between items doesn't split one order into two.
ORDER_RESUME_WINDOW_SEC   = float(os.environ.get("ORDER_RESUME_WINDOW_SEC",   "6.0"))
# Hard cap on the pending-close state if the table is busy but OCR never decides.
ORDER_RESUME_MAX_SEC      = float(os.environ.get("ORDER_RESUME_MAX_SEC",      "30.0"))

import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(name)s - %(levelname)s - %(message)s",
)

def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def ts() -> str:
    """ISO-8601 UTC timestamp with milliseconds."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + \
           f".{int(time.time() * 1000) % 1000:03d}Z"

log(f"[CONFIG] station={STATION_ID} pipeline={PIPELINE_ID}")
log(f"[CONFIG] pre_buffer={PRE_BUFFER_FRAMES} ocr_interval={OCR_SAMPLE_INTERVAL} "
    f"confirm={OCR_CONFIRM_COUNT} idle_timeout={ORDER_IDLE_TIMEOUT_SEC}s "
    f"ocr_warmup={OCR_WARMUP_FRAMES}")
log(f"[CONFIG] ocr_sequential_mode={OCR_SEQUENTIAL_MODE} "
    f"sequential_timeout={OCR_SEQUENTIAL_TIMEOUT}s")

# ==========================================================
# OCR SUBPROCESS  (lazy — NOT initialised at module load time)
#
# EasyOCR/PyTorch spawns internal worker threads that conflict with
# GStreamer's native memory management, causing SIGSEGV when both live
# in the same process.  We isolate EasyOCR in a completely separate
# spawned Python process and communicate via multiprocessing Queues.
#
# IMPORTANT: Nothing is created here at module level.
# gvapython loads this module during READY→PAUSED state transition;
# any blocking or subprocess-spawning code at import time causes
# exit code 255.  All setup is deferred to the first process_frame call.
# ==========================================================

# All OCR state starts as None / False; _ensure_ocr_proc() initialises on
# the first frame call.
_ocr_ctx        = None
_ocr_input_q    = None
_ocr_output_q   = None
_ocr_proc       = None
_ocr_proc_ready = False
_ocr_last_submitted: int = -1

# OCR warmup tracking: count valid OCR responses after 'ready' signal
# Order confirmation is blocked until we receive OCR_WARMUP_FRAMES valid responses
# This ensures OCR is truly warmed up and ready to detect order 384 (the first order)
_ocr_warmup_count: int = 0
_ocr_warmed_up: bool = False


def _do_ocr_warmup():
    """
    Warm up OCR models using synthetic frames BEFORE video starts.
    
    This is called at pipeline startup to load EasyOCR models. We send
    dummy black frames to the OCR subprocess, which forces model loading.
    
    With on-demand RTSP streaming, no sync signaling is needed — the RTSP
    stream only starts when GStreamer connects, guaranteeing frame 0 capture.
    """
    global _ocr_warmup_count, _ocr_warmed_up, _ocr_proc_ready
    
    if _ocr_warmed_up:
        return
    
    log(f"[OCR-WARMUP] [{ts()}] Starting synthetic warmup with {OCR_WARMUP_FRAMES} dummy frames...")
    
    # Ensure OCR process is started
    if not _ensure_ocr_proc():
        log(f"[OCR-WARMUP] [{ts()}] Failed to start OCR process")
        return
    
    import numpy as np
    
    # Create a synthetic black frame (640x360 to match OCR resize)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Send warmup frames to OCR - use blocking put since we need all frames sent
    for i in range(OCR_WARMUP_FRAMES):
        warmup_fid = -(i + 1)  # Negative frame IDs for warmup
        try:
            _ocr_input_q.put((warmup_fid, 360, 640, dummy_frame.tobytes()), timeout=30)
            log(f"[OCR-WARMUP] [{ts()}] Sent warmup frame {i+1}/{OCR_WARMUP_FRAMES}")
        except Exception as e:
            log(f"[OCR-WARMUP] [{ts()}] Failed to send warmup frame: {e}")
            break
    
    # Wait for warmup responses (with timeout)
    import time
    warmup_responses = 0
    warmup_start = time.time()
    warmup_timeout = 60  # 60s max for model loading
    
    while warmup_responses < OCR_WARMUP_FRAMES:
        elapsed = time.time() - warmup_start
        if elapsed > warmup_timeout:
            log(f"[OCR-WARMUP] [{ts()}] Timeout after {elapsed:.1f}s - proceeding anyway")
            break
        
        try:
            result = _ocr_output_q.get(timeout=10)
            fid, oid = result
            
            if fid == 'ready':
                _ocr_proc_ready = True
                log(f"[OCR-WARMUP] [{ts()}] OCR subprocess models loaded")
                continue
            
            warmup_responses += 1
            log(f"[OCR-WARMUP] [{ts()}] Warmup response {warmup_responses}/{OCR_WARMUP_FRAMES}")
            
        except Exception:
            # Queue timeout - keep waiting
            log(f"[OCR-WARMUP] [{ts()}] Waiting for OCR response... ({elapsed:.1f}s)")
    
    # Mark as warmed up
    _ocr_proc_ready = True  # OCR is definitely ready after successful warmup
    _ocr_warmed_up = True
    _ocr_warmup_count = OCR_WARMUP_FRAMES
    log(f"[OCR-WARMUP] [{ts()}] OCR warmup complete — ready to process frames")


def _ensure_ocr_proc() -> bool:
    """
    Lazily start the OCR subprocess on the first frame.
    Returns True when the subprocess exists (even if models are still loading).
    Safe to call every frame — no-ops if already started.
    """
    global _ocr_ctx, _ocr_input_q, _ocr_output_q, _ocr_proc

    if _ocr_proc is not None:
        return True  # already started

    try:
        import multiprocessing as _mp

        # In this container ./src is mounted as /app — use it directly.
        # (gvapython does not set __file__ so dynamic detection is unreliable.)
        _src_dir = '/app'
        _cur_pp  = os.environ.get('PYTHONPATH', '')
        if _src_dir not in _cur_pp.split(':'):
            os.environ['PYTHONPATH'] = f"{_src_dir}:{_cur_pp}" if _cur_pp else _src_dir

        # 'spawn' = brand-new Python interpreter with zero GStreamer state
        _ocr_ctx      = _mp.get_context('spawn')
        _ocr_input_q  = _ocr_ctx.Queue(maxsize=10)  # increased for warmup + normal operation
        _ocr_output_q = _ocr_ctx.Queue(maxsize=20)

        from ocr_worker import run_worker
        _ocr_proc = _ocr_ctx.Process(
            target=run_worker,
            args=(_ocr_input_q, _ocr_output_q, '/models/easyocr'),
            daemon=True,
            name=f"ocr-worker-{STATION_ID}",
        )
        _ocr_proc.start()
        log(f"[OCR] subprocess started pid={_ocr_proc.pid} (models loading in background)")
        return True
    except Exception as _e:
        log(f"[OCR] failed to start subprocess: {_e}")
        import traceback as _tb
        log(f"[OCR] traceback: {_tb.format_exc()}")
        return False


def _ocr_submit_frame(frame_id: int, image: np.ndarray) -> None:
    """Send a frame to the OCR subprocess (non-blocking; drops if busy)."""
    global _ocr_last_submitted
    if _ocr_input_q is None or frame_id == _ocr_last_submitted:
        return
    try:
        small = cv2.resize(image, (640, 360))
        h, w  = small.shape[:2]
        _ocr_input_q.put_nowait((frame_id, h, w, small.tobytes()))
        _ocr_last_submitted = frame_id
    except Exception:
        pass  # queue full — skip this frame


def _ocr_submit_frame_blocking(frame_id: int, image: np.ndarray) -> tuple:
    """
    SEQUENTIAL MODE: Send frame to OCR and BLOCK until result arrives.
    Returns (frame_id, order_id_or_None) or (frame_id, None) on timeout.
    """
    global _ocr_last_submitted, _ocr_proc_ready
    if _ocr_input_q is None:
        return (frame_id, None)
    
    try:
        small = cv2.resize(image, (640, 360))
        h, w  = small.shape[:2]
        # Blocking put with timeout
        _ocr_input_q.put((frame_id, h, w, small.tobytes()), timeout=OCR_SEQUENTIAL_TIMEOUT)
        _ocr_last_submitted = frame_id
        log(f"[OCR-SEQ] [{ts()}] submitted frame={frame_id}, waiting for result...")
    except Exception as e:
        log(f"[OCR-SEQ] [{ts()}] failed to submit frame={frame_id}: {e}")
        return (frame_id, None)
    
    # Now BLOCK waiting for the result
    start_wait = time.time()
    while True:
        elapsed = time.time() - start_wait
        if elapsed >= OCR_SEQUENTIAL_TIMEOUT:
            log(f"[OCR-SEQ] [{ts()}] TIMEOUT waiting for frame={frame_id} after {elapsed:.2f}s")
            return (frame_id, None)
        
        try:
            remaining = max(0.1, OCR_SEQUENTIAL_TIMEOUT - elapsed)
            item = _ocr_output_q.get(timeout=remaining)
            
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            
            fid, oid = item
            
            # Handle 'ready' sentinel
            if fid == 'ready':
                if not _ocr_proc_ready:
                    log(f"[OCR-SEQ] [{ts()}] subprocess ready (models loaded)")
                    _ocr_proc_ready = True
                continue
            
            # Check if this is our result
            if fid == frame_id:
                if oid:
                    captured_at = _frame_ts_map.get(fid, 0)
                    cap_str = time.strftime("%H:%M:%S", time.gmtime(captured_at / 1000)) \
                              + f".{captured_at % 1000:03d}" if captured_at else "unknown"
                    log(f"[OCR-SEQ] [{ts()}] frame={fid} captured_at={cap_str} detected order_id={oid}")
                else:
                    log(f"[OCR-SEQ] [{ts()}] frame={fid} no order detected")
                return (fid, oid)
            else:
                # Result for a different frame (shouldn't happen in sequential mode)
                log(f"[OCR-SEQ] [{ts()}] got result for frame={fid} but expected frame={frame_id}")
                # Still return it - might be useful
                if fid > frame_id:
                    return (fid, oid)
                    
        except Exception:
            # Timeout on get - loop will check elapsed time
            pass


def _ocr_drain_results() -> list:
    """
    Drain all pending OCR results from the subprocess output queue.
    Returns list of (frame_id, order_id_or_None) tuples in arrival order.
    frame_id here is the id of the frame that was submitted to OCR —
    we use it to look up capture_ts_ms in _frame_ts_map for timestamp-based
    bucket routing.
    """
    global _ocr_proc_ready
    if _ocr_output_q is None:
        return []
    results = []
    try:
        while True:
            item = _ocr_output_q.get_nowait()
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            fid, oid = item
            if fid == 'ready':          # startup sentinel from ocr_worker
                if not _ocr_proc_ready:
                    log(f"[OCR] [{ts()}] subprocess ready (models loaded)")
                    _ocr_proc_ready = True
                continue
            if oid:
                # Look up when this frame was actually captured
                captured_at = _frame_ts_map.get(fid, 0)
                cap_str = time.strftime("%H:%M:%S", time.gmtime(captured_at / 1000)) \
                          + f".{captured_at % 1000:03d}" if captured_at else "unknown"
                log(f"[OCR] [{ts()}] frame={fid} captured_at={cap_str} detected order_id={oid}")
            results.append((fid, oid))
    except Exception:
        pass
    return results

# ==========================================================
# FRAME TIMESTAMP MAP
#
# Maps frame_id → capture_ts_ms for the most recent FRAME_TS_MAP_MAXSIZE
# frames.  When an async OCR result arrives carrying the frame_id that was
# submitted, we resolve exactly when that frame was captured so we can
# slice the pre-buffer at the right boundary.
# Uses an OrderedDict to evict the oldest entry on overflow (O(1)).
# ==========================================================

_frame_ts_map: OrderedDict = OrderedDict()  # frame_id → capture_ts_ms


def _record_frame_ts(frame_id: int, capture_ts_ms: int) -> None:
    """Insert frame_id → capture_ts_ms; evict oldest when map exceeds max size."""
    _frame_ts_map[frame_id] = capture_ts_ms
    if len(_frame_ts_map) > FRAME_TS_MAP_MAXSIZE:
        _frame_ts_map.popitem(last=False)  # drop oldest


# ==========================================================
# PRE-BUFFER
#
# Rolling deque of FrameMeta(frame_id, capture_ts_ms, jpeg_bytes).
# Nothing is uploaded until OCR confirms an order_id.
# On IDLE→COLLECTING: flush ALL pre-buffer entries.
# On ORDER TRANSITION X→Y: flush only entries where
#   capture_ts_ms >= capture_ts_ms of the OCR-triggering frame
#   (i.e. frames that appeared after the slip changed).
# ==========================================================

_pre_buffer: deque = deque(maxlen=PRE_BUFFER_FRAMES)  # deque[FrameMeta]


# ==========================================================
# MINIO CLIENT
# ==========================================================

_minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_CFG["access_key"],
    secret_key=MINIO_CFG["secret_key"],
    secure=MINIO_CFG.get("secure", False),
)
_buckets_ensured: set = set()


def _ensure_bucket(bucket_name: str) -> None:
    if bucket_name in _buckets_ensured:
        return
    try:
        if not _minio_client.bucket_exists(bucket_name):
            _minio_client.make_bucket(bucket_name)
            log(f"[MINIO] [{ts()}] created bucket: {bucket_name}")
        _buckets_ensured.add(bucket_name)
    except Exception as e:
        log(f"[MINIO] [{ts()}] bucket ensure failed for {bucket_name}: {e}")


# ==========================================================
# UPLOAD QUEUE + WORKER THREAD
#
# Queue items:
#   ("frame", order_id, frame_id, capture_ts_ms, jpeg_bytes)
#   ("eos",   order_id)
# The worker is the ONLY thread that calls MinIO — no GStreamer
# pipeline thread is ever blocked by network I/O.
# ==========================================================

_upload_queue: queue.Queue = queue.Queue(maxsize=UPLOAD_QUEUE_MAXSIZE)


def _upload_worker() -> None:
    """Background daemon thread — drains _upload_queue and writes to MinIO."""
    import concurrent.futures as _cf
    log(f"[UPLOAD-WORKER] [{ts()}] thread started")
    _ensure_bucket(FRAMES_BUCKET)

    while True:
        try:
            item = _upload_queue.get(timeout=2)
        except queue.Empty:
            continue

        try:
            if item[0] == "frame":
                _, order_id, frame_id, capture_ts_ms, jpeg_bytes = item
                cap_str = time.strftime("%H:%M:%S", time.gmtime(capture_ts_ms / 1000)) \
                          + f".{capture_ts_ms % 1000:03d}"
                key = f"{STATION_ID}/{order_id}/frame_{frame_id}.jpg"

                # Burn a visible timestamp overlay so inspection of MinIO frames
                # immediately shows when/which order each frame belongs to.
                try:
                    _img = cv2.imdecode(
                        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if _img is not None:
                        _label = f"{cap_str}  fid={frame_id}  order={order_id}"
                        # black shadow for contrast, then green text on top
                        cv2.putText(_img, _label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(_img, _label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 255, 0), 1, cv2.LINE_AA)
                        _ok, _buf = cv2.imencode(".jpg", _img)
                        if _ok:
                            jpeg_bytes = _buf.tobytes()
                except Exception as _ov_err:
                    log(f"[UPLOAD] [{ts()}] overlay error frame={frame_id}: {_ov_err}")

                t0  = time.time()
                log(f"[UPLOAD] [{ts()}] order_id={order_id} frame={frame_id} "
                    f"captured_at={cap_str} key={key}")
                try:
                    with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                        _ex.submit(
                            _minio_client.put_object,
                            FRAMES_BUCKET, key,
                            io.BytesIO(jpeg_bytes), len(jpeg_bytes),
                            content_type="image/jpeg",
                        ).result(timeout=MINIO_UPLOAD_TIMEOUT)
                    elapsed = (time.time() - t0) * 1000
                    log(f"[UPLOAD] [{ts()}] done order_id={order_id} frame={frame_id} "
                        f"elapsed={elapsed:.1f}ms")
                except _cf.TimeoutError:
                    log(f"[UPLOAD] [{ts()}] TIMEOUT order_id={order_id} "
                        f"frame={frame_id} (>{MINIO_UPLOAD_TIMEOUT}s)")
                except Exception as e:
                    log(f"[UPLOAD] [{ts()}] ERROR order_id={order_id} frame={frame_id}: {e}")

            elif item[0] == "eos":
                _, order_id = item
                key = f"{STATION_ID}/{order_id}/__EOS__"
                log(f"[EOS] [{ts()}] writing order_id={order_id} key={key}")
                t0  = time.time()
                try:
                    with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                        _ex.submit(
                            _minio_client.put_object,
                            FRAMES_BUCKET, key,
                            io.BytesIO(b"finalized"), 9,
                            content_type="text/plain",
                        ).result(timeout=MINIO_UPLOAD_TIMEOUT)
                    elapsed = (time.time() - t0) * 1000
                    log(f"[EOS] [{ts()}] done order_id={order_id} elapsed={elapsed:.1f}ms")
                except _cf.TimeoutError:
                    log(f"[EOS] [{ts()}] TIMEOUT order_id={order_id} (>{MINIO_UPLOAD_TIMEOUT}s)")
                except Exception as e:
                    log(f"[EOS] [{ts()}] ERROR order_id={order_id}: {e}")
        finally:
            _upload_queue.task_done()


_upload_thread = threading.Thread(
    target=_upload_worker,
    daemon=True,
    name=f"upload-worker-{STATION_ID}",
)
_upload_thread.start()
log(f"[CONFIG] [{ts()}] upload worker thread started")


def _enqueue_frame(meta: FrameMeta, order_id: str) -> None:
    """Non-blocking enqueue of a frame upload. Drops and warns if queue is full."""
    try:
        _upload_queue.put_nowait(
            ("frame", order_id, meta.frame_id, meta.capture_ts_ms, meta.jpeg_bytes)
        )
    except queue.Full:
        log(f"[UPLOAD] [{ts()}] QUEUE FULL — dropping frame={meta.frame_id} "
            f"order_id={order_id}")


def _enqueue_eos(order_id: str) -> None:
    try:
        _upload_queue.put_nowait(("eos", order_id))
    except queue.Full:
        log(f"[EOS] [{ts()}] QUEUE FULL — could not enqueue EOS order_id={order_id}")


def _take_pending_close_frames() -> list:
    """Remove staged frames from the rolling pre-buffer and return them."""
    frames = list(_pending_close_frames)
    _pending_close_frames.clear()
    if frames:
        staged_ids = {meta.frame_id for meta in frames}
        retained = [meta for meta in _pre_buffer if meta.frame_id not in staged_ids]
        _pre_buffer.clear()
        _pre_buffer.extend(retained)
    return frames


# ==========================================================
# ORDER STATE MACHINE
# ==========================================================

_active_order_id:          Optional[str] = None   # None = IDLE
_active_order_frame_count: int           = 0
_active_order_start_ts:    float         = 0.0

# OCR vote tracking
_ocr_candidate:      Optional[str] = None
_ocr_vote_count:     int           = 0
_ocr_first_vote_fid: int           = -1  # frame_id of the FIRST detection of current candidate
_last_ocr_hit_ts:          float = 0.0   # wall-clock of most recent OCR result
_last_frame_collected_ts:  float = 0.0   # wall-clock when the most recent frame was enqueued
                                          # used by the idle-timeout safety-net so that the
                                          # timeout only fires when NO frames are being collected
                                          # (OCR is only triggered once per order, so using
                                          # _last_ocr_hit_ts would prematurely end active orders)
_last_order_end_ts_ms: int   = 0         # epoch-ms when last order ended via IDLE TIMEOUT

# YOLO table-presence subprocess state
# YOLO must run in a separate spawned process — same reason as EasyOCR:
# loading PyTorch/ultralytics inside the GStreamer gvapython process causes
# "double free or corruption" memory crashes due to native thread conflicts.
_yolo_ctx         = None
_yolo_input_q     = None
_yolo_output_q    = None
_yolo_proc        = None
_yolo_proc_ready: bool = False
# Latest table state known from YOLO subprocess results.
# Initialised to True ("assume busy") so the system is not locked out before
# the YOLO model finishes loading — same safe-default as the fallback.
_table_has_objects: bool            = True
_table_busy_since:  Optional[float] = None  # wall-clock when objects FIRST detected
_table_empty_since: Optional[float] = None  # wall-clock when table FIRST seen empty
_ocr_last_trigger_ts: float         = 0.0   # wall-clock when OCR was last submitted
_order_lock_until:    float         = 0.0   # table-empty timer suspended until this time
_pending_close_since: Optional[float] = None  # table looked empty; order kept open for possible resume
_pending_close_frames: list = []  # frames held until resume-window OCR resolves the order


def _flush_pre_buffer(order_id: str, since_ts_ms: Optional[int] = None) -> int:
    """
    Flush frames from the pre-buffer into the upload queue for order_id.

    since_ts_ms — if given, only flush frames with capture_ts_ms >= since_ts_ms
                  (used on order transitions / post-IDLE to exclude wrong-order frames).
                  If None, flush ALL frames (used on the very first order start).

    Returns the count of frames flushed.
    """
    flushed = 0
    discarded = 0
    while _pre_buffer:
        meta = _pre_buffer.popleft()
        if since_ts_ms is None or meta.capture_ts_ms >= since_ts_ms:
            _enqueue_frame(meta, order_id)
            flushed += 1
        else:
            # Frame belongs to the previous order — discard it.
            # Do NOT re-append: these were captured before the order change
            # and must not leak into any future order's bucket.
            discarded += 1

    boundary = (
        time.strftime("%H:%M:%S", time.gmtime(since_ts_ms / 1000))
        + f".{since_ts_ms % 1000:03d}"
        if since_ts_ms else "start"
    )
    log(f"[STATE] [{ts()}] flushed {flushed} pre-buffer frames → order_id={order_id} "
        f"(since={boundary} discarded={discarded})")
    return flushed


def _on_ocr_confirmed(confirmed_order_id: str, ocr_fid: int) -> None:
    """
    Called when OCR_CONFIRM_COUNT consecutive detections agree on an order_id.
    ocr_fid is the frame_id of the FIRST detection of the new candidate — used
    to look up the capture timestamp for timestamp-based bucket routing.

    Sets ORDER_LOCK_SEC so that a briefly empty table (hand leaving after
    placing the slip) does NOT prematurely trigger EOS.
    """
    global _active_order_id, _active_order_frame_count, _active_order_start_ts
    global _order_lock_until, _pending_close_since

    # Apply post-confirmation lock: ignore table-empty signals for ORDER_LOCK_SEC.
    _order_lock_until = time.time() + ORDER_LOCK_SEC
    _pending_close_since = None

    # Compute the earliest valid timestamp for this order's frames.
    # We take the MAXIMUM of three candidates so that we never accidentally
    # include frames that belonged to the previous order:
    #   a) _table_busy_since  — when YOLO first saw objects for this order
    #   b) _last_order_end_ts_ms — when the previous order was finalised (EOS sent)
    #   c) capture_ts_ms of the OCR-triggering frame (fallback)
    # Using max() guarantees the boundary is always >= the previous order's end.
    _candidates: list = []
    if _table_busy_since:
        _candidates.append(int(_table_busy_since * 1000))
    if _last_order_end_ts_ms > 0:
        _candidates.append(_last_order_end_ts_ms)
    _ocr_fid_ts = _frame_ts_map.get(ocr_fid)
    if _ocr_fid_ts:
        _candidates.append(_ocr_fid_ts)
    flush_since: Optional[int] = max(_candidates) if _candidates else None

    if _active_order_id is None:
        # ── IDLE → COLLECTING ──────────────────────────────────────────
        log(f"[STATE] [{ts()}] IDLE → COLLECTING order_id={confirmed_order_id} "
            f"lock={ORDER_LOCK_SEC}s (ocr_fid={ocr_fid} flush_since={flush_since})")
        _active_order_id          = confirmed_order_id
        _active_order_frame_count = 0
        _active_order_start_ts    = time.time()
        pending_frames = _take_pending_close_frames()
        for meta in pending_frames:
            if flush_since is None or meta.capture_ts_ms >= flush_since:
                _enqueue_frame(meta, confirmed_order_id)
                _active_order_frame_count += 1
        _flush_pre_buffer(confirmed_order_id, since_ts_ms=flush_since)

    elif confirmed_order_id != _active_order_id:
        # ── ORDER TRANSITION ───────────────────────────────────────────
        prev = _active_order_id
        log(f"[STATE] [{ts()}] ORDER CHANGE {prev} → {confirmed_order_id} "
            f"after {_active_order_frame_count} frames "
            f"lock={ORDER_LOCK_SEC}s (ocr_fid={ocr_fid} boundary_ts={flush_since})")
        pending_frames = _take_pending_close_frames()
        for meta in pending_frames:
            if flush_since is not None and meta.capture_ts_ms < flush_since:
                _enqueue_frame(meta, prev)
                _active_order_frame_count += 1
        _enqueue_eos(prev)

        _active_order_id          = confirmed_order_id
        _active_order_frame_count = 0
        _active_order_start_ts    = time.time()
        for meta in pending_frames:
            if flush_since is None or meta.capture_ts_ms >= flush_since:
                _enqueue_frame(meta, confirmed_order_id)
                _active_order_frame_count += 1
        _flush_pre_buffer(confirmed_order_id, since_ts_ms=flush_since)

    # same order → already collecting; lock is refreshed above regardless


def _check_order_idle_timeout() -> None:
    """
    Safety-net: finalise the active order if ORDER_IDLE_TIMEOUT_SEC seconds pass
    with no frame being collected for it.

    The normal finalisation path is:
        YOLO table-empty ≥ TABLE_EMPTY_THRESHOLD_SEC → _finalize_current_order()

    This only fires when YOLO is unavailable, or the camera freezes, or YOLO
    persistently returns false-positive detections that prevent the empty timer
    from accumulating.

    NOTE: we use _last_frame_collected_ts (not _last_ocr_hit_ts) because OCR is
    intentionally triggered only ONCE per order — basing the timeout on OCR hits
    caused premature EOS 8 seconds after order confirmation even while frames
    were actively being collected.
    """
    if _active_order_id is None or _last_frame_collected_ts == 0.0:
        return

    elapsed = time.time() - _last_frame_collected_ts
    if elapsed >= ORDER_IDLE_TIMEOUT_SEC:
        log(f"[STATE] [{ts()}] IDLE-TIMEOUT safety-net "
            f"({elapsed:.1f}s since last frame collected) — "
            f"finalising order_id={_active_order_id}")
        _finalize_current_order()


# ==========================================================
# YOLO TABLE PRESENCE DETECTION  (subprocess-based)
# ---------------------------------------------------------
# ultralytics/PyTorch must NOT be imported in the GStreamer gvapython process.
# Loading it causes "double free or corruption" crashes from native thread
# conflicts — identical to the EasyOCR issue solved by ocr_worker.py.
# Solution: spawn a dedicated YOLO worker process and communicate via queues.
# ==========================================================

def _ensure_yolo_proc() -> bool:
    """
    Lazily start the YOLO worker subprocess on the first frame.
    Returns True when the subprocess exists (even if model is still loading).
    Safe to call every frame — no-ops if already started.
    """
    global _yolo_ctx, _yolo_input_q, _yolo_output_q, _yolo_proc

    if _yolo_proc is not None:
        return True

    try:
        import multiprocessing as _mp

        _src_dir = '/app'
        _cur_pp  = os.environ.get('PYTHONPATH', '')
        if _src_dir not in _cur_pp.split(':'):
            os.environ['PYTHONPATH'] = f"{_src_dir}:{_cur_pp}" if _cur_pp else _src_dir

        _yolo_ctx     = _mp.get_context('spawn')
        _yolo_input_q = _yolo_ctx.Queue(maxsize=4)   # small: we only need latest frame
        _yolo_output_q = _yolo_ctx.Queue(maxsize=16)

        from yolo_worker import run_worker
        _yolo_proc = _yolo_ctx.Process(
            target=run_worker,
            args=(_yolo_input_q, _yolo_output_q, YOLO_MODEL_PATH, YOLO_TABLE_CONF),
            daemon=True,
            name=f"yolo-worker-{STATION_ID}",
        )
        _yolo_proc.start()
        log(f"[YOLO] [{ts()}] subprocess started pid={_yolo_proc.pid} (model loading...)")
        return True
    except Exception as _e:
        log(f"[YOLO] [{ts()}] failed to start subprocess: {_e}")
        return False


def _yolo_submit_frame(frame_id: int, image: np.ndarray) -> None:
    """Send a frame to the YOLO subprocess (non-blocking; drops if queue full)."""
    if _yolo_input_q is None:
        return
    try:
        small = cv2.resize(image, (320, 320))
        h, w  = small.shape[:2]
        _yolo_input_q.put_nowait((frame_id, h, w, small.tobytes()))
    except Exception:
        pass  # queue full — use cached _table_has_objects state


def _yolo_drain_results() -> None:
    """
    Drain all pending YOLO results from the output queue and update the
    module-level _table_has_objects state variable.
    The pipeline thread reads _table_has_objects; this updates it asynchronously.
    """
    global _yolo_proc_ready, _table_has_objects

    if _yolo_output_q is None:
        return
    try:
        while True:
            item = _yolo_output_q.get_nowait()
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            fid, value = item
            if fid == 'ready':
                if not _yolo_proc_ready:
                    log(f"[YOLO] [{ts()}] subprocess ready (model loaded)")
                    _yolo_proc_ready = True
                continue
            _table_has_objects = bool(value)
    except Exception:
        pass


def _finalize_current_order() -> None:
    """
    Write EOS for the active order and reset all order/table state to IDLE.

    Normal trigger: table detected empty + ORDER_LOCK_SEC has elapsed.
    Safety-net trigger: ORDER_IDLE_TIMEOUT_SEC reached without an OCR hit
                        (in case YOLO table-detection misbehaves).
    """
    global _active_order_id, _active_order_frame_count, _active_order_start_ts
    global _ocr_candidate, _ocr_vote_count, _ocr_first_vote_fid, _last_ocr_hit_ts
    global _last_order_end_ts_ms, _order_lock_until, _ocr_last_trigger_ts
    global _table_busy_since, _table_empty_since, _last_frame_collected_ts
    global _pending_close_since

    _pending_close_since = None
    if _active_order_id is None:
        return

    finalized_order_id = _active_order_id
    log(f"[STATE] [{ts()}] FINALISE order_id={_active_order_id} "
        f"({_active_order_frame_count} frames) — writing EOS")
    _enqueue_eos(_active_order_id)

    _last_order_end_ts_ms     = int(time.time() * 1000)
    _active_order_id          = None
    _active_order_frame_count = 0
    _active_order_start_ts    = 0.0
    _ocr_candidate            = None
    _ocr_vote_count           = 0
    _ocr_first_vote_fid       = -1
    _last_ocr_hit_ts          = 0.0
    _last_frame_collected_ts  = 0.0
    _order_lock_until         = 0.0
    _ocr_last_trigger_ts      = 0.0
    _table_busy_since         = None
    _table_empty_since        = None
    if _pending_close_frames:
        log(f"[STATE] [{ts()}] discarding {len(_pending_close_frames)} unresolved "
            f"pending-close frames after finalizing order_id={finalized_order_id}")
        _pending_close_frames.clear()
    _pre_buffer.clear()


# ==========================================================
# SAFE IMAGE EXTRACTION
# ==========================================================

def safe_get_image(frame) -> Optional[np.ndarray]:
    """Extract a numpy BGR image from a DL Streamer gvapython VideoFrame."""
    try:
        with frame.data() as img:
            if isinstance(img, np.ndarray):
                return img.copy()
            log(f"[PIPELINE] safe_get_image: unexpected type {type(img)}")
    except Exception as e:
        log(f"[PIPELINE] safe_get_image exception: {e}")
    return None


def _process_ocr_result(ocr_fid: int, ocr_oid: Optional[str]) -> None:
    """
    Process a single OCR result through the voting state machine.
    Shared by both async and sequential modes.
    
    WARMUP LOGIC: Orders are NOT confirmed until OCR has been 'warmed up'
    by receiving OCR_WARMUP_FRAMES valid responses. This prevents losing
    order 384 (the first order) while OCR models are still loading.
    """
    global _ocr_candidate, _ocr_vote_count, _ocr_first_vote_fid, _last_ocr_hit_ts
    global _ocr_warmup_count, _ocr_warmed_up, _pending_close_since, _order_lock_until

    # Track OCR warmup - count valid responses until warmed up
    if _ocr_proc_ready and not _ocr_warmed_up:
        _ocr_warmup_count += 1
        if _ocr_warmup_count >= OCR_WARMUP_FRAMES:
            _ocr_warmed_up = True
            log(f"[OCR-WARMUP] [{ts()}] OCR warmed up after {_ocr_warmup_count} responses — "
                f"order detection ENABLED (pre-buffer has {len(_pre_buffer)} frames)")
        else:
            log(f"[OCR-WARMUP] [{ts()}] warmup {_ocr_warmup_count}/{OCR_WARMUP_FRAMES} "
                f"(buffering frames, order detection DISABLED)")
            # During warmup, do NOT process OCR results for order confirmation
            # This ensures all frames stay in pre-buffer until OCR is ready
            return

    if not ocr_oid:
        # OCR returned nothing (no slip visible / blur / transition zone).
        # Do NOT reset the vote streak — a single blank frame must not
        # undo accumulated votes and re-enable uploading to the old order.
        return

    _last_ocr_hit_ts = time.time()

    if ocr_oid == _ocr_candidate:
        _ocr_vote_count += 1
    elif ocr_oid == _active_order_id:
        # OCR confirmed the CURRENT active order — clear any pending
        # candidate so the hold-back guard is released and frames flow
        # normally into the active bucket again.
        if _ocr_candidate is not None:
            log(f"[OCR-VOTE] [{ts()}] candidate={_ocr_candidate} overridden by "
                f"active order re-confirm fid={ocr_fid} — clearing candidate")
        if _pending_close_since is not None:
            log(f"[STATE] [{ts()}] RESUME order_id={_active_order_id} "
                f"(same order re-read {time.time() - _pending_close_since:.1f}s after table empty)")
            pending_frames = _take_pending_close_frames()
            for meta in pending_frames:
                _enqueue_frame(meta, _active_order_id)
                _active_order_frame_count += 1
            if pending_frames:
                _last_frame_collected_ts = time.time()
            _pending_close_since = None
            _order_lock_until = time.time() + ORDER_LOCK_SEC
        _ocr_candidate      = None
        _ocr_vote_count     = 0
        _ocr_first_vote_fid = -1
    else:
        # New candidate — record the frame_id of the FIRST detection.
        # We use this as the pre-buffer boundary so all frames from the
        # moment the new slip was first seen go into the new order's bucket.
        _ocr_candidate      = ocr_oid
        _ocr_vote_count     = 1
        _ocr_first_vote_fid = ocr_fid
        log(f"[OCR-VOTE] [{ts()}] new candidate={ocr_oid} "
            f"(first_fid={ocr_fid} capture_ts={_frame_ts_map.get(ocr_fid, '?')})")

    log(f"[OCR-VOTE] [{ts()}] candidate={_ocr_candidate} "
        f"votes={_ocr_vote_count}/{OCR_CONFIRM_COUNT}")

    if _ocr_vote_count >= OCR_CONFIRM_COUNT:
        if ocr_oid != _active_order_id:
            # Pass first-vote fid so the boundary is at the earliest frame
            # that detected the new order, not the last confirming frame.
            _on_ocr_confirmed(ocr_oid, _ocr_first_vote_fid)
        # Reset votes to avoid repeated transitions
        _ocr_candidate      = None
        _ocr_vote_count     = 0
        _ocr_first_vote_fid = -1


# ==========================================================
# MAIN ENTRYPOINT  (called by gvapython on every frame)
# ==========================================================

_frame_counter: int = 0


def process_frame(frame: "VideoFrame") -> bool:
    """
    GStreamer gvapython entrypoint. Returns True to pass frame downstream.

    Per-frame work (all non-blocking in async mode):
      1. Capture wall-clock timestamp (capture_ts_ms)
      2. Extract image from VideoFrame
      3. JPEG-encode → build FrameMeta(frame_id, capture_ts_ms, jpeg_bytes)
      4. Record frame_id → capture_ts_ms in _frame_ts_map
      5. Add FrameMeta to rolling pre-buffer
      6. OCR processing (mode-dependent):
         - ASYNC:  Drain pending results, submit every N frames
         - SEQUENTIAL: Submit and BLOCK until result arrives
      7. If COLLECTING: enqueue FrameMeta for MinIO upload
      8. Every 10 frames: check ORDER_IDLE_TIMEOUT
    """
    global _frame_counter
    global _ocr_candidate, _ocr_vote_count, _ocr_first_vote_fid, _last_ocr_hit_ts
    global _active_order_id, _active_order_frame_count

    _frame_counter += 1
    capture_ts_ms = int(time.time() * 1000)  # capture wall-clock immediately

    # Start OCR and YOLO subprocesses on very first frame
    if _frame_counter == 1:
        _ensure_ocr_proc()
        _ensure_yolo_proc()

    # Periodic status log
    if _frame_counter <= 3 or _frame_counter % 50 == 0:
        mode_str = "SEQUENTIAL" if OCR_SEQUENTIAL_MODE else "ASYNC"
        log(f"[PIPELINE] [{ts()}] frame={_frame_counter} mode={mode_str} "
            f"active_order={_active_order_id} queue_size={_upload_queue.qsize()}")

    # 1. Extract image
    image = safe_get_image(frame)
    if image is None:
        log(f"[PIPELINE] [{ts()}] frame={_frame_counter} could not extract image — skipping")
        return True

    # 2. JPEG-encode once; bytes reused across pre-buffer + upload queue
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        log(f"[PIPELINE] [{ts()}] frame={_frame_counter} JPEG encode failed — skipping")
        return True
    jpeg_bytes = buf.tobytes()

    # 3. Build FrameMeta with capture timestamp
    meta = FrameMeta(
        frame_id=_frame_counter,
        capture_ts_ms=capture_ts_ms,
        jpeg_bytes=jpeg_bytes,
    )

    # 4. Record timestamp so OCR results can resolve their boundary
    _record_frame_ts(_frame_counter, capture_ts_ms)

    # 5. Always feed the rolling pre-buffer
    _pre_buffer.append(meta)
    staged_during_pending_close = _pending_close_since is not None
    if staged_during_pending_close:
        _pending_close_frames.append(meta)

    # 6. YOLO table-presence gate + YOLO-triggered OCR
    global _table_busy_since, _table_empty_since, _ocr_last_trigger_ts, _order_lock_until

    # Submit this frame to the YOLO subprocess (non-blocking; drops if queue full).
    # Drain any results that arrived since the last frame → updates _table_has_objects.
    _yolo_submit_frame(_frame_counter, image)
    _yolo_drain_results()

    if _table_has_objects:
        # ── Table is BUSY ───────────────────────────────────────────────
        # Any object resets the empty timer silently — no log until confirmed.
        _table_empty_since = None

        if _table_busy_since is None:
            _table_busy_since = time.time()
            log(f"[TABLE] [{ts()}] frame={_frame_counter} objects detected — table BUSY")

        busy_sec     = time.time() - _table_busy_since
        ocr_retry_ok = (time.time() - _ocr_last_trigger_ts) >= OCR_RETRY_INTERVAL_SEC

        if (_pending_close_since is not None
                and time.time() - _pending_close_since >= ORDER_RESUME_MAX_SEC):
            log(f"[STATE] [{ts()}] pending close exceeded {ORDER_RESUME_MAX_SEC}s "
                f"without OCR decision — finalising order_id={_active_order_id}")
            _finalize_current_order()

        # Fire OCR when the table has been busy long enough and no order is
        # active, or the active order is pending close (same order resumes,
        # a different order triggers ORDER CHANGE).
        awaiting_order = _active_order_id is None or _pending_close_since is not None
        if awaiting_order and busy_sec >= TABLE_BUSY_THRESHOLD_SEC and ocr_retry_ok:
            log(f"[TABLE] [{ts()}] frame={_frame_counter} busy {busy_sec:.1f}s — "
                f"triggering OCR (retry_interval={OCR_RETRY_INTERVAL_SEC}s)")
            _ocr_last_trigger_ts = time.time()
            ocr_fid, ocr_oid = _ocr_submit_frame_blocking(_frame_counter, image)
            _process_ocr_result(ocr_fid, ocr_oid)

    else:
        # ── Table appears empty ─────────────────────────────────────────
        # Do NOT reset _table_busy_since here.  YOLO can flicker (one frame
        # with no detections between frames that do have objects).  Resetting
        # the busy timer on any single empty frame means the 1.5s threshold
        # can never accumulate when YOLO oscillates, so OCR is never triggered.
        # _table_busy_since is only cleared in _finalize_current_order() once
        # the table is confirmed empty for TABLE_EMPTY_THRESHOLD_SEC.

        # ORDER_LOCK_SEC takes priority: while the lock is active (hand just
        # placed the slip and left), suppress the empty timer entirely.
        # No log — waiting silently until lock expires.
        lock_active = time.time() <= _order_lock_until

        if not lock_active:
            # Lock expired (or no order active) — run the 2-second empty timer.
            if _table_empty_since is None:
                _table_empty_since = time.time()
                # Intentionally no log here — only log once confirmed empty.
            else:
                empty_sec = time.time() - _table_empty_since
                if empty_sec >= TABLE_EMPTY_THRESHOLD_SEC:
                    if _active_order_id is not None:
                        if _pending_close_since is None:
                            log(f"[TABLE] [{ts()}] table empty for {empty_sec:.1f}s — "
                                f"order_id={_active_order_id} pending close "
                                f"(resume window {ORDER_RESUME_WINDOW_SEC}s)")
                            _pending_close_since = time.time()
                            _table_busy_since = None
                            _ocr_candidate = None
                            _ocr_vote_count = 0
                            _ocr_first_vote_fid = -1
                        elif time.time() - _pending_close_since >= ORDER_RESUME_WINDOW_SEC:
                            log(f"[TABLE] [{ts()}] table confirmed empty for "
                                f"{empty_sec:.1f}s — finalising order_id={_active_order_id}")
                            _finalize_current_order()
                    else:
                        # No active order — reset so the timer doesn't keep firing.
                        _table_empty_since = None
        else:
            # Lock active — silently discard any empty-timer state.
            _table_empty_since = None

    # 7. Collect frames for the active order.
    # Frames are sent to MinIO for every frame while an order is open —
    # regardless of live table state.  The order remains open until
    # _finalize_current_order() is called (table confirmed empty for 2s).
    if not staged_during_pending_close:
        if _pending_close_since is not None:
            _pending_close_frames.append(meta)
        elif _active_order_id is not None:
            _enqueue_frame(meta, _active_order_id)
            _active_order_frame_count += 1
            _last_frame_collected_ts = time.time()  # keep idle-timeout safety-net alive
            if _active_order_frame_count % 50 == 0:
                log(f"[COLLECT] [{ts()}] order_id={_active_order_id} "
                    f"frames_collected={_active_order_frame_count} "
                    f"frame={_frame_counter}")

    # 8. Safety-net idle-timeout (fires if YOLO table detection misbehaves)
    if _frame_counter % 10 == 0:
        _check_order_idle_timeout()

    return True


# ==========================================================
# SHUTDOWN HANDLER — flush last active order on video-file EOF
# ==========================================================
# When gst-launch-1.0 exits normally after a file-source EOS the Python
# interpreter starts tearing down.  _check_order_idle_timeout() is only
# called from inside process_frame(), which stops being called once
# GStreamer sends EOS, so the last active order (e.g. 925) **never** gets
# its per-order __EOS__ marker written through the normal path.
#
# This atexit handler fires while daemon threads are still alive:
#   1. Enqueues EOS for the currently active order (if any).
#   2. Blocks on _upload_queue.join() until the upload worker has written
#      every remaining frame AND the EOS marker to MinIO.
#
# After this handler completes the pipeline_runner (the parent process) can
# safely write the station-level __EOS__ marker, knowing all frames and
# per-order EOS markers are already in MinIO.
# ==========================================================

def _pipeline_atexit_handler() -> None:
    """Flush in-flight uploads and write per-order EOS for the last order."""
    global _active_order_id, _active_order_frame_count

    log(f"[ATEXIT] [{ts()}] Pipeline shutdown — active_order={_active_order_id} "
        f"queue_size={_upload_queue.qsize()}")

    # 1. Enqueue EOS for the active order so frame-selector can finalise it
    if _active_order_id is not None:
        log(f"[ATEXIT] [{ts()}] Enqueueing EOS for last active order: {_active_order_id}")
        _enqueue_eos(_active_order_id)
        _active_order_id = None
        _active_order_frame_count = 0

    # 2. Drain the upload queue (blocks until the upload-worker thread is idle)
    # The upload-worker is a daemon thread and is still alive during atexit.
    drain_timeout = 60  # seconds
    deadline = time.time() + drain_timeout
    log(f"[ATEXIT] [{ts()}] Waiting up to {drain_timeout}s for upload queue to drain "
        f"(current size={_upload_queue.qsize()})...")
    try:
        _upload_queue.join()  # blocks until all put()s have a matching task_done()
        log(f"[ATEXIT] [{ts()}] Upload queue drained successfully")
    except Exception as e:
        log(f"[ATEXIT] [{ts()}] Error waiting for upload queue: {e}")


atexit.register(_pipeline_atexit_handler)
log(f"[INIT] [{ts()}] Registered pipeline shutdown atexit handler")

# ==========================================================
# SIGNAL HANDLER — flush last active order on SIGABRT/SIGTERM
# ==========================================================
# When gst-launch-1.0 exits after a video file ends, the process often
# receives SIGABRT (exit code 134) from OpenCV TLS teardown.  Python's
# atexit handlers do NOT run when the process is killed by a signal.
#
# This signal handler ensures the final order's __EOS__ marker is written
# to MinIO even on abnormal termination, so the frame-selector / station
# worker can detect the completed order.
# ==========================================================

import signal as _signal

_shutdown_handler_called = False

def _pipeline_signal_handler(signum, frame):
    """Flush pending orders on SIGTERM/SIGABRT before the process dies."""
    global _shutdown_handler_called
    if _shutdown_handler_called:
        return  # Prevent double-execution if both signal + atexit fire
    _shutdown_handler_called = True

    sig_name = _signal.Signals(signum).name if hasattr(_signal, 'Signals') else str(signum)
    log(f"[SIGNAL] [{ts()}] Received {sig_name} — flushing active order before exit")
    try:
        _pipeline_atexit_handler()
    except Exception as e:
        log(f"[SIGNAL] [{ts()}] Error during signal shutdown: {e}")

    # Re-raise the signal with default handler to get correct exit code
    _signal.signal(signum, _signal.SIG_DFL)
    os.kill(os.getpid(), signum)

# Wrap atexit handler to skip if signal handler already ran
_original_atexit_handler = _pipeline_atexit_handler

def _guarded_atexit_handler():
    if not _shutdown_handler_called:
        _original_atexit_handler()

# Re-register guarded version
atexit.unregister(_pipeline_atexit_handler)
atexit.register(_guarded_atexit_handler)

# Register signal handlers
try:
    _signal.signal(_signal.SIGTERM, _pipeline_signal_handler)
    log(f"[INIT] [{ts()}] Registered SIGTERM signal handler")
except Exception as e:
    log(f"[INIT] [{ts()}] Could not register SIGTERM handler: {e}")

try:
    _signal.signal(_signal.SIGABRT, _pipeline_signal_handler)
    log(f"[INIT] [{ts()}] Registered SIGABRT signal handler")
except Exception as e:
    log(f"[INIT] [{ts()}] Could not register SIGABRT handler: {e}")

# ==========================================================
# MODULE INITIALIZATION - OCR WARMUP
# ==========================================================
# Run OCR warmup at module load time (BEFORE any frames arrive).
# If OCR_PRE_WARMED is set by station_worker, skip warmup — models
# are already in OS page cache and PyTorch JIT is warmed.
# The OCR subprocess will be spawned lazily on first frame
# and will load models much faster (~2-5s vs ~21s).
# ==========================================================
if os.environ.get('OCR_PRE_WARMED') == '1':
    log(f"[INIT] [{ts()}] OCR pre-warmed by station_worker — skipping module-level warmup")
    _ensure_ocr_proc()  # Start subprocess (non-blocking), models load from hot page cache
    _ocr_warmed_up = True  # Skip dummy frame warmup
    _ocr_warmup_count = OCR_WARMUP_FRAMES
else:
    log(f"[INIT] [{ts()}] Starting OCR warmup at module load...")
    _do_ocr_warmup()
log(f"[INIT] [{ts()}] Module initialization complete - ready for frames")
