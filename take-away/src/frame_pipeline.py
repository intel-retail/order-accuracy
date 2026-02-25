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
PRE_BUFFER_FRAMES      = int(os.environ.get("PRE_BUFFER_FRAMES",      "30"))
OCR_SAMPLE_INTERVAL    = int(os.environ.get("OCR_SAMPLE_INTERVAL",     "1"))
ORDER_IDLE_TIMEOUT_SEC = int(os.environ.get("ORDER_IDLE_TIMEOUT_SEC",   "8"))
MINIO_UPLOAD_TIMEOUT   = int(os.environ.get("MINIO_UPLOAD_TIMEOUT",    "10"))
UPLOAD_QUEUE_MAXSIZE   = int(os.environ.get("UPLOAD_QUEUE_MAXSIZE",   "500"))
FRAME_TS_MAP_MAXSIZE   = int(os.environ.get("FRAME_TS_MAP_MAXSIZE",   "200"))

# Sequential OCR mode: when enabled, frame processing BLOCKS until OCR returns
# Default: True (sequential mode) for accuracy. Set to false for high-throughput async mode.
OCR_SEQUENTIAL_MODE    = os.environ.get("OCR_SEQUENTIAL_MODE", "true").lower() in ("true", "1", "yes")
OCR_SEQUENTIAL_TIMEOUT = float(os.environ.get("OCR_SEQUENTIAL_TIMEOUT", "5.0"))  # max wait per frame (seconds)

# In sequential mode, reduce voting to 1 for immediate response (like old simple version)
# In async mode, keep default of 3 for noise rejection
_default_confirm = "1" if OCR_SEQUENTIAL_MODE else "3"
OCR_CONFIRM_COUNT      = int(os.environ.get("OCR_CONFIRM_COUNT", _default_confirm))

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
    f"confirm={OCR_CONFIRM_COUNT} idle_timeout={ORDER_IDLE_TIMEOUT_SEC}s")
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
        _ocr_input_q  = _ocr_ctx.Queue(maxsize=2)   # drop frames when busy
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
_last_ocr_hit_ts:    float         = 0.0   # wall-clock of most recent OCR result
_last_order_end_ts_ms: int         = 0     # epoch-ms when last order ended via IDLE TIMEOUT


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
    """
    global _active_order_id, _active_order_frame_count, _active_order_start_ts

    # Resolve the capture timestamp of the OCR-triggering frame
    ocr_frame_ts: Optional[int] = _frame_ts_map.get(ocr_fid)

    if _active_order_id is None:
        # ── IDLE → COLLECTING ──────────────────────────────────────────
        log(f"[STATE] [{ts()}] IDLE → COLLECTING order_id={confirmed_order_id} "
            f"(ocr_fid={ocr_fid} ocr_frame_ts={ocr_frame_ts})")
        _active_order_id          = confirmed_order_id
        _active_order_frame_count = 0
        _active_order_start_ts    = time.time()
        # If a previous order existed (_last_order_end_ts_ms > 0), the pre-buffer
        # may contain frames that OCR labelled as a DIFFERENT order that never
        # reached confirm-count (e.g. 925 frames landing in 384's bucket).
        # Guard against this by using the first-detection timestamp as the
        # boundary — only flush frames captured from when the new slip was
        # first seen.  For the very first order ever (_last_order_end_ts_ms==0)
        # flush everything so we capture the pre-buffer look-back.
        flush_since = ocr_frame_ts if _last_order_end_ts_ms > 0 else None
        _flush_pre_buffer(confirmed_order_id, since_ts_ms=flush_since)

    elif confirmed_order_id != _active_order_id:
        # ── ORDER TRANSITION ───────────────────────────────────────────
        prev = _active_order_id
        log(f"[STATE] [{ts()}] ORDER CHANGE {prev} → {confirmed_order_id} "
            f"after {_active_order_frame_count} frames "
            f"(ocr_fid={ocr_fid} boundary_ts={ocr_frame_ts})")
        _enqueue_eos(prev)

        _active_order_id          = confirmed_order_id
        _active_order_frame_count = 0
        _active_order_start_ts    = time.time()
        # Only flush pre-buffer frames captured AFTER the OCR boundary
        # — frames before it already belonged to the previous order.
        _flush_pre_buffer(confirmed_order_id, since_ts_ms=ocr_frame_ts)

    # same order → already collecting, nothing to do


def _check_order_idle_timeout() -> None:
    """Finalise current order if no OCR result has arrived for ORDER_IDLE_TIMEOUT_SEC."""
    global _active_order_id, _active_order_frame_count, _active_order_start_ts
    global _ocr_candidate, _ocr_vote_count, _ocr_first_vote_fid, _last_order_end_ts_ms

    if _active_order_id is None or _last_ocr_hit_ts == 0.0:
        return

    elapsed = time.time() - _last_ocr_hit_ts
    if elapsed >= ORDER_IDLE_TIMEOUT_SEC:
        log(f"[STATE] [{ts()}] IDLE TIMEOUT ({elapsed:.1f}s no OCR) — "
            f"finalising order_id={_active_order_id} "
            f"({_active_order_frame_count} frames uploaded)")
        _enqueue_eos(_active_order_id)
        _last_order_end_ts_ms     = int(time.time() * 1000)  # mark end of this order
        _active_order_id          = None
        _active_order_frame_count = 0
        _active_order_start_ts    = 0.0
        _ocr_candidate            = None
        _ocr_vote_count           = 0
        _ocr_first_vote_fid       = -1
        _pre_buffer.clear()  # stale frames must not leak into next order


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
    """
    global _ocr_candidate, _ocr_vote_count, _ocr_first_vote_fid, _last_ocr_hit_ts

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

    # Start OCR subprocess on very first frame
    if _frame_counter == 1:
        _ensure_ocr_proc()

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

    # 6. OCR processing — mode-dependent
    if OCR_SEQUENTIAL_MODE:
        # ═══════════════════════════════════════════════════════════════
        # SEQUENTIAL MODE: Submit EVERY frame and BLOCK until OCR returns
        # Frame processing halts until we get the OCR result
        # ═══════════════════════════════════════════════════════════════
        ocr_fid, ocr_oid = _ocr_submit_frame_blocking(_frame_counter, image)
        _process_ocr_result(ocr_fid, ocr_oid)
    else:
        # ═══════════════════════════════════════════════════════════════
        # ASYNC MODE: Non-blocking drain + periodic submission
        # Frames flow continuously; OCR runs in parallel
        # ═══════════════════════════════════════════════════════════════
        # Drain OCR results FIRST so that the upload decision below (step 7)
        # always uses the freshest candidate/vote state.
        for ocr_fid, ocr_oid in _ocr_drain_results():
            _process_ocr_result(ocr_fid, ocr_oid)

        # Submit current frame to OCR subprocess every N frames
        if _frame_counter % OCR_SAMPLE_INTERVAL == 0:
            _ocr_submit_frame(_frame_counter, image)

    # 7. If collecting, route frame directly to upload queue.
    # IMPORTANT: stop live-uploading to the current order once a competing
    # OCR candidate appears.  Those transition frames will be held in the
    # pre-buffer and flushed to the NEW order when the vote is confirmed.
    # Without this guard, frames showing order Y would be uploaded to order
    # X's bucket during the vote-accumulation window.
    if _active_order_id is not None:
        candidate_is_different = (
            _ocr_candidate is not None
            and _ocr_candidate != _active_order_id
        )
        if not candidate_is_different:
            _enqueue_frame(meta, _active_order_id)
            _active_order_frame_count += 1
        else:
            log(f"[PIPELINE] [{ts()}] frame={_frame_counter} holding back from "
                f"order_id={_active_order_id} (candidate={_ocr_candidate} "
                f"votes={_ocr_vote_count}/{OCR_CONFIRM_COUNT})")

    # 8. Idle-timeout check every 10 frames
    if _frame_counter % 10 == 0:
        _check_order_idle_timeout()

    return True
