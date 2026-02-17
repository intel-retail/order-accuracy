# frame_to_minio.py
import os
import io
import time
import traceback
import cv2
import numpy as np
import threading
import queue
import atexit
from collections import deque
from minio import Minio
from minio.commonconfig import CopySource
from config_loader import load_config
cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]

FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]

# Import OrderTracker for robust frame accumulation
from core.order_tracker import OrderTracker, OrderTrackerConfig

# Try to import VideoFrame for typing only (gvapython provides it)
try:
    from gstgva import VideoFrame
except Exception:
    VideoFrame = object

# ====== Configuration (from docker-compose envs) ======
MINIO_ENDPOINT = MINIO["endpoint"]

HAND_LABELS = {"hand", "person"}

# Station ID from environment variable (set by worker process)
STATION_ID = os.environ.get('STATION_ID', 'station_unknown')

# Use stderr for logging (stdout not captured by GStreamer pipeline)
import sys
def log(msg):
    print(msg, file=sys.stderr, flush=True)

# ====== Async OCR Configuration ======
OCR_QUEUE_SIZE = 50  # Max frames waiting for OCR
OCR_RESULT_CACHE_SIZE = 100  # Recent OCR results cache
OCR_SAMPLE_INTERVAL = 3  # Run OCR every Nth frame (skip frames for speed)
PENDING_ORDER_PREFIX = "pending"  # Prefix for frames awaiting OCR

# ====== Async OCR System ======
class AsyncOCRProcessor:
    """
    Asynchronous OCR processor that runs OCR in a separate thread.
    
    This prevents OCR from blocking the GStreamer pipeline, allowing
    continuous frame capture while OCR processes in the background.
    
    Features:
    - Non-blocking frame submission
    - Frame sampling (only OCR every Nth frame)
    - Recent result caching for order ID consistency
    - Thread-safe state management with single lock
    - LRU-based cache eviction
    - Graceful shutdown
    """
    
    def __init__(self, station_id: str, sample_interval: int = 3):
        self.station_id = station_id
        self.sample_interval = sample_interval
        
        # Frame counter for sampling
        self._frame_counter = 0
        
        # Queue for frames to process
        self._ocr_queue = queue.Queue(maxsize=OCR_QUEUE_SIZE)
        
        # Single lock for all state (prevents race conditions)
        self._state_lock = threading.RLock()
        
        # Cache recent OCR results (frame_id -> (order_id, timestamp))
        self._result_cache = {}
        
        # Current detected order (sticky until new order detected)
        self._current_order_id = None
        self._last_detection_time = 0.0
        
        # Worker thread control
        self._running = False
        self._stop_event = threading.Event()
        self._worker_thread = None
        
        # Initialize EasyOCR (lazy, in worker thread)
        self._reader = None
        
        log(f"[ASYNC-OCR] Initialized for {station_id}, sample_interval={sample_interval}")
    
    def start(self):
        """Start the OCR worker thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._ocr_worker,
            daemon=False,  # Non-daemon so we can drain queue on exit
            name=f"ocr-worker-{self.station_id}"
        )
        self._worker_thread.start()
        log(f"[ASYNC-OCR] Worker thread started")
    
    def stop(self, drain_queue: bool = False, max_drain_time: float = 30.0):
        """Stop the OCR worker thread gracefully.
        
        Args:
            drain_queue: If True, wait for queued items to be processed before stopping
            max_drain_time: Maximum seconds to wait for queue to drain
        """
        if drain_queue and self._running:
            # Wait for queue to drain
            log(f"[ASYNC-OCR] Draining queue ({self._ocr_queue.qsize()} items)...")
            start_time = time.time()
            while not self._ocr_queue.empty() and (time.time() - start_time) < max_drain_time:
                remaining = self._ocr_queue.qsize()
                if remaining > 0 and (time.time() - start_time) % 5 < 1:
                    log(f"[ASYNC-OCR] Queue drain: {remaining} items remaining")
                time.sleep(0.5)
            if not self._ocr_queue.empty():
                log(f"[ASYNC-OCR] Queue drain timeout, {self._ocr_queue.qsize()} items remaining")
            else:
                log(f"[ASYNC-OCR] Queue drained successfully")
        
        self._running = False
        self._stop_event.set()
        if self._worker_thread:
            # Put poison pill to unblock queue
            try:
                self._ocr_queue.put(None, timeout=1)
            except queue.Full:
                pass
            self._worker_thread.join(timeout=5)
        log(f"[ASYNC-OCR] Worker thread stopped")
    
    def is_running(self) -> bool:
        """Check if OCR processor is running (for health checks)."""
        return bool(self._running and self._worker_thread and self._worker_thread.is_alive())
    
    def submit_frame(self, frame_id: int, image: np.ndarray) -> "Optional[str]":
        """
        Submit a frame for OCR processing.
        
        Returns:
            Current best guess for order ID (may be from cache or previous detection)
        """
        self._frame_counter += 1
        
        # Only submit every Nth frame for OCR (sampling)
        if self._frame_counter % self.sample_interval == 0:
            try:
                # Copy image since original may be invalidated
                image_copy = image.copy()
                self._ocr_queue.put_nowait((frame_id, image_copy))
            except queue.Full:
                # Queue full, skip this frame
                log(f"[ASYNC-OCR] Queue full, skipping frame {frame_id}")
        
        # Return current order ID atomically
        with self._state_lock:
            return self._current_order_id
    
    def get_current_order_id(self) -> "Optional[str]":
        """Get the most recently detected order ID."""
        with self._state_lock:
            return self._current_order_id
    
    def get_result(self, frame_id: int) -> "Optional[str]":
        """Get OCR result for a specific frame (if available)."""
        with self._state_lock:
            entry = self._result_cache.get(frame_id)
            return entry[0] if entry else None
    
    def _cleanup_cache_lru(self):
        """LRU-based cache cleanup - must be called with lock held."""
        if len(self._result_cache) > OCR_RESULT_CACHE_SIZE:
            # Sort by timestamp (oldest first) and remove half
            sorted_entries = sorted(
                self._result_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            entries_to_remove = len(self._result_cache) - (OCR_RESULT_CACHE_SIZE // 2)
            for frame_id, _ in sorted_entries[:entries_to_remove]:
                del self._result_cache[frame_id]
            log(f"[ASYNC-OCR] Cache cleanup: removed {entries_to_remove} entries")
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader (called in worker thread)."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ['en'],
                gpu=False,
                verbose=False,
                quantize=True,
                model_storage_directory='/models/easyocr'
            )
            log(f"[ASYNC-OCR] EasyOCR initialized in worker thread")
    
    def _ocr_worker(self):
        """Worker thread that processes OCR queue."""
        log(f"[ASYNC-OCR] Worker thread starting...")
        
        # Initialize EasyOCR in this thread
        self._init_easyocr()
        
        while self._running and not self._stop_event.is_set():
            try:
                item = self._ocr_queue.get(timeout=1)
                
                if item is None:
                    # Poison pill - shutdown
                    break
                
                frame_id, image = item
                
                # Run OCR
                start_time = time.time()
                order_id = self._run_ocr(image)
                elapsed = time.time() - start_time
                
                if order_id:
                    # Atomically update both order ID and cache
                    with self._state_lock:
                        if order_id != self._current_order_id:
                            log(f"[ASYNC-OCR] Order changed: {self._current_order_id} -> {order_id}")
                        self._current_order_id = order_id
                        self._last_detection_time = time.time()
                        
                        # Cache result with timestamp for LRU eviction
                        self._result_cache[frame_id] = (order_id, time.time())
                        self._cleanup_cache_lru()
                    
                    log(f"[ASYNC-OCR] Frame {frame_id}: detected order_id={order_id} ({elapsed:.2f}s)")
                
                self._ocr_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                log(f"[ASYNC-OCR] Worker error: {e}")
                traceback.print_exc(file=sys.stderr)
    
    def _preprocess_roi(self, roi):
        """Lightweight preprocessing for OCR."""
        roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _normalize_order_id(self, raw_id: str) -> str:
        """Normalize OCR-detected order ID."""
        if not raw_id:
            return raw_id
        if len(raw_id) == 3:
            return raw_id
        if len(raw_id) >= 4:
            return raw_id[:3]
        return raw_id
    
    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image and extract order ID."""
        thresh = self._preprocess_roi(image)
        
        results = self._reader.readtext(
            thresh,
            detail=1,
            paragraph=False,
            width_ths=0.5,
            text_threshold=0.7,
            low_text=0.5,
            link_threshold=0.3,
            canvas_size=1280,
            mag_ratio=1.5,
            allowlist='0123456789#'
        )
        
        candidates = []
        for (bbox, text, conf) in results:
            raw = text.replace(" ", "")
            if "#" in raw:
                after = raw.split("#", 1)[1]
                digits = ""
                for c in after:
                    if c.isdigit():
                        digits += c
                    else:
                        break
                if digits:
                    normalized = self._normalize_order_id(digits)
                    candidates.append((normalized, conf, digits))
        
        if not candidates:
            return None
        
        # Prefer 3-digit numbers
        three_digit = [(n, c, o) for n, c, o in candidates if len(n) == 3]
        if three_digit:
            best = max(three_digit, key=lambda x: x[1])
            return best[0]
        
        best = max(candidates, key=lambda x: x[1])
        return best[0]


# ====== Global Async OCR Processor ======
_async_ocr = AsyncOCRProcessor(STATION_ID, OCR_SAMPLE_INTERVAL)
_async_ocr.start()

def _cleanup_on_exit():
    """Cleanup handler to drain OCR queue before process exit."""
    global _async_ocr
    if _async_ocr:
        log("[CLEANUP] Process exiting, draining OCR queue...")
        _async_ocr.stop(drain_queue=True, max_drain_time=60.0)
        log("[CLEANUP] OCR cleanup complete")

atexit.register(_cleanup_on_exit)


# ====== MinIO client ======
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=MINIO.get("secure", False),
)

if not client.bucket_exists(FRAMES_BUCKET):
    try:
        client.make_bucket(FRAMES_BUCKET)
        log(f"[frame_to_minio] Created bucket: {FRAMES_BUCKET}")
    except Exception as e:
        log(f"[frame_to_minio] Warning - cannot create bucket: {e}")


# ====== Pipeline State (Encapsulated) ======
from typing import Optional, Dict, Any

class PipelineState:
    """
    Encapsulated state for the frame pipeline.
    
    Prevents issues with module-level global state:
    - Non-deterministic behavior across pipeline restarts
    - Memory leaks from accumulated data
    - Race conditions on module reload
    """
    
    def __init__(self, station_id: str, config: dict):
        self.station_id: str = station_id
        self.frame_counter: int = 0
        self.current_order_id: Optional[str] = None
        self.previous_order_id: Optional[str] = None  # For grace period frame collection
        self.order_frame_count: int = 0
        self.last_order_time: float = time.time()
        # Limit pending frames to prevent memory leak
        self.pending_frames: deque = deque(maxlen=50)  # Reduced from 100
        
        # Initialize OrderTracker with config from application.yaml
        tracker_config = OrderTrackerConfig.from_dict(config.get("order_tracker", {}))
        self.order_tracker: OrderTracker = OrderTracker(station_id, tracker_config)
        
        # Error metrics for observability
        self.metrics: Dict[str, int] = {
            'frames_processed': 0,
            'frames_failed': 0,
            'upload_failures': 0,
            'ocr_errors': 0,
            'tracker_errors': 0,
        }
        
        log(f"[PIPELINE] State initialized for {station_id}")
    
    def reset(self):
        """Reset state for new video stream."""
        self.frame_counter = 0
        self.current_order_id = None
        self.previous_order_id = None
        self.order_frame_count = 0
        self.last_order_time = time.time()
        self.pending_frames.clear()
        self.order_tracker.reset()
        log(f"[PIPELINE] State reset for {self.station_id}")
    
    def get_metrics(self) -> dict:
        """Get pipeline metrics for health monitoring."""
        return {
            **self.metrics,
            'pending_frames_count': len(self.pending_frames),
            'active_orders': len(self.order_tracker.get_active_orders()),
        }

# Global pipeline state instance
_pipeline_state = PipelineState(STATION_ID, cfg)

# Legacy aliases for backward compatibility (deprecation path)
_frame_counter = 0  # Kept for direct function access
_current_order_id = None
_previous_order_id = None
_order_frame_count = 0
_last_order_time = time.time()
_pending_frames = _pipeline_state.pending_frames
_tracker_config = OrderTrackerConfig.from_dict(cfg.get("order_tracker", {}))
_order_tracker = _pipeline_state.order_tracker

# ====== helpers ======
def safe_get_image(frame):
    """
    Robustly extract a numpy HxWxC BGR image from gvapython VideoFrame.
    Handles:
      - frame.image() returning numpy array
      - frame.image() returning generator (yielding numpy arrays)
      - frame.image() returning context manager (enter -> image)
      - frame.tensor() in some setups (rare)
    Returns numpy array or None.
    """
    # Preferred method: frame.image()
    for attr in ("image", "data", "tensor"):
        getter = getattr(frame, attr, None)
        if getter is None:
            continue

        try:
            img_obj = getter() if callable(getter) else getter
        except Exception as e:
            # some implementations require calling without parentheses, try that
            try:
                img_obj = getter
            except Exception:
                img_obj = None

        if img_obj is None:
            continue

        # If it's already a numpy array
        if isinstance(img_obj, np.ndarray):
            return img_obj

        # If it's a context manager: support "with img_obj as img: ..."
        if hasattr(img_obj, "__enter__") and hasattr(img_obj, "__exit__"):
            try:
                with img_obj as arr:
                    if isinstance(arr, np.ndarray):
                        return arr
                    # sometimes arr is bytes; try decode
                    if isinstance(arr, (bytes, bytearray)):
                        nparr = np.frombuffer(arr, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        return img
            except Exception:
                # give up on this attr and try next
                pass

        # If it's an iterator/generator
        if hasattr(img_obj, "__iter__") and not isinstance(img_obj, (bytes, bytearray, str)):
            try:
                it = iter(img_obj)
                first = next(it)
                if isinstance(first, np.ndarray):
                    return first
                # if it's bytes
                if isinstance(first, (bytes, bytearray)):
                    nparr = np.frombuffer(first, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return img
            except StopIteration:
                pass
            except Exception:
                pass

        # If it's bytes
        if isinstance(img_obj, (bytes, bytearray)):
            nparr = np.frombuffer(img_obj, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img

    # fallback None
    return None


def now_ms():
    return int(time.time() * 1000)


def upload_frame(order_id: str, frame_idx: int, image_bgr, max_retries: int = 3):
    """Upload frame to MinIO with retry logic."""
    try:
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            log("[frame_to_minio] Failed to encode frame")
            return False
        data = buf.tobytes()
        # Upload with station prefix: {STATION_ID}/{order_id}/frame_{idx}.jpg
        key = f"{STATION_ID}/{order_id}/frame_{frame_idx}.jpg"
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                client.put_object(
                    FRAMES_BUCKET,
                    key,
                    io.BytesIO(data),
                    len(data),
                    content_type="image/jpeg"
                )
                log(f"[frame_to_minio] Uploaded {FRAMES_BUCKET}/{key}")
                return True
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 0.1  # 0.1s, 0.2s, 0.4s
                    log(f"[frame_to_minio] Upload attempt {attempt+1} failed, retrying in {backoff:.1f}s: {e}")
                    time.sleep(backoff)
        
        log(f"[frame_to_minio] Upload failed after {max_retries} attempts: {last_error}")
        return False
    except Exception as e:
        log(f"[frame_to_minio] Upload error: {e}")
        return False


def copy_frame(src_order_id: str, src_frame_idx: int, dst_order_id: str, dst_frame_idx: int, max_retries: int = 3):
    """Copy a frame from one order folder to another in MinIO."""
    try:
        src_key = f"{STATION_ID}/{src_order_id}/frame_{src_frame_idx}.jpg"
        dst_key = f"{STATION_ID}/{dst_order_id}/frame_{dst_frame_idx}.jpg"
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(max_retries):
            try:
                client.copy_object(
                    FRAMES_BUCKET,
                    dst_key,
                    CopySource(FRAMES_BUCKET, src_key)
                )
                log(f"[frame_to_minio] Copied {src_key} -> {dst_key}")
                return True
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 0.1
                    log(f"[frame_to_minio] Copy attempt {attempt+1} failed, retrying in {backoff:.1f}s: {e}")
                    time.sleep(backoff)
        
        log(f"[frame_to_minio] Copy failed after {max_retries} attempts: {last_error}")
        return False
    except Exception as e:
        log(f"[frame_to_minio] Copy error: {e}")
        return False


def finalize_order(order_id: str):
    """
    Mark an order as finalized (ready for processing).
    
    NOTE: This does NOT write the EOS marker anymore. The EOS marker is ONLY written
    by pipeline_runner.py when the GStreamer pipeline actually ends. Writing EOS here
    caused a bug where frame-selector would prematurely cleanup all frames when
    the first order completed, missing subsequent orders in the video.
    
    The frame-selector will process all orders when it sees the final EOS marker
    from pipeline_runner.py (after video ends).
    """
    log(f"[frame_to_minio] Order {order_id} finalized and ready for processing")
    log(f"[frame_to_minio] Will be processed by frame-selector when video completes (EOS from pipeline_runner)")
    return True

# ====== gvapython entrypoint ======
import sys

def process_frame(frame: "VideoFrame"):
    """
    Called by gvapython plugin. Return True to continue pipeline.
    
    For persistent pipeline, this handles:
    - Non-blocking async OCR for order detection
    - Immediate frame capture without waiting for OCR
    - Order segmentation when OCR detects new order
    - Quality-based frame scoring (not hard filtering)
    - Grace period frame collection
    - EOS marker writing when order finalizes
    
    Uses AsyncOCRProcessor for non-blocking OCR and OrderTracker for
    robust frame accumulation and intelligent finalization.
    
    Exception handling strategy:
    - Each step has isolated error handling
    - Failures in one step don't block other steps
    - All errors are logged with context
    - Pipeline always continues (returns True)
    """
    global _frame_counter, _current_order_id, _previous_order_id, _order_frame_count
    global _last_order_time, _order_tracker, _async_ocr, _pending_frames, _pipeline_state
    _frame_counter += 1
    _pipeline_state.frame_counter = _frame_counter
    
    # Debug: Log to stderr (captured by pipeline log file)
    if _frame_counter == 1 or _frame_counter % 30 == 0:
        log(f"[DEBUG] Frame {_frame_counter} received, station={STATION_ID}")

    # Step 1: Extract image from frame
    try:
        image = safe_get_image(frame)
        if image is None:
            log(f"[frame_to_minio] Frame#{_frame_counter}: could not extract image. skipping.")
            _pipeline_state.metrics['frames_failed'] += 1
            return True

        # Convert CHW -> HWC if needed (some dlstreamer versions give C,H,W)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
            image = image.transpose(1, 2, 0).copy()

        # Validate shape
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            log(f"[frame_to_minio] invalid image shape: {getattr(image, 'shape', None)}")
            _pipeline_state.metrics['frames_failed'] += 1
            return True
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: image extraction error: {e}")
        _pipeline_state.metrics['frames_failed'] += 1
        return True

    # Step 2: Check for occlusions (YOLO regions)
    has_occlusion = False
    item_count = 0
    try:
        regs = list(frame.regions()) if hasattr(frame, "regions") else []
        for r in regs:
            label = (getattr(r, "label", "") or "").lower()
            if label in HAND_LABELS:
                has_occlusion = True
            else:
                item_count += 1
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: region extraction error (ignored): {e}")
        # Continue without occlusion info

    # Step 3: Submit frame for async OCR
    order_id = None
    try:
        order_id = _async_ocr.submit_frame(_frame_counter, image)
        latest_order_id = _async_ocr.get_current_order_id()
        if latest_order_id:
            order_id = latest_order_id
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: OCR submission error: {e}")
        _pipeline_state.metrics['ocr_errors'] += 1
        # Continue without OCR result

    # Debug logging
    if _frame_counter % 10 == 0:
        log(f"[DEBUG] Frame {_frame_counter}: order_id={order_id}, occluded={has_occlusion}, items={item_count}")

    # Step 4: Handle grace period for previous order
    try:
        if _previous_order_id and _previous_order_id != order_id:
            prev_order = _order_tracker.get_order(_previous_order_id)
            if prev_order and prev_order.grace_frames_remaining > 0:
                grace_frame_idx = prev_order.frame_count + 1
                grace_key = f"{STATION_ID}/{_previous_order_id}/grace_{grace_frame_idx}.jpg"
                if _order_tracker.add_grace_frame(
                    _previous_order_id, grace_key, image, item_count, has_occlusion
                ):
                    upload_frame(_previous_order_id, grace_frame_idx, image)
                    log(f"[TRACKER] Grace frame added for order {_previous_order_id}")
            else:
                _previous_order_id = None
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: grace period error: {e}")
        _pipeline_state.metrics['tracker_errors'] += 1
        _previous_order_id = None  # Reset to prevent repeated errors

    # Step 5: Check for orders ready to finalize
    try:
        ready_orders = _order_tracker.get_orders_ready_to_finalize()
        for ready_id in ready_orders:
            if ready_id != order_id:
                ready_order = _order_tracker.get_order(ready_id)
                if ready_order:
                    log(f"[TRACKER] Order {ready_id} ready to finalize: "
                        f"{ready_order.frame_count} frames, "
                        f"clean={ready_order.clean_frames_count}, "
                        f"occluded={ready_order.occluded_frames_count}")
                finalize_order(ready_id)
                _order_tracker.mark_finalized(ready_id)
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: finalization check error: {e}")
        _pipeline_state.metrics['tracker_errors'] += 1

    # Step 6: If no order detected, store as pending
    if not order_id:
        try:
            # Don't store full image copy - just metadata to reduce memory
            # Upload frame immediately and store reference
            pending_key = f"{STATION_ID}/pending/frame_{_frame_counter}.jpg"
            upload_frame("pending", _frame_counter, image)
            _pending_frames.append((_frame_counter, pending_key, has_occlusion, item_count))
            
            if _frame_counter % 10 == 0:
                log(f"[DEBUG] Frame {_frame_counter} stored as pending (OCR in progress)")
        except Exception as e:
            log(f"[frame_to_minio] Frame#{_frame_counter}: pending frame storage error: {e}")
        return True

    # Step 7: Handle order change and process pending frames
    try:
        if order_id != _current_order_id:
            if _pending_frames:
                log(f"[TRACKER] Assigning {len(_pending_frames)} pending frames to order {order_id}")
                for pf_idx, pf_key_or_image, pf_occluded, pf_items in _pending_frames:
                    _order_frame_count += 1
                    pf_key = f"{STATION_ID}/{order_id}/frame_{_order_frame_count}.jpg"
                    
                    # Handle both old format (image) and new format (key)
                    if isinstance(pf_key_or_image, np.ndarray):
                        pf_image = pf_key_or_image
                        _order_tracker.update_order(
                            order_id=order_id,
                            frame_key=pf_key,
                            item_count=pf_items,
                            has_occlusion=pf_occluded
                        )
                        upload_frame(order_id, _order_frame_count, pf_image)
                    else:
                        # New format: key reference (image already in MinIO as pending)
                        # Copy from pending/ to order_id/ folder
                        _order_tracker.update_order(
                            order_id=order_id,
                            frame_key=pf_key,
                            item_count=pf_items,
                            has_occlusion=pf_occluded
                        )
                        # Copy frame from pending to correct order folder
                        copy_frame("pending", pf_idx, order_id, _order_frame_count)
                _pending_frames.clear()
            
            if _current_order_id:
                _previous_order_id = _current_order_id
                log(f"[TRACKER] Order change: {_current_order_id} -> {order_id}")
            
            _current_order_id = order_id
            _pipeline_state.current_order_id = order_id
            existing_order = _order_tracker.get_order(order_id)
            _order_frame_count = len(existing_order.frames) if existing_order else 0
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: order change handling error: {e}")
        traceback.print_exc(file=sys.stderr)
        _pipeline_state.metrics['tracker_errors'] += 1

    # Step 8: Update tracker and upload current frame
    try:
        _order_frame_count += 1
        _last_order_time = time.time()
        _pipeline_state.order_frame_count = _order_frame_count
        
        frame_key = f"{STATION_ID}/{order_id}/frame_{_order_frame_count}.jpg"
        
        tracked = _order_tracker.update_order(
            order_id=order_id,
            frame_key=frame_key,
            item_count=item_count,
            has_occlusion=has_occlusion
        )
        
        if not upload_frame(order_id, _order_frame_count, image):
            _pipeline_state.metrics['upload_failures'] += 1
        
        _pipeline_state.metrics['frames_processed'] += 1
        
        if _order_frame_count % 5 == 0 or _order_frame_count == 1:
            log(f"[TRACKER] Order {order_id}: {tracked.frame_count} frames "
                f"(clean={tracked.clean_frames_count}, occluded={tracked.occluded_frames_count})")
    except Exception as e:
        log(f"[frame_to_minio] Frame#{_frame_counter}: tracker update error: {e}")
        traceback.print_exc(file=sys.stderr)
        _pipeline_state.metrics['tracker_errors'] += 1
    
    return True
