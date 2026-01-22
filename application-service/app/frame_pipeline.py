# frame_to_minio.py
import os
import io
import time
import traceback
import cv2
import numpy as np
from minio import Minio
from config_loader import load_config
cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]

FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]

print("[ocr] Loading EasyOCR globally...")
from ocr_component import reader, read_order_id
EASY_OCR_READER = reader
READ_ORDER_ID_FN = read_order_id
print("[ocr] EasyOCR ready.")

ocr_module = None   # lazy import for ocr.py

# Try to import VideoFrame for typing only (gvapython provides it)
try:
    from gstgva import VideoFrame
except Exception:
    VideoFrame = object

# ====== Configuration (from docker-compose envs) ======
MINIO_ENDPOINT = MINIO["endpoint"]

HAND_LABELS = {"hand", "person"}

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
        print(f"[frame_to_minio] Created bucket: {FRAMES_BUCKET}")
    except Exception as e:
        print("[frame_to_minio] Warning - cannot create bucket:", e)

# ====== globals ======
_frame_counter = 0

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


def preprocess_for_ocr(img_bgr):
    # Slight enhance for printed slips
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )
        return thresh
    except Exception:
        return img_bgr


def read_order_id(img_bgr):
    """
    Extract order id as number immediately after '#' using pytesseract.
    Returns string or None.
    """
    try:
        proc = preprocess_for_ocr(img_bgr)
        data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT)
        candidates = []
        for text in data.get("text", []):
            print("[frame_to_minio] OCR TEXT:", text)
            raw = (text or "").replace(" ", "")
            if "#" in raw:
                after = raw.split("#", 1)[1]
                digits = ""
                for c in after:
                    if c.isdigit():
                        digits += c
                    else:
                        break
                if digits:
                    candidates.append(digits)

        if not candidates:
            return None

        # prefer 3-digit
        three = [c for c in candidates if len(c) == 3]
        if three:
            return three[0]
        return candidates[0]
    except Exception as e:
        print("[frame_to_minio] OCR error:", e)
        return None


def upload_frame(order_id: str, frame_idx: int, image_bgr):
    try:
        ok, buf = cv2.imencode(".jpg", image_bgr)
        if not ok:
            print("[frame_to_minio] Failed to encode frame")
            return False
        data = buf.tobytes()
        key = f"{order_id}/{frame_idx}.jpg"
        client.put_object(
            FRAMES_BUCKET,
            key,
            io.BytesIO(data),
            len(data),
            content_type="image/jpeg"
        )
        print(f"[frame_to_minio] Uploaded {FRAMES_BUCKET}/{key}")
        return True
    except Exception as e:
        print("[frame_to_minio] Upload error:", e)
        return False


# ====== gvapython entrypoint ======
def process_frame(frame: "VideoFrame"):
    """
    Called by gvapython plugin. Return True to continue pipeline.
    """
    global _frame_counter
    _frame_counter += 1
    print(f"[frame_to_minio] Frame#{_frame_counter} hit")

    try:
        # 1) get image robustly
        image = safe_get_image(frame)
        if image is None:
            print(f"[frame_to_minio] Frame#{_frame_counter}: could not extract image. skipping.")
            return True

        # Convert CHW -> HWC if needed (some dlstreamer versions give C,H,W)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[2] not in (1, 3, 4):
            # likely CHW
            image = image.transpose(1, 2, 0).copy()

        # Validate shape
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            print("[frame_to_minio] invalid image shape:", getattr(image, "shape", None))
            return True

        # 2) use detections already supplied by gvadetect if present
        try:
            regs = list(frame.regions()) if hasattr(frame, "regions") else []
            for r in regs:
                label = (getattr(r, "label", "") or "").lower()
                if label in HAND_LABELS:
                    # skip frame
                    # print("[frame_to_minio] skipping - hand/person detected")
                    return True
        except Exception:
            pass

        order_id = READ_ORDER_ID_FN(image)

        if not order_id:
            return True

        upload_frame(order_id, _frame_counter, image)
        return True

    except Exception:
        print("[frame_to_minio] Unexpected error in process_frame():")
        traceback.print_exc()
        return True
