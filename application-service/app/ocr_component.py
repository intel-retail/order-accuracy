import cv2
import easyocr
import time
import os

CONFIG = {
    "write_debug": True,
    "fps_process": 1,
}

print("[OCR] Initializing EasyOCR (optimized for CPU)...")
reader = easyocr.Reader(
    ['en'], 
    gpu=False, 
    verbose=False,
    quantize=True,
    model_storage_directory='/tmp/easyocr_models'
)
print("[OCR] EasyOCR ready")


def now_ms():
    return int(time.time() * 1000)


def preprocess_roi(roi):
    """Lightweight preprocessing for speed."""
    # Optimization 1: Resize to 2x (reduced from 3x for speed)
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Optimization 1: Simple histogram equalization (faster than CLAHE)
    gray = cv2.equalizeHist(gray)
    
    # Optimization 1: Otsu's thresholding (faster than adaptive)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def read_order_id(frame, frame_idx=None):
    """Extract order ID after '#' using EasyOCR on full frame."""

    thresh = preprocess_roi(frame)
    
    # Optimization 2 & 3: Smaller canvas + character allowlist
    results = reader.readtext(
        thresh,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        text_threshold=0.6,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=1280,  # Optimization 2: Reduced from 2560
        mag_ratio=1.0,
        allowlist='0123456789#'  # Optimization 3: Only digits and #
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
                candidates.append((digits, conf))
    
    if not candidates:
        return None
    
    # Prefer 3-digit numbers
    three_digit = [(n, c) for n, c in candidates if len(n) == 3]
    if three_digit:
        return max(three_digit, key=lambda x: x[1])[0]
    
    return max(candidates, key=lambda x: x[1])[0]