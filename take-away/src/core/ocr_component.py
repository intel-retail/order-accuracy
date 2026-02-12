import cv2
import easyocr
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CONFIG = {
    "write_debug": True,
    "fps_process": 1,
}

logger.info("Initializing EasyOCR (optimized for CPU)")
reader = easyocr.Reader(
    ['en'], 
    gpu=False, 
    verbose=False,
    quantize=True,
    model_storage_directory='/tmp/easyocr_models'
)
logger.info("EasyOCR initialized successfully")


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
    logger.debug(f"[OCR] Starting OCR for frame_idx={frame_idx}")

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
        canvas_size=1024,  # Optimization 2: Reduced from 2560
        mag_ratio=1.0,
        allowlist='0123456789#'  # Optimization 3: Only digits and #
    )
    
    logger.debug(f"[OCR] EasyOCR found {len(results)} text regions")
    candidates = []
    
    for (bbox, text, conf) in results:
        raw = text.replace(" ", "")
        logger.debug(f"[OCR] Detected text: '{raw}' (confidence={conf:.2f})")
        
        if "#" in raw:
            after = raw.split("#", 1)[1]
            
            digits = ""
            for c in after:
                if c.isdigit():
                    digits += c
                else:
                    break
            
            if digits:
                logger.debug(f"[OCR] Found order ID candidate: '{digits}' (confidence={conf:.2f})")
                candidates.append((digits, conf))
    
    if not candidates:
        logger.debug(f"[OCR] No order ID found in frame_idx={frame_idx}")
        return None
    
    # Prefer 3-digit numbers
    three_digit = [(n, c) for n, c in candidates if len(n) == 3]
    if three_digit:
        order_id = max(three_digit, key=lambda x: x[1])[0]
        logger.info(f"[OCR] Extracted order ID: {order_id} (frame_idx={frame_idx})")
        return order_id
    
    order_id = max(candidates, key=lambda x: x[1])[0]
    logger.info(f"[OCR] Extracted order ID: {order_id} (frame_idx={frame_idx})")
    return order_id
