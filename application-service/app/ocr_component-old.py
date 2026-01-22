import cv2
import time
import os
from frame_pipeline import read_order_id

CONFIG = {
    "write_debug": True,
    "fps_process": 1,
}

reader = easyocr.Reader(['en'], gpu=False, verbose=False)


def now_ms():
    return int(time.time() * 1000)

def preprocess_roi(roi):
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )
    return thresh

def read_order_id(frame, frame_idx=None):
    """Extract order ID after '#' using EasyOCR on full frame."""

    thresh = preprocess_roi(frame)
    results = reader.readtext(thresh)

    candidates = []

    for (_, text, conf) in results:
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

    three_digit = [(n, c) for n, c in candidates if len(n) == 3]
    if three_digit:
        return max(three_digit, key=lambda x: x[1])[0]

    return max(candidates, key=lambda x: x[1])[0]
