"""
ocr_worker.py - Standalone EasyOCR worker process.

Run by multiprocessing (spawn context) from frame_pipeline.py.
Completely isolated from GStreamer so PyTorch workers cannot
crash the gst-launch-1.0 process.
"""
import numpy as np


def _load_known_orders():
    """Load valid order IDs from orders.json for OCR validation."""
    import json
    import os
    orders_path = os.environ.get("ORDERS_PATH", "/config/orders.json")
    try:
        with open(orders_path, "r") as f:
            data = json.load(f)
        known = set(str(k) for k in data.keys())
        import sys
        print(f"[OCR-WORKER] Loaded {len(known)} known order IDs: {known}", file=sys.stderr, flush=True)
        return known
    except Exception as e:
        import sys
        print(f"[OCR-WORKER] WARNING: Could not load {orders_path}: {e}", file=sys.stderr, flush=True)
        return set()


def run_worker(input_queue, output_queue, models_dir):
    """
    Entry point of the OCR worker subprocess.

    Loads EasyOCR once, then loops reading (frame_id, h, w, img_bytes)
    from input_queue and putting (frame_id, order_id_or_None) to output_queue.
    A None on input_queue signals shutdown.
    """
    import cv2
    import easyocr
    import sys

    known_orders = _load_known_orders()

    try:
        reader = easyocr.Reader(
            ['en'], gpu=False, verbose=False,
            model_storage_directory=models_dir
        )
        output_queue.put(('ready', None))
    except Exception as exc:
        output_queue.put(('error', str(exc)))
        return

    while True:
        item = input_queue.get()
        if item is None:
            break
        frame_id, h, w, img_bytes = item
        try:
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 3))
            small = cv2.resize(img, (640, 360))
            results = reader.readtext(small, detail=0)
            order_id = None
            
            # DEBUG: Log all OCR results for troubleshooting
            if results:
                print(f"[OCR-DEBUG] frame={frame_id} raw_results={results}", file=sys.stderr, flush=True)
            
            for text in results:
                if '#' in text:
                    digits = ''.join(c for c in text if c.isdigit())
                    if len(digits) >= 3:
                        candidate = digits[:3]
                        # Validate against known orders to reject misreads
                        if known_orders and candidate not in known_orders:
                            print(f"[OCR-VALIDATE] frame={frame_id} rejected '{candidate}' "
                                  f"(not in known orders {known_orders})",
                                  file=sys.stderr, flush=True)
                            continue
                        order_id = candidate
                        break
            output_queue.put((frame_id, order_id))
        except Exception:
            output_queue.put((frame_id, None))
