"""
ocr_worker.py - Standalone EasyOCR worker process.

Run by multiprocessing (spawn context) from frame_pipeline.py.
Completely isolated from GStreamer so PyTorch workers cannot
crash the gst-launch-1.0 process.
"""
import numpy as np


def run_worker(input_queue, output_queue, models_dir):
    """
    Entry point of the OCR worker subprocess.

    Loads EasyOCR once, then loops reading (frame_id, h, w, img_bytes)
    from input_queue and putting (frame_id, order_id_or_None) to output_queue.
    A None on input_queue signals shutdown.
    """
    import cv2
    import easyocr

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
            for text in results:
                if '#' in text:
                    digits = ''.join(c for c in text if c.isdigit())
                    if len(digits) >= 3:
                        order_id = digits[:3]
                        break
            output_queue.put((frame_id, order_id))
        except Exception:
            output_queue.put((frame_id, None))
