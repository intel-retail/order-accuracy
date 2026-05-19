"""
YOLO Worker — table-presence detection subprocess
===================================================

Runs in a separate spawned Python process (zero GStreamer state) to avoid
the memory-corruption crash caused by loading PyTorch/ultralytics inside the
GStreamer gvapython process.  Identical isolation strategy to ocr_worker.py.

Protocol
--------
Input queue:  (frame_id: int, h: int, w: int, img_bytes: bytes)
Output queue: (frame_id: int, has_objects: bool)
              ('ready', None)  — sent once after the model is loaded

'has_objects' is True when YOLO detects at least one object that is NOT in
HAND_LABELS above CONF threshold.  An empty table returns False.
"""

import os
import sys
import time
import traceback
import numpy as np


HAND_LABELS = {"hand", "person"}


def run_worker(input_q, output_q, model_path: str, conf_threshold: float = 0.25):
    """
    Entry-point called by multiprocessing.Process.

    Loads the INT8 OpenVINO YOLO model once, then loops reading frames from
    input_q and writing (frame_id, has_objects) results to output_q.
    """
    # Redirect stdout/stderr so logs appear tagged in the parent's stderr.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="yolo-worker - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("yolo-worker")

    # ── Load model ────────────────────────────────────────────────────────
    # Read TARGET_DEVICE from the environment (set in .env / docker-compose).
    # OPENVINO_DEVICE is used as a fallback in case TARGET_DEVICE is absent.
    # model.overrides['device'] must be set BEFORE the first predict() call
    # because Ultralytics compiles the OpenVINO model on first inference and
    # ignores device= changes thereafter.
    target_device = (
        os.environ.get('TARGET_DEVICE')
        or os.environ.get('OPENVINO_DEVICE', 'CPU')
    )
    model = None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path, task="detect")
        model.overrides['device'] = target_device.lower()  # Ultralytics normalises to lowercase
        log.info(f"YOLO model loaded: {model_path} (device={target_device})")
    except Exception as e:
        log.error(f"Failed to load YOLO model ({e}) — will return has_objects=True for all frames")
        traceback.print_exc(file=sys.stderr)

    # Signal parent that we're ready (model loaded or load failed)
    try:
        output_q.put(("ready", None))
    except Exception:
        pass

    # ── Inference loop ────────────────────────────────────────────────────
    while True:
        try:
            item = input_q.get(timeout=5)
        except Exception:
            # Queue empty / timeout — keep waiting
            continue

        if item is None:
            # Shutdown sentinel
            break

        try:
            frame_id, h, w, img_bytes = item
            frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, 3)
        except Exception as e:
            log.warning(f"Bad frame payload: {e}")
            continue

        has_objects = True  # safe default if inference fails

        if model is not None:
            try:
                results = model(frame, verbose=False)
                has_objects = False
                for r in results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for box in r.boxes:
                        label = r.names[int(box.cls[0])]
                        conf  = float(box.conf[0])
                        if label not in HAND_LABELS and conf >= conf_threshold:
                            has_objects = True
                            break
                    if has_objects:
                        break
            except Exception as e:
                log.warning(f"YOLO inference error frame={frame_id}: {e}")
                has_objects = True  # safe fallback

        try:
            output_q.put_nowait((frame_id, has_objects))
        except Exception:
            pass  # output queue full — parent will use cached state
