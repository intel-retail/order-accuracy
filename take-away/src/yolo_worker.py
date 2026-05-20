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
    #
    # IMPORTANT — Ultralytics 8.3.0 + OpenVINO backend device handling:
    #   AutoBackend.__init__() hardcodes device_name="AUTO" when calling
    #   core.compile_model() for .xml models. This means model.overrides['device']
    #   has NO effect on which OpenVINO device executes inference; OpenVINO's
    #   AUTO plugin automatically selects the best available device (GPU when
    #   present, CPU otherwise).
    #
    #   model.overrides['device'] only controls the PyTorch tensor device used
    #   outside the OpenVINO inference path. It must be a value that
    #   select_device() accepts. On Intel-only systems (no CUDA) the only valid
    #   string is "cpu". Using "gpu", "intel:GPU", or "intel:CPU" all raise
    #   "Invalid CUDA device" errors in select_device() in 8.3.0, causing every
    #   inference call to throw and fall through to the has_objects=True fallback.
    target_device = (
        os.environ.get('TARGET_DEVICE')
        or os.environ.get('OPENVINO_DEVICE', 'CPU')
    ).upper()
    # "cpu" is the only device string select_device() accepts on Intel-only systems.
    # Actual OpenVINO execution device is controlled via the monkey-patch below.
    yolo_device = "cpu"

    # ── Force OpenVINO to respect TARGET_DEVICE ───────────────────────────
    # Ultralytics 8.3.0 hardcodes device_name="AUTO" in core.compile_model()
    # for OpenVINO .xml models, ignoring model.overrides['device'] entirely.
    # We intercept compile_model() before YOLO loads and substitute the actual
    # target device so inference runs on CPU or GPU as configured.
    ov_device = target_device  # e.g. "CPU" or "GPU"
    try:
        # DLStreamer container has 'openvino' but not 'openvino.runtime'
        import openvino as _ov
        _orig_compile = _ov.Core.compile_model

        def _patched_compile(self, model_or_path, device_name=None, config=None, **kwargs):
            if device_name == "AUTO" or device_name is None:
                device_name = ov_device
            if config is not None:
                return _orig_compile(self, model_or_path, device_name, config=config, **kwargs)
            return _orig_compile(self, model_or_path, device_name, **kwargs)

        _ov.Core.compile_model = _patched_compile
        log.info(f"OpenVINO compile_model patched: AUTO → {ov_device}")
    except Exception as e:
        log.warning(f"Could not patch OpenVINO compile_model ({e}); device selection relies on AUTO")

    model = None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path, task="detect")
        model.overrides['device'] = yolo_device
        log.info(f"YOLO model loaded: {model_path} (pytorch_device={yolo_device}, ov_device={ov_device})")
    except Exception as e:
        log.error(f"Failed to load YOLO model ({e}) — will return has_objects=True for all frames")
        traceback.print_exc(file=sys.stderr)
    finally:
        # Restore original compile_model to avoid side-effects on other OV models in this process
        try:
            _ov.Core.compile_model = _orig_compile
        except NameError:
            pass

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
