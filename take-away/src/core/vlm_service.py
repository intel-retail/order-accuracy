import os
import re
import time
import asyncio
import logging
import numpy as np
from PIL import Image
from minio import Minio
from minio.error import S3Error
from .config_loader import load_config
from .order_results import add_result
from .validation_agent import validate_order
from .vlm_backend_factory import VLMBackendFactory
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# OVMS-only backend
import openvino as ov

# ============================================================
# CONFIG
# ============================================================

cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]
VLM_CFG = cfg["vlm"]

MINIO_ENDPOINT = MINIO["endpoint"]
SELECTED_BUCKET = BUCKETS["selected"]

# Station ID (for multi-station deployment)
STATION_ID = os.environ.get('STATION_ID', 'station_1')

# OVMS-only configuration
MAX_NEW_TOKENS = VLM_CFG.get("max_new_tokens", 512)
TEMPERATURE = VLM_CFG.get("temperature", 0.2)

logger.info(f"VLM Configuration: backend=OVMS, max_tokens={MAX_NEW_TOKENS}, temperature={TEMPERATURE}")

# ============================================================
# INVENTORY
# ============================================================

INVENTORY = cfg.get("inventory", [])

if not INVENTORY:
    logger.warning("Inventory list is empty - VLM will have no product constraints")
else:
    logger.info(f"Loaded inventory with {len(INVENTORY)} items")
    logger.debug(f"Inventory items: {INVENTORY}")

INVENTORY_TEXT = "\n".join(f"- {item}" for item in INVENTORY)

# ============================================================
# LOAD EXPECTED ORDERS
# ============================================================

ORDERS_FILE = "/config/orders.json"

try:
    with open(ORDERS_FILE, "r") as f:
        EXPECTED_ORDERS = json.load(f)
    logger.info(f"Loaded {len(EXPECTED_ORDERS)} expected orders from {ORDERS_FILE}")
    logger.debug(f"Order IDs: {list(EXPECTED_ORDERS.keys())}")
except Exception as e:
    logger.error(f"Failed to load orders.json: {e}")
    EXPECTED_ORDERS = {}

# ============================================================
# MINIO CLIENT
# ============================================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=False,
)

# ============================================================
# VLM MODEL SINGLETON
# ============================================================

BLACKLIST = {
    "total", "total items", "items", "quantity",
    "subtotal", "tax", "bill", "amount", "price"
}

class VLMComponent:
    _model = None
    _config_key = None
    
    def __init__(self, backend_type, config, max_new_tokens, temperature):
        config_key = (backend_type, str(config), max_new_tokens, temperature)

        if VLMComponent._model is None or VLMComponent._config_key != config_key:
            logger.info(f"Initializing {backend_type.upper()} backend...")
            logger.debug(f"Backend config: {config}")

            # Use factory to create backend
            self.vlm, self.gen_config = VLMBackendFactory.create_backend(
                backend_type=backend_type,
                config=config,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            VLMComponent._model = self.vlm
            VLMComponent._config_key = config_key
            logger.info(f"{backend_type.upper()} backend initialized successfully")
        else:
            logger.debug(f"Reusing existing {backend_type.upper()} backend instance")
            self.vlm = VLMComponent._model
            # Reconstruct gen_config from cached model
            if hasattr(VLMComponent._model, 'gen_config'):
                self.gen_config = VLMComponent._model.gen_config
            else:
                # For OVMS client, create mock config
                from .ovms_client import MockGenerationConfig
                self.gen_config = MockGenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False
                )

    @staticmethod
    def extract_items(text: str):
        # Pattern allows hyphens, digits, and spaces in item names
        # Matches items like "coca-cola 2 liter bottle x 1" or "water bottle x 2"
        item_pattern = r"([A-Za-z][A-Za-z0-9\- ]*?)\s*[x:\-]\s*(\d+)"
        items = {}

        for name, qty in re.findall(item_pattern, text, flags=re.IGNORECASE):
            clean_name = name.strip().lower()
            # Normalize common variations
            clean_name = clean_name.replace('coca-cola', 'coke').replace('coca cola', 'coke')
            if clean_name and not clean_name.isdigit():
                items[clean_name] = int(qty)

        clean_items = {}
        for k, v in items.items():
            if not any(b in k for b in BLACKLIST):
                clean_items[k] = v

        return clean_items

    def process(self, images: list[np.ndarray], unique_id: str = None, expected_items: list = None):
        if not images:
            logger.error("VLM process called with empty images list")
            raise ValueError("images list is empty")

        logger.debug(f"Processing {len(images)} images with VLM")
        ov_frames = [ov.Tensor(img) for img in images]
        num_frames = len(ov_frames)

        # OVMS automatically associates images with the prompt
        img_tags = ""

        # Build order-aware hint if expected items are known.
        # This steers the VLM to actively search for every item rather than
        # stopping after the most prominent one.
        if expected_items:
            expected_text = "\n".join(
                f"  - {item['name']} (qty expected: {item['quantity']})"
                for item in expected_items
            )
            order_hint = (
                f"This order is expected to contain:\n{expected_text}\n\n"
                f"Carefully examine ALL {num_frames} frames for EACH of the above items.\n"
                f"Report the total quantity of EACH item visible across ALL frames combined.\n\n"
            )
        else:
            order_hint = ""

        # ===== Inventory-aware + order-aware prompt =====
        prompt = (
            f"You will receive {num_frames} frames from a grocery order packing station.\n\n"
            f"{order_hint}"
            f"Recognize products ONLY from this inventory list:\n"
            f"{INVENTORY_TEXT}\n\n"
            f"Rules:\n"
            f"- Always choose the closest matching inventory item name.\n"
            f"- Never invent new product names outside the list.\n"
            f"- Count items carefully — report separate lines for different items.\n"
            f"- Format strictly as: inventory_item_name x quantity (one item per line)\n"
            f"- If no inventory items are visible, output NO_ITEMS.\n"
            f"{img_tags}"
        )

        logger.debug(f"VLM prompt length: {len(prompt)} chars")
        logger.info(f"VLM prompt (first 500 chars): {prompt[:500]}")
        logger.info(f"VLM prompt (last 200 chars): {prompt[-200:]}")
        logger.info(f"Image tags used: {repr(img_tags)}")
        logger.info(f"Number of ov_frames: {len(ov_frames)}, shapes: {[f.shape for f in ov_frames]}")
        
        # Write prompt to debug file
        with open('/tmp/vlm_prompt.txt', 'w') as f:
            f.write(f"Full Prompt:\n{prompt}\n\n")
            f.write(f"Image tags: {img_tags}\n")
            f.write(f"Number of frames: {num_frames}\n")
        
        logger.info(f"Starting VLM generation...")
        start = time.perf_counter()

        try:
            output = self.vlm.generate(
                prompt,
                images=ov_frames,
                generation_config=self.gen_config,
                unique_id=unique_id
            )
            logger.info(f"VLM generation completed, extracting text...")
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            raise

        elapsed = time.perf_counter() - start
        
        try:
            raw_text = output.texts[0]
            logger.info(f"Extracted raw text, length={len(raw_text)}")
        except Exception as e:
            logger.error(f"Failed to extract text from output: {e}")
            logger.error(f"Output type: {type(output)}, dir: {dir(output)}")
            raise
        
        # Debug: Write raw output to file
        try:
            with open('/tmp/vlm_raw_output.txt', 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Timestamp: {time.time()}\n")
                f.write(f"Raw text length: {len(raw_text)}\n")
                f.write(f"Raw text: {repr(raw_text)}\n")
                f.write(f"{'='*70}\n")
            logger.info("Debug output written to /tmp/vlm_raw_output.txt")
        except Exception as e:
            logger.error(f"Failed to write debug output: {e}")
        
        logger.debug(f"VLM raw output: {raw_text}")
        logger.info(f"VLM raw output text: {raw_text}")  # Also log at INFO level
        print(f"\n{'='*70}\nVLM RAW OUTPUT:\n{raw_text}\n{'='*70}\n", flush=True)  # Force print to stdout
        
        items = self.extract_items(raw_text)

        response = {
            "items": [{"name": k, "quantity": v} for k, v in items.items()],
            "num_frames": num_frames,
            "inference_time_sec": round(elapsed, 3),
        }

        logger.info(f"VLM inference completed: {len(items)} items detected in {elapsed:.3f}s")
        logger.debug(f"VLM response: {response}")
        return response


# ============================================================
# GLOBAL VLM INSTANCE
# ============================================================

# OVMS backend configuration
backend_config = {
    "ovms_endpoint": os.getenv("OVMS_ENDPOINT", VLM_CFG.get("ovms_endpoint", "http://" \
    ":8000")),
    "ovms_model": os.getenv("OVMS_MODEL_NAME", VLM_CFG.get("ovms_model", "Qwen/Qwen2.5-VL-7B-Instruct")),
    "timeout_sec": VLM_CFG.get("timeout_sec", 300),
}

logger.info(f"Initializing OVMS backend: {backend_config['ovms_endpoint']}")

vlm_instance = VLMComponent(
    backend_type="ovms",
    config=backend_config,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
)

# ============================================================
# QUEUE + WORKER
# ============================================================

vlm_queue: asyncio.Queue = asyncio.Queue()
_worker_started = False

async def _vlm_worker():
    logger.info("VLM worker started (sequential processing mode)")

    while True:
        order_id, station_id, future = await vlm_queue.get()

        try:
            logger.info(f"[WORKER] Processing order_id={order_id}, station_id={station_id} (queue_remaining={vlm_queue.qsize()})")
            result = await _run_vlm_internal(order_id, station_id)
            future.set_result(result)
            logger.info(f"[WORKER] Completed order_id={order_id}, status={result.get('status')}")
        except Exception as e:
            logger.error(f"[WORKER] Failed processing order_id={order_id}: {e}", exc_info=True)
            future.set_result({
                "order_id": order_id,
                "station_id": station_id,
                "status": "error",
                "reason": str(e)
            })
        finally:
            vlm_queue.task_done()

# ============================================================
# INTERNAL EXECUTION
# ============================================================

async def _run_vlm_internal(order_id: str, station_id: str):
    # Generate unique transaction ID: stationid_orderid
    unique_id = f"{station_id}_{order_id}"
    
    logger.info(f"="*80)
    logger.info(f"[VLM-START] Transaction ID: {unique_id}")
    logger.info(f"[INTERNAL] Starting VLM processing for order_id={order_id}, station={station_id}")
    logger.info(f"="*80)

    # Resolve expected items early — needed for the order-aware VLM prompt.
    expected_items = EXPECTED_ORDERS.get(order_id)
    if expected_items is None:
        logger.error(f"[INTERNAL] Order not found in orders.json: order_id={order_id}")
        return {"order_id": order_id, "status": "error", "reason": "order_not_found"}
    logger.info(f"[INTERNAL] Expected items for order_id={order_id}: {len(expected_items)} items")
    logger.debug(f"[INTERNAL] Expected items detail: {expected_items}")

    try:
        frames = []
        # Frame selector saves to: {station_id}/{order_id}/rank_{N}.jpg
        frame_prefix = f"{station_id}/{order_id}/"
        logger.debug(f"[INTERNAL] Listing frames from MinIO bucket={SELECTED_BUCKET}, prefix={frame_prefix}")
        try:
            for obj in client.list_objects(
                SELECTED_BUCKET,
                prefix=frame_prefix,
                recursive=True
            ):
                if obj.object_name.lower().endswith(".jpg"):
                    frames.append(obj.object_name)
        except S3Error as e:
            logger.error(f"[INTERNAL] MinIO list failed for order_id={order_id}: {e}")
            return {"order_id": order_id, "status": "error", "reason": "minio_list_failed"}

        if not frames:
            logger.warning(f"[INTERNAL] No frames found for order_id={order_id} at prefix={frame_prefix}")
            return {"order_id": order_id, "status": "no_frames"}

        frames.sort()
        logger.info(f"[INTERNAL] Found {len(frames)} frames for order_id={order_id}")
        logger.debug(f"[INTERNAL] Frame keys: {frames}")

        images = []
        for f in frames:
            logger.debug(f"[INTERNAL] Loading frame: {f}")
            data = client.get_object(SELECTED_BUCKET, f)
            img = Image.open(data).convert("RGB").resize((512, 512))
            images.append(np.array(img))
        
        logger.info(f"[INTERNAL] Loaded {len(images)} images, starting VLM inference for order_id={order_id}")

        # ---- Run VLM (order-aware: pass expected items so prompt includes them) ----
        logger.info(f"[VLM-CALL] Transaction ID: {unique_id} - Calling VLM with {len(images)} frames")
        vlm_result = vlm_instance.process(images, unique_id=unique_id, expected_items=expected_items)
        detected_items = vlm_result["items"]
        logger.info(f"[VLM-RESPONSE] Transaction ID: {unique_id} - VLM returned {len(detected_items)} items")

        # ---- Retry on empty response ----
        # The model occasionally returns an empty string when the prompt with expected
        # items plus 5 frames is large.  Retry once with inventory-only prompt
        # (no expected-items hint) to maximize the chance of getting a valid response.
        if not detected_items:
            logger.warning(f"[VLM-RETRY] {unique_id} - VLM returned empty response, retrying with simplified prompt")
            vlm_result = vlm_instance.process(images, unique_id=f"{unique_id}_retry", expected_items=None)
            detected_items = vlm_result["items"]
            logger.info(f"[VLM-RETRY] {unique_id} - Retry returned {len(detected_items)} items")

        logger.info(f"[INTERNAL] VLM detected {len(detected_items)} items for order_id={order_id}")
        logger.debug(f"[INTERNAL] Detected items: {detected_items}")
    except Exception as e:
        logger.error(f"[INTERNAL] Error before validation for order_id={order_id}: {e}", exc_info=True)
        raise

    logger.info(f"[INTERNAL] Starting validation for order_id={order_id}")
    validation = validate_order(expected_items, detected_items, vlm_instance.vlm, transaction_id=unique_id)
    logger.info(f"[INTERNAL] Validation complete for order_id={order_id}")
    logger.debug(f"[INTERNAL] Validation results: {validation}")

    has_errors = (
        validation["missing"] or
        validation["extra"] or
        validation["quantity_mismatch"]
    )

    final_result = {
        "order_id": order_id,
        "station_id": station_id,
        "expected_items": expected_items,
        "detected_items": detected_items,
        "validation": validation,
        "status": "validated" if not has_errors else "mismatch",
        "num_frames": vlm_result["num_frames"],
        "inference_time_sec": vlm_result["inference_time_sec"]
    }

    logger.info(f"[INTERNAL] Final status for order_id={order_id}: {final_result['status']}")
    logger.debug(f"[INTERNAL] Storing result in order_results deque")
    add_result(final_result, station_id=station_id)
    logger.info(f"[INTERNAL] Result stored successfully for order_id={order_id}")
    
    logger.info(f"="*80)
    logger.info(f"[VLM-END] Transaction ID: {unique_id} - Status: {final_result['status']}")
    logger.info(f"[VLM-END] Transaction ID: {unique_id} - Inference time: {final_result['inference_time_sec']}s")
    logger.info(f"="*80)
    
    return final_result


# ============================================================
# PUBLIC API
# ============================================================

async def run_vlm(order_id: str, station_id: str = None):
    global _worker_started

    # Use provided station_id or default
    if station_id is None:
        station_id = STATION_ID

    logger.info(f"[API] VLM request received for order_id={order_id}, station_id={station_id}")
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    await vlm_queue.put((order_id, station_id, future))
    queue_size = vlm_queue.qsize()
    logger.info(f"[API] Order queued: order_id={order_id}, station_id={station_id}, queue_size={queue_size}")

    if not _worker_started:
        logger.info("[API] Starting VLM worker task")
        asyncio.create_task(_vlm_worker())
        _worker_started = True

    logger.debug(f"[API] Waiting for VLM processing of order_id={order_id}")
    result = await future
    logger.info(f"[API] VLM processing completed for order_id={order_id}")
    return result
