import os
import re
import time
import asyncio
import numpy as np
import openvino as ov
from PIL import Image
from minio import Minio
from minio.error import S3Error
from openvino_genai import VLMPipeline, GenerationConfig
from config_loader import load_config
from order_results import add_result

# ============================================================
# CONFIG
# ============================================================

cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]
VLM_CFG = cfg["vlm"]

MINIO_ENDPOINT = MINIO["endpoint"]
SELECTED_BUCKET = BUCKETS["selected"]

MODEL_PATH = os.getenv(
    "VLM_MODEL_PATH",
    VLM_CFG.get("model_path", "/models/Qwen2.5-VL-7B-Instruct-ov-int8")
)

DEVICE = os.getenv(
    "OPENVINO_DEVICE",
    VLM_CFG.get("device", "CPU")
)
MAX_NEW_TOKENS = VLM_CFG.get("max_new_tokens", 512)
TEMPERATURE = VLM_CFG.get("temperature", 0.2)

# ============================================================
# MINIO CLIENT (single instance)
# ============================================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=False,
)

# ============================================================
# VLM MODEL (SINGLETON, LOADED ONCE)
# ============================================================

BLACKLIST = {
    "total", "total items", "items", "quantity",
    "subtotal", "tax", "bill", "amount", "price"
}
class VLMComponent:
    _model = None
    _config_key = None

    def __init__(self, model_path, device, max_new_tokens, temperature):
        config_key = (model_path, device, max_new_tokens, temperature)

        if VLMComponent._model is None or VLMComponent._config_key != config_key:
            print(f"[VLM] Loading model from {model_path} on {device}", flush=True)
            core = ov.Core()
            if device.upper().startswith("GPU"):
                core.set_property("GPU", {
                    "GPU_THROUGHPUT_STREAMS": "1"
                })
            VLMComponent._model = VLMPipeline(
                models_path=model_path,
                device=device
            )

            VLMComponent._config_key = config_key
            print("[VLM] Model loaded successfully", flush=True)

        self.vlm = VLMComponent._model
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

    @staticmethod
    def extract_items(text: str):
        item_pattern = r"([A-Za-z ]+?)\s*[x:\-]\s*(\d+)"
        items = {}

        for name, qty in re.findall(item_pattern, text, flags=re.IGNORECASE):
            clean_name = name.strip().lower()
            if not clean_name.isdigit():
                items[clean_name] = int(qty)

        clean_items = {}
        for k, v in items.items():
            if not any(b in k for b in BLACKLIST):
                clean_items[k] = v
        return clean_items

    def process(self, images: list[np.ndarray]):
        if not images:
            raise ValueError("images list is empty")

        ov_frames = [ov.Tensor(img) for img in images]
        num_frames = len(ov_frames)

        img_tags = "".join(f"<ov_genai_image_{i}>" for i in range(num_frames))

        prompt = (
            f"You will receive {num_frames} frames.\n"
            f"Extract ONLY real product/item names visible in the images.\n"
            f"DO NOT include words like total, quantity, subtotal, tax, bill, items.\n"
            f"Format strictly as: item_name x number\n"
            f"If no real items are visible, output 'NO_ITEMS'.\n"
            f"{img_tags}"
        )

        start = time.perf_counter()

        output = self.vlm.generate(
            prompt,
            images=ov_frames,
            generation_config=self.gen_config
        )

        elapsed = time.perf_counter() - start

        raw_text = output.texts[0]
        items = self.extract_items(raw_text)

        response = {
            "items": [{"name": k, "quantity": v} for k, v in items.items()],
            "num_frames": num_frames,
            "inference_time_sec": round(elapsed, 3),
        }
        print("VLM -->> ")
        print(response)
        return response


# ============================================================
# GLOBAL VLM INSTANCE
# ============================================================

vlm_instance = VLMComponent(
    MODEL_PATH,
    device=DEVICE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
)

# ============================================================
# QUEUE + WORKER (SEQUENTIAL EXECUTION)
# ============================================================

vlm_queue: asyncio.Queue = asyncio.Queue()
_worker_started = False

async def _vlm_worker():
    print("[VLM-WORKER] Started (sequential mode)", flush=True)

    while True:
        order_id, future = await vlm_queue.get()

        try:
            print(f"[VLM-WORKER] Processing order {order_id}", flush=True)
            result = await _run_vlm_internal(order_id)
            future.set_result(result)
        except Exception as e:
            future.set_result({
                "order_id": order_id,
                "status": "error",
                "reason": str(e)
            })
        finally:
            vlm_queue.task_done()


# ============================================================
# INTERNAL VLM LOGIC (UNCHANGED SEMANTICS)
# ============================================================

async def _run_vlm_internal(order_id: str):
    print(f"[VLM] Order ID: {order_id}", flush=True)

    frames = []
    try:
        for obj in client.list_objects(
            SELECTED_BUCKET,
            prefix=f"{order_id}/",
            recursive=True
        ):
            if obj.object_name.lower().endswith(".jpg"):
                frames.append(obj.object_name)
    except S3Error:
        return {
            "order_id": order_id,
            "status": "error",
            "reason": "minio_list_failed"
        }

    if not frames:
        return {
            "order_id": order_id,
            "status": "no_frames"
        }

    frames.sort()

    images = []
    for f in frames:
        data = client.get_object(SELECTED_BUCKET, f)
        img = Image.open(data).convert("RGB").resize((512, 512))
        images.append(np.array(img))

    result = vlm_instance.process(images)
    result.update({
        "order_id": order_id,
        "status": "ok"
    })

    add_result(result)

    return result


# ============================================================
# PUBLIC ENTRYPOINT (API CALLS THIS)
# ============================================================

async def run_vlm(order_id: str):
    """
    ✔ Adds request to queue
    ✔ WAITS for its turn
    ✔ RETURNS final VLM response
    """

    global _worker_started

    loop = asyncio.get_running_loop()
    future = loop.create_future()

    await vlm_queue.put((order_id, future))

    print(
        f"[VLM-QUEUE] Registered order {order_id} "
        f"(queue_size={vlm_queue.qsize()})",
        flush=True
    )

    if not _worker_started:
        asyncio.create_task(_vlm_worker())
        _worker_started = True

    # IMPORTANT:
    # This await BLOCKS (async) until VLM finishes
    return await future
