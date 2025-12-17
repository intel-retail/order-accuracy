import sys
import asyncio

async def run_vlm(order_id, images):
    print("[VLM] START", flush=True)
    print(f"[VLM] Order ID: {order_id}", flush=True)
    print("[VLM] Frames received:", flush=True)

    for img in images:
        print(f"   - {img}", flush=True)

    # simulate processing
    await asyncio.sleep(0.1)

    print("[VLM] DONE", flush=True)

    return {
        "order_id": order_id,
        "items": ["item1", "item2"],
        "status": "ok"
    }
