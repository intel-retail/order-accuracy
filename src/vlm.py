"""Vision Language Model integration for grocery item detection."""

import json
from typing import List, Dict, Any, Tuple

import requests
from config import VLM_URL, VLM_MODEL, FRAME_BASE_URL_TEMPLATE, VSS_IP, logger
from save_vlm_result import save_to_minio

def build_vlm_payload(frame_records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    """Build payload for VLM API call using video format."""
    # Build the video URLs array
    video_urls = []
    for rec in frame_records:
        # Use imageUri from MQTT message if available, otherwise fall back to template
        if 'image_uri' in rec and rec['image_uri']:
            # Convert relative path to full URL
            base_url = f"http://{VSS_IP}:12345/datastore"
            image_url = base_url + rec['image_uri']
        else:
            # Fallback to old template-based URL construction
            image_url = FRAME_BASE_URL_TEMPLATE.format(
                frame_id=rec['frame_id'], 
                chunk=rec['chunk'], 
                frame=rec['frame_number']
            )
        
        logger.info(f"Adding image to VLM payload: {image_url}")
        video_urls.append(image_url)
    
    # Build the message content with video format
    messages_content = [
        {
            "type": "text",
            "text": """
                Analyze this image from a grocery checkout counter. 
                Identify every distinct grocery item you see. 
                Also, find and read the bill number from any receipt or screen present.

                Special rules:                
                - Do not include any item with a count of zero in the output.
                - BillNumber must always be a plain integer or null if it is not visible.
                - Do not wrap BillNumber in an object, string, or special formatting.
                - If any order starts with zero(0), example BillNumber=050 or 007, consider it as 50 or 7 only, please don't add in response.
                - Dont add space in name and count key. It should be "Name" and "Count". Output format must be strictly followed.


                Output format (strict JSON):
                {
                "Items": [
                    {"Name": "Name of the Item", "Count": 0},
                    {"Name": "Name of the Item", "Count": 0}
                ],
                "BillNumber": 0
                }

                Additional Context: 
                We only have the following items in our inventory: 
                  - Red Apple
                  - Green Apple
                  - Banana
                  - Coca-Cola
                  - JIF Peanut Butter Packet
                  - Mott's Apple Sauce
                  - Lemon
                  - Water Bottle
                  - Plastic Knife
                  - Plastic Spoon
                  - Plastic Fork
                  
              
                Additional considerations:
                - Focus only on the checkout bay area.
                - Ignore irrelevant frames (customers, unrelated inventory).
                - Handle variations in lighting, angles, or item arrangements.
                - Please ensure Output must be a valid JSON and follow the schema exactly.
                - If any order starts with zero(0), please don't add in response.
             """
        },
        {
            "type": "video",
            "video": video_urls
        }
    ]
    
    return {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": messages_content}],
        "max_completion_tokens": 500,
        "temperature": 0.1,
        "top_p": 0.3,
        "frequency_penalty": 1,
        "seed": seed  # <-- Add seed for deterministic output
    }


def call_vlm(
    frame_records: List[Dict[str, Any]],
    seed: int = 42,
    order_id: str = None,
    video_id: str = None
) -> Tuple[bool, Dict[str, Any], str]:
    """Call the Vision Language Model to analyze frames. Optionally save output to MinIO with order_id and accept video_id."""
    import time
    payload = build_vlm_payload(frame_records, seed=seed)
    logger.info(f"##########video_id==================: {video_id}")
    try:
        start_time = time.time()
        resp = requests.post(VLM_URL, json=payload, timeout=600)
        elapsed = time.time() - start_time
        logger.info(f"######### VLM call time taken: {elapsed:.2f} seconds")
        if resp.status_code != 200:
            return False, {}, f"VLM call failed: {resp.status_code} {resp.text}"
        try:
            data = resp.json()
        except Exception:
            return False, {}, f"Invalid VLM response: {resp.text[:200]}"
        content = None
        if isinstance(data, dict):
            choices = data.get('choices')
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get('message')
                    if isinstance(message, dict):
                        content = message.get('content')
        if not content:
            return False, {}, f"No content in VLM response: {data}"
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = content[json_start: json_end + 1]
            try:
                parsed = json.loads(json_str)
                # Use BillNumber from parsed JSON as order_id if present and not None/empty
                bill_number = parsed.get("BillNumber")
                logger.info(f"############bill_number #############: {bill_number}")
                minio_order_id = str(bill_number) if bill_number not in (None, "", 0) else order_id
                logger.info(f"############minio_order_id #############: {minio_order_id}")
                if minio_order_id is not None and str(minio_order_id).strip() != "":
                    save_to_minio(minio_order_id, parsed, video_id=video_id)
                return True, parsed, ""
            except Exception as e:
                return False, {}, f"Failed to parse JSON: {e}; content: {content}"
        return False, {}, f"JSON not found in content: {content}"
    except Exception as e:
        return False, {}, f"Exception calling VLM: {e}"
