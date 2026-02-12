from __future__ import annotations

import json
import logging
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from io import BytesIO

import gradio as gr


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8080/api"


@dataclass(frozen=True)
class Scenario:
    image_path: Path
    order_manifest: Dict[str, object]
    validation: Dict[str, object]
    metrics: Dict[str, object]


APP_TITLE = "Dine-In Order Accuracy Benchmark"
APP_DESCRIPTION = (
    "Staff-triggered plate validation workflow that mirrors full-service restaurant expo operations. "
    "Demonstrates zero-training deployment with vision-language models, semantic order reconciliation, "
    "and latency instrumentation aligned to the two-second service window."
)

# Since app.py is in /app/src/, go up one level to /app/
ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"
CONFIGS_DIR = ROOT_DIR / "configs"
ORDERS_PATH = CONFIGS_DIR / "orders.json"
INVENTORY_PATH = CONFIGS_DIR / "inventory.json"


def _default_validation(image_id: str) -> Dict[str, object]:
    return {
        "order_complete": False,
        "missing_items": [],
        "extra_items": [],
        "modifier_validation": {
            "status": "pending",
            "details": [f"No validation profile configured for {image_id}"],
        },
        "accuracy_score": None,
    }


def _default_metrics(image_id: str) -> Dict[str, object]:
    return {
        "end_to_end_latency_ms": None,
        "vlm_inference_ms": None,
        "agent_reconciliation_ms": None,
        "within_operational_window": None,
        "notes": f"No metric profile configured for {image_id}",
    }


def _load_orders() -> Dict[str, Scenario]:
    if not ORDERS_PATH.exists():
        return {}

    with ORDERS_PATH.open("r", encoding="utf-8") as orders_file:
        data = json.load(orders_file)

    scenarios: Dict[str, Scenario] = {}

    for order in data.get("orders", []):
        image_id = order.get("image_id")
        if not image_id:
            continue

        image_path = IMAGES_DIR / f"{image_id}.jpg"
        manifest = {
            key: value
            for key, value in order.items()
            if key != "image_id"
        }

        label = (
            f"{order.get('image_id')} – {order.get('restaurant', 'Unknown')} "
            f"Table {order.get('table_number', '?')}"
        )

        scenarios[label] = Scenario(
            image_path=image_path,
            order_manifest=manifest,
            validation=_default_validation(image_id),
            metrics=_default_metrics(image_id),
        )

    return scenarios


_SCENARIOS = _load_orders()
_DEFAULT_SCENARIO = next(iter(_SCENARIOS)) if _SCENARIOS else ""


def load_scenario(name: str) -> Tuple[str, Dict[str, object]]:
    scenario = _SCENARIOS.get(name)
    if not scenario:
        return "", {"error": f"Scenario '{name}' is not available"}

    image_value = str(scenario.image_path) if scenario.image_path.exists() else ""
    return image_value, scenario.order_manifest


def validate_plate(name: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Validate plate by calling the FastAPI endpoint.
    
    Args:
        name: Scenario name/identifier
        
    Returns:
        Tuple of (validation_result, metrics)
    """
    logger.info(f"[GRADIO] validate_plate called with name: {name}")
    
    scenario = _SCENARIOS.get(name)
    if not scenario:
        logger.warning(f"[GRADIO] No scenario found for: {name}")
        return _default_validation(name), _default_metrics(name)
    
    # Check if image exists
    if not scenario.image_path.exists():
        logger.warning(f"[GRADIO] Image not found: {scenario.image_path}")
        return _default_validation(name), _default_metrics(name)
    
    try:
        logger.info(f"[GRADIO] Calling API for validation...")
        
        # Prepare order data from scenario manifest
        # Map items_ordered from JSON to items for API
        order_items = scenario.order_manifest.get("items_ordered", [])
        api_items = [{"name": item.get("item"), "quantity": item.get("quantity")} for item in order_items]
        
        order_data = {
            "order_id": scenario.order_manifest.get("image_id", name),
            "table_number": scenario.order_manifest.get("table_number", ""),
            "restaurant": scenario.order_manifest.get("restaurant", ""),
            "items": api_items,
            "modifiers": scenario.order_manifest.get("modifiers", [])
        }
        
        logger.info(f"[GRADIO] Order has {len(api_items)} expected items: {[item['name'] for item in api_items]}")
        
        logger.debug(f"[GRADIO] Order data: {order_data}")
        
        # Open and send image file
        with open(scenario.image_path, 'rb') as image_file:
            files = {'image': (scenario.image_path.name, image_file, 'image/png')}
            data = {'order': json.dumps(order_data)}
            
            logger.info(f"[GRADIO] Sending POST request to {API_BASE_URL}/validate")
            
            # Call FastAPI validation endpoint (extended timeout for 7B model with inventory)
            response = requests.post(
                f"{API_BASE_URL}/validate",
                files=files,
                data=data,
                timeout=320  # 5+ minutes for 7B model inference
            )
        
        logger.info(f"[GRADIO] API Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"[GRADIO] API returned successful validation: {result.get('validation_id')}")
            
            # Extract validation and metrics
            validation = {
                "order_complete": result.get("order_complete", False),
                "missing_items": result.get("missing_items", []),
                "extra_items": result.get("extra_items", []),
                "modifier_validation": result.get("modifier_validation", {}),
                "accuracy_score": result.get("accuracy_score", 0.0)
            }
            
            metrics = result.get("metrics", _default_metrics(name))
            
            return validation, metrics
        else:
            logger.error(f"[GRADIO] API error: {response.status_code} - {response.text}")
            # Return default empty values on API error
            return _default_validation(name), _default_metrics(name)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"[GRADIO] Request failed: {e}")
        # Return default empty values on connection error
        return _default_validation(name), _default_metrics(name)
    except Exception as e:
        logger.exception(f"[GRADIO] Validation error: {e}")
        return _default_validation(name), _default_metrics(name)


with gr.Blocks(
    title=APP_TITLE,
    css="""
    :root {
        --primary-500: #1f6feb !important;
        --primary-600: #1a5fd0 !important;
        --primary-700: #174ea6 !important;
    }

    #validate-plate-btn {
        width: 200px !important;
    }

    #validate-plate-btn button {
        width: 200px !important;
        color: white !important;
        padding: 0.35rem 0.8rem !important;
        white-space: nowrap;
    }
    """
) as app:
    gr.Markdown(f"## {APP_TITLE}\n{APP_DESCRIPTION}")

    with gr.Row():
        with gr.Column():
            scenario_dropdown = gr.Dropdown(
                label="Scenario",
                choices=list(_SCENARIOS.keys()),
                value=_DEFAULT_SCENARIO or None,
                interactive=True,
            )
            validate_button = gr.Button(
    "Validate Plate",
    variant="primary",
    elem_id="validate-plate-btn",
)

    with gr.Row():
        image_display = gr.Image(label="Plate Image", interactive=False)
        order_display = gr.JSON(label="Order Ticket", value={})

    validation_display = gr.JSON(label="Validation Result")
    metrics_display = gr.JSON(label="Performance Metrics")

    def _on_scenario_change(
        name: str,
    ) -> Tuple[str, Dict[str, object], Dict[str, object], Dict[str, object]]:
        image_path, order_manifest = load_scenario(name)
        empty_validation = {
            "order_complete": None,
            "missing_items": [],
            "extra_items": [],
            "modifier_validation": {"status": "pending"},
            "accuracy_score": None,
        }
        empty_metrics = {
            "end_to_end_latency_ms": None,
            "vlm_inference_ms": None,
            "agent_reconciliation_ms": None,
            "within_operational_window": None,
            "cpu_utilization": None,
            "gpu_utilization": None,
            "memory_utilization": None,
            "gpu_memory_utilization": None
        }
        return image_path, order_manifest, empty_validation, empty_metrics

    def _on_validate(name: str) -> Tuple[Dict[str, object], Dict[str, object]]:
        return validate_plate(name)

    scenario_dropdown.change(
        fn=_on_scenario_change,
        inputs=scenario_dropdown,
        outputs=[image_display, order_display, validation_display, metrics_display],
        show_progress=False,
    )

    validate_button.click(
        fn=_on_validate,
        inputs=scenario_dropdown,
        outputs=[validation_display, metrics_display],
    )

    if _DEFAULT_SCENARIO:
        (
            initial_image,
            initial_order,
            initial_validation,
            initial_metrics,
        ) = _on_scenario_change(_DEFAULT_SCENARIO)
        image_display.value = initial_image
        order_display.value = initial_order
        validation_display.value = initial_validation
        metrics_display.value = initial_metrics

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
