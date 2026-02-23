from __future__ import annotations

import json
import logging
import httpx
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
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


APP_TITLE = "🍽️ Dine-In Order Accuracy"
APP_DESCRIPTION = "AI-powered plate validation for full-service restaurant operations"

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

        # Try multiple image extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = IMAGES_DIR / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        # Skip orders without images (user must add images first)
        if image_path is None:
            logger.warning(f"[APP] Skipping order {image_id} - no image found in {IMAGES_DIR}")
            continue
            
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

# Pre-compute initial values for UI components
_INITIAL_IMAGE, _INITIAL_ORDER = (None, {})
if _DEFAULT_SCENARIO:
    _scenario = _SCENARIOS.get(_DEFAULT_SCENARIO)
    if _scenario:
        _INITIAL_IMAGE = str(_scenario.image_path) if _scenario.image_path.exists() else None
        _INITIAL_ORDER = _scenario.order_manifest


def load_scenario(name: str) -> Tuple[Optional[str], Dict[str, object]]:
    scenario = _SCENARIOS.get(name)
    if not scenario:
        return None, {"error": f"Scenario '{name}' is not available"}

    image_value = str(scenario.image_path) if scenario.image_path.exists() else None
    if image_value is None:
        logger.warning(f"[APP] Image not found: {scenario.image_path}")
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
            # Using httpx for better connection management
            response = httpx.post(
                f"{API_BASE_URL}/validate",
                files=files,
                data=data,
                timeout=320.0  # 5+ minutes for 7B model inference
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
            
    except httpx.HTTPError as e:
        logger.error(f"[GRADIO] Request failed: {e}")
        # Return default empty values on connection error
        return _default_validation(name), _default_metrics(name)
    except Exception as e:
        logger.exception(f"[GRADIO] Validation error: {e}")
        return _default_validation(name), _default_metrics(name)


def _format_order_ticket(order_manifest: Dict[str, object]) -> str:
    """Format order ticket as beautiful HTML card."""
    if not order_manifest or "error" in order_manifest:
        return '<div style="color: #6c757d; padding: 40px; text-align: center; font-style: italic;">Select a scenario to view order details</div>'
    
    restaurant = order_manifest.get("restaurant", "Unknown")
    table = order_manifest.get("table_number", "?")
    order_id = order_manifest.get("order_id", "N/A")
    items = order_manifest.get("items_ordered", [])
    
    # Build items table rows
    items_rows = ""
    for item in items:
        item_name = item.get("item", "Unknown")
        qty = item.get("quantity", 1)
        items_rows += f'''
        <tr>
            <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef;">{item_name}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; font-weight: 600;">{qty}</td>
        </tr>'''
    
    html = f'''
    <div style="background: linear-gradient(135deg, #0071C5 0%, #00285A 100%); border-radius: 12px; padding: 3px;">
        <div style="background: white; border-radius: 10px; overflow: hidden;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #0071C5 0%, #00285A 100%); color: white; padding: 15px 20px;">
                <div style="font-size: 18px; font-weight: 700; margin-bottom: 4px;">🏪 {restaurant}</div>
                <div style="font-size: 13px; opacity: 0.9;">Table {table} • Order #{order_id}</div>
            </div>
            
            <!-- Items Table -->
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #E6F3FB;">
                        <th style="padding: 12px 15px; text-align: left; font-weight: 600; color: #00285A; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Item</th>
                        <th style="padding: 12px 15px; text-align: center; font-weight: 600; color: #00285A; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;">Qty</th>
                    </tr>
                </thead>
                <tbody>
                    {items_rows}
                </tbody>
            </table>
            
            <!-- Footer -->
            <div style="background: #f8f9fa; padding: 12px 15px; text-align: center; font-size: 12px; color: #6c757d;">
                {len(items)} item(s) ordered
            </div>
        </div>
    </div>
    '''
    return html


def _format_validation_result(validation: Dict[str, object], has_result: bool = False) -> str:
    """Format validation result as beautiful HTML card."""
    if not has_result or not validation or validation.get("order_complete") is None:
        return '''
        <div style="background: #f8f9fa; border-radius: 12px; padding: 40px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 15px;">🔍</div>
            <div style="color: #6c757d; font-size: 14px;">Click <strong>"Validate Plate"</strong> to analyze</div>
        </div>
        '''
    
    is_complete = validation.get("order_complete", False)
    accuracy = validation.get("accuracy_score")
    accuracy_pct = f"{accuracy:.0%}" if accuracy is not None else "N/A"
    
    # Status styling
    if is_complete:
        status_color = "#10b981"
        status_bg = "#d1fae5"
        status_icon = "✅"
        status_text = "Order Complete"
    else:
        status_color = "#ef4444"
        status_bg = "#fee2e2"
        status_icon = "❌"
        status_text = "Order Incomplete"
    
    missing = validation.get("missing_items", [])
    extra = validation.get("extra_items", [])
    
    # Missing items section
    missing_html = ""
    if missing:
        missing_items = "".join([f'<div style="padding: 6px 0; border-bottom: 1px solid #fecaca;">• {item.get("name", "Unknown")} <span style="color: #9ca3af;">(×{item.get("quantity", 1)})</span></div>' for item in missing])
        missing_html = f'''
        <div style="background: #fef2f2; border-radius: 8px; padding: 12px; margin-top: 12px;">
            <div style="font-weight: 600; color: #dc2626; margin-bottom: 8px; font-size: 13px;">⚠️ Missing Items</div>
            <div style="color: #7f1d1d; font-size: 13px;">{missing_items}</div>
        </div>
        '''
    
    # Extra items section
    extra_html = ""
    if extra:
        extra_items = "".join([f'<div style="padding: 6px 0; border-bottom: 1px solid #fed7aa;">• {item.get("name", "Unknown")} <span style="color: #9ca3af;">(×{item.get("quantity", 1)})</span></div>' for item in extra])
        extra_html = f'''
        <div style="background: #fff7ed; border-radius: 8px; padding: 12px; margin-top: 12px;">
            <div style="font-weight: 600; color: #ea580c; margin-bottom: 8px; font-size: 13px;">➕ Extra Items Detected</div>
            <div style="color: #7c2d12; font-size: 13px;">{extra_items}</div>
        </div>
        '''
    
    html = f'''
    <div style="background: white; border-radius: 12px; border: 1px solid #e5e7eb; overflow: hidden;">
        <!-- Status Banner -->
        <div style="background: {status_bg}; padding: 16px 20px; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 28px;">{status_icon}</span>
            <div>
                <div style="font-weight: 700; color: {status_color}; font-size: 16px;">{status_text}</div>
                <div style="color: #6b7280; font-size: 12px;">Accuracy: {accuracy_pct}</div>
            </div>
        </div>
        
        <!-- Details -->
        <div style="padding: 16px 20px;">
            {missing_html}
            {extra_html}
            {'' if missing or extra else '<div style="text-align: center; color: #10b981; padding: 10px;">All items matched correctly! 🎉</div>'}
        </div>
    </div>
    '''
    return html


def _format_metrics_table(metrics: Dict[str, object], has_result: bool = False) -> str:
    """Format performance metrics as beautiful HTML card."""
    if not has_result or not metrics or metrics.get("vlm_inference_ms") is None:
        return '''
        <div style="background: #f8f9fa; border-radius: 12px; padding: 40px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 15px;">📊</div>
            <div style="color: #6c757d; font-size: 14px;">Performance metrics will appear after validation</div>
        </div>
        '''
    
    e2e = metrics.get("end_to_end_latency_ms")
    image_decode = metrics.get("image_decode_ms")
    vlm = metrics.get("vlm_inference_ms")
    semantic = metrics.get("agent_reconciliation_ms")
    within_window = metrics.get("within_operational_window", False)
    cpu = metrics.get("cpu_utilization")
    gpu = metrics.get("gpu_utilization")
    mem = metrics.get("memory_utilization")
    
    # Format values
    e2e_str = f"{e2e:,.0f} ms" if e2e else "N/A"
    decode_str = f"{image_decode:,.0f} ms" if image_decode else "N/A"
    vlm_str = f"{vlm:,.0f} ms" if vlm else "N/A"
    semantic_str = f"{semantic:,.0f} ms" if semantic else "N/A"
    
    # Window status
    window_color = "#10b981" if within_window else "#f59e0b"
    window_icon = "✅" if within_window else "⚠️"
    window_text = "Yes" if within_window else "No"
    
    html = f'''
    <div style="background: white; border-radius: 12px; border: 1px solid #e5e7eb; overflow: hidden;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #0071C5 0%, #00285A 100%); color: white; padding: 14px 20px;">
            <div style="font-weight: 600; font-size: 14px;">⚡ Performance Metrics</div>
        </div>
        
        <!-- Latency Metrics -->
        <div style="padding: 16px 20px; border-bottom: 1px solid #e5e7eb;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; text-align: center;">
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #1f2937;">{e2e_str}</div>
                    <div style="font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">End-to-End</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #059669;">{decode_str}</div>
                    <div style="font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Image Decode</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #0071C5;">{vlm_str}</div>
                    <div style="font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">VLM Inference</div>
                </div>
                <div>
                    <div style="font-size: 20px; font-weight: 700; color: #8b5cf6;">{semantic_str}</div>
                    <div style="font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Semantic Match</div>
                </div>
            </div>
        </div>
        
        <!-- Window Status -->
        <div style="padding: 12px 20px; background: #f8fafc; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 13px; color: #64748b;">Within 2s Operational Window</span>
            <span style="color: {window_color}; font-weight: 600;">{window_icon} {window_text}</span>
        </div>
        
        <!-- Resource Utilization -->
        <div style="padding: 16px 20px;">
            <div style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">Resource Utilization</div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;">
                <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 18px; font-weight: 700; color: #16a34a;">{cpu:.1f}%</div>
                    <div style="font-size: 11px; color: #6b7280;">CPU</div>
                </div>
                <div style="background: #fef3c7; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 18px; font-weight: 700; color: #d97706;">{gpu:.1f}%</div>
                    <div style="font-size: 11px; color: #6b7280;">GPU</div>
                </div>
                <div style="background: #ede9fe; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 18px; font-weight: 700; color: #7c3aed;">{mem:.1f}%</div>
                    <div style="font-size: 11px; color: #6b7280;">Memory</div>
                </div>
            </div>
        </div>
    </div>
    '''
    return html


# Intel Brand Colors
# Primary: #0071C5 (Intel Blue)
# Dark: #00285A (Intel Dark Blue)  
# Light: #00C7FD (Intel Cyan)
# Accent: #E6F3FB (Light Blue Background)

CUSTOM_CSS = """
/* Intel Theme Variables */
:root {
    --primary-500: #0071C5 !important;
    --primary-600: #005A9E !important;
    --primary-700: #00285A !important;
    --neutral-50: #f8fafc;
    --neutral-100: #f1f5f9;
    --neutral-200: #e2e8f0;
}

/* Full Width Layout */
.gradio-container {
    max-width: 100% !important;
    padding: 20px 40px !important;
    margin: 0 !important;
}

/* Hide Gradio Footer Elements */
footer {
    display: none !important;
}

.built-with {
    display: none !important;
}

#footer {
    display: none !important;
}

.gradio-container > footer {
    display: none !important;
}

div[class*="footer"] {
    display: none !important;
}

a[href*="gradio.app"] {
    display: none !important;
}

/* Header Banner - Intel Blue */
.header-banner {
    background: linear-gradient(135deg, #0071C5 0%, #00285A 100%);
    color: white;
    padding: 28px 40px;
    border-radius: 12px;
    margin-bottom: 28px;
    box-shadow: 0 4px 12px rgba(0, 113, 197, 0.3);
}

.header-banner h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.header-banner p {
    margin: 10px 0 0 0;
    opacity: 0.95;
    font-size: 15px;
}

/* Card Styling */
.card-panel {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}

/* Validate Button - Intel Blue */
#validate-btn {
    background: linear-gradient(135deg, #0071C5 0%, #00285A 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 12px rgba(0, 113, 197, 0.3) !important;
    transition: all 0.2s ease !important;
}

#validate-btn:hover {
    background: linear-gradient(135deg, #005A9E 0%, #001d47 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0, 113, 197, 0.4) !important;
}

/* Dropdown Styling */
.scenario-dropdown {
    border-radius: 10px !important;
}

/* Image Display */
.image-container img {
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Accordion Styling */
.accordion {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    overflow: hidden !important;
    margin-top: 16px !important;
}

/* Remove default Gradio spacing */
.gap-4 {
    gap: 20px !important;
}

/* Footer */
.footer-info {
    text-align: center;
    color: #00285A;
    font-size: 13px;
    padding: 20px;
    border-top: 2px solid #E6F3FB;
    margin-top: 32px;
    background: #f8fafc;
}

/* Section Headers */
.section-header {
    font-weight: 600;
    color: #00285A;
    font-size: 14px;
    margin-bottom: 10px;
}

/* Responsive Full Width */
@media (min-width: 1200px) {
    .gradio-container {
        padding: 24px 60px !important;
    }
}
"""

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
    fill_width=True,
) as app:
    
    # Header Banner with Intel Branding
    gr.HTML(f'''
        <div class="header-banner">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">{APP_TITLE}</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.95; font-size: 15px;">{APP_DESCRIPTION} • Powered by OpenVINO™ & Vision-Language Models</p>
                </div>
                <div style="text-align: right; font-size: 13px; opacity: 0.9;">
                    <div style="font-weight: 600; font-size: 16px;">intel</div>
                    <div>AI Solutions</div>
                </div>
            </div>
        </div>
    ''')
    
    # Controls Row
    with gr.Row():
        with gr.Column(scale=3):
            scenario_dropdown = gr.Dropdown(
                label="📋 Select Order Scenario",
                choices=list(_SCENARIOS.keys()),
                value=_DEFAULT_SCENARIO if _DEFAULT_SCENARIO else None,
                interactive=True,
                elem_classes=["scenario-dropdown"],
            )
        with gr.Column(scale=1):
            validate_button = gr.Button(
                "🚀 Validate Plate",
                variant="primary",
                elem_id="validate-btn",
                size="lg",
            )
    
    gr.HTML('<div style="height: 8px;"></div>')
    
    # Main Content Row
    with gr.Row(equal_height=True):
        # Left Column - Image
        with gr.Column(scale=1):
            gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">📸 Plate Image</div>')
            image_display = gr.Image(
                label="",
                value=_INITIAL_IMAGE,
                interactive=False,
                show_label=False,
                elem_classes=["image-container"],
                height=380,
            )
        
        # Right Column - Order Ticket
        with gr.Column(scale=1):
            gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">🎫 Order Ticket</div>')
            order_display = gr.HTML(
                value=_format_order_ticket(_INITIAL_ORDER),
                elem_classes=["card-panel"],
            )
    
    gr.HTML('<div style="height: 20px;"></div>')
    
    # Results Row
    with gr.Row(equal_height=True):
        # Validation Result
        with gr.Column(scale=1):
            gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">✅ Validation Result</div>')
            validation_display = gr.HTML(
                value=_format_validation_result({}, has_result=False),
            )
        
        # Performance Metrics
        with gr.Column(scale=1):
            gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">📊 Performance Metrics</div>')
            metrics_display = gr.HTML(
                value=_format_metrics_table({}, has_result=False),
            )
    
    # Footer with Intel Branding
    gr.HTML('''
        <div class="footer-info">
            <div style="font-weight: 700; color: #0071C5; font-size: 15px; margin-bottom: 4px;">intel</div>
            <div><strong>AI Solutions</strong> • Dine-In Order Accuracy System</div>
            <div style="margin-top: 4px; font-size: 12px; color: #64748b;">Powered by Qwen2.5-VL-7B Vision-Language Model on OpenVINO™</div>
        </div>
    ''')

    def _on_scenario_change(name: str):
        """Handle scenario change - load image and order, reset results."""
        if not name:
            return (
                None,
                _format_order_ticket({}),
                _format_validation_result({}, has_result=False),
                _format_metrics_table({}, has_result=False)
            )
        image_path, order_manifest = load_scenario(name)
        return (
            image_path, 
            _format_order_ticket(order_manifest),
            _format_validation_result({}, has_result=False),
            _format_metrics_table({}, has_result=False)
        )

    def _on_validate(name: str):
        """Handle validation - call API and show results."""
        logger.info(f"[GRADIO] Validate button clicked for: {name}")
        
        if not name:
            name = _DEFAULT_SCENARIO
            logger.info(f"[GRADIO] Using default scenario: {name}")
        
        if not name:
            logger.warning("[GRADIO] No scenario selected")
            return (
                _format_validation_result({}, has_result=False),
                _format_metrics_table({}, has_result=False)
            )
        
        validation, metrics = validate_plate(name)
        logger.info(f"[GRADIO] Validation returned: order_complete={validation.get('order_complete')}")
        
        validation_html = _format_validation_result(validation, has_result=True)
        metrics_html = _format_metrics_table(metrics, has_result=True)
        
        return validation_html, metrics_html

    def _load_initial():
        """Load initial scenario on app start."""
        if _DEFAULT_SCENARIO:
            image_path, order_manifest = load_scenario(_DEFAULT_SCENARIO)
            return (
                _DEFAULT_SCENARIO,
                image_path, 
                _format_order_ticket(order_manifest),
                _format_validation_result({}, has_result=False),
                _format_metrics_table({}, has_result=False)
            )
        return None, None, _format_order_ticket({}), _format_validation_result({}, has_result=False), _format_metrics_table({}, has_result=False)

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
        show_progress="full",
    )

    # Load initial scenario on app start - properly initialize all components including dropdown
    app.load(
        fn=_load_initial,
        outputs=[scenario_dropdown, image_display, order_display, validation_display, metrics_display]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
    )
