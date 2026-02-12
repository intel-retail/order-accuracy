"""
Validation agent for order comparison.
Supports both local semantic matching and external semantic-comparison-service.
"""

import os
import logging
from .semantic_matcher import semantic_match
from .semantic_client import SemanticComparisonClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize semantic service client if enabled
SEMANTIC_SERVICE_ENDPOINT = os.getenv("SEMANTIC_SERVICE_ENDPOINT", "http://oa_semantic_service:8080")
USE_SEMANTIC_SERVICE = os.getenv("USE_SEMANTIC_SERVICE", "false").lower() == "true"

logger.info(f"Semantic service configuration: enabled={USE_SEMANTIC_SERVICE}, endpoint={SEMANTIC_SERVICE_ENDPOINT}")

semantic_client = None
if USE_SEMANTIC_SERVICE:
    semantic_client = SemanticComparisonClient(SEMANTIC_SERVICE_ENDPOINT)
    if semantic_client.health_check():
        logger.info(f"Semantic service connected successfully: {SEMANTIC_SERVICE_ENDPOINT}")
    else:
        logger.warning(f"Semantic service unavailable, falling back to local matching")
        semantic_client = None

def semantic_match_wrapper(vlm_pipeline, expected_name, detected_name, transaction_id=None):
    """
    Wrapper for semantic matching that tries external service first, then falls back to local.
    
    Args:
        vlm_pipeline: VLM pipeline for local semantic matching
        expected_name: Expected product name
        detected_name: Detected product name
        transaction_id: Unique transaction ID for logging
    
    Returns:
        bool: True if items match semantically
    """
    tx_prefix = f"[{transaction_id}] " if transaction_id else ""
    logger.debug(f"{tx_prefix}[SEMANTIC] Matching: '{expected_name}' vs '{detected_name}'")
    
    # Try semantic service first if enabled and available
    if semantic_client:
        try:
            result = semantic_client.semantic_match(
                text1=expected_name,
                text2=detected_name,
                context="grocery products"
            )
            match_result = result["match"]
            confidence = result.get('confidence', 0.0)
            logger.info(f"{tx_prefix}[SEMANTIC-SERVICE] Match result: '{expected_name}' ↔ '{detected_name}' = {match_result} (confidence={confidence:.2f})")
            return match_result
        except Exception as e:
            logger.error(f"{tx_prefix}[SEMANTIC-SERVICE] Error occurred, falling back to local matching: {e}")
    
    # Fallback to local semantic matching
    logger.debug(f"{tx_prefix}[SEMANTIC-LOCAL] Using local matching for '{expected_name}' vs '{detected_name}'")
    result = semantic_match(vlm_pipeline, expected_name, detected_name)
    logger.info(f"{tx_prefix}[SEMANTIC-LOCAL] Match result: '{expected_name}' ↔ '{detected_name}' = {result}")
    return result

def validate_order(expected_items, detected_items, vlm_pipeline, transaction_id=None):
    tx_prefix = f"[{transaction_id}] " if transaction_id else ""
    logger.info(f"{tx_prefix}[VALIDATION] Starting order validation: {len(expected_items)} expected, {len(detected_items)} detected")
    logger.debug(f"{tx_prefix}[VALIDATION] Expected items: {expected_items}")
    logger.debug(f"{tx_prefix}[VALIDATION] Detected items: {detected_items}")
    
    missing = []
    extra = []
    quantity_mismatch = []
    matched_detected = set()

    # ---- Pass 1: exact match ----
    logger.debug(f"{tx_prefix}[VALIDATION] Pass 1: Exact name matching")
    for exp in expected_items:
        exp_name = exp["name"].lower()
        exp_qty = exp["quantity"]

        found = False
        for det in detected_items:
            det_name = det["name"].lower()
            det_qty = det["quantity"]

            if det_name == exp_name:
                found = True
                matched_detected.add(det_name)
                logger.debug(f"{tx_prefix}[VALIDATION] Exact match found: '{exp_name}'")
                if det_qty != exp_qty:
                    logger.info(f"{tx_prefix}[VALIDATION] Quantity mismatch: '{exp_name}' expected={exp_qty}, detected={det_qty}")
                    quantity_mismatch.append({
                        "name": exp_name,
                        "expected": exp_qty,
                        "detected": det_qty
                    })
                break

        if not found:
            logger.debug(f"{tx_prefix}[VALIDATION] No exact match for '{exp_name}', will try semantic matching")
            missing.append(exp)

    logger.info(f"{tx_prefix}[VALIDATION] Pass 1 complete: {len(missing)} items need semantic matching")

    # ---- Pass 2: semantic match on leftovers ----
    if missing:
        logger.debug(f"{tx_prefix}[VALIDATION] Pass 2: Semantic matching for unmatched items")
        still_missing = []

        for exp in missing:
            exp_name = exp["name"].lower()
            exp_qty = exp["quantity"]
            matched = False

            for det in detected_items:
                det_name = det["name"].lower()
                if det_name in matched_detected:
                    continue

                if semantic_match_wrapper(vlm_pipeline, exp_name, det_name, transaction_id):
                    matched = True
                    matched_detected.add(det_name)
                    logger.info(f"{tx_prefix}[VALIDATION] Semantic match found: '{exp_name}' matched with '{det_name}'")
                    if det["quantity"] != exp_qty:
                        logger.info(f"{tx_prefix}[VALIDATION] Quantity mismatch after semantic match: '{exp_name}' expected={exp_qty}, detected={det['quantity']}")
                        quantity_mismatch.append({
                            "name": exp_name,
                            "expected": exp_qty,
                            "detected": det["quantity"]
                        })
                    break

            if not matched:
                logger.warning(f"{tx_prefix}[VALIDATION] No match found for '{exp_name}' (exact or semantic)")
                still_missing.append(exp)

        missing = still_missing
        logger.info(f"{tx_prefix}[VALIDATION] Pass 2 complete: {len(missing)} items truly missing")

    # ---- Extras ----
    logger.debug(f"{tx_prefix}[VALIDATION] Checking for extra detected items")
    for det in detected_items:
        if det["name"].lower() not in matched_detected:
            logger.info(f"{tx_prefix}[VALIDATION] Extra item detected: '{det['name']}' (not in expected order)")
            extra.append(det)

    validation_result = {
        "missing": missing,
        "extra": extra,
        "quantity_mismatch": quantity_mismatch
    }
    
    logger.info(f"{tx_prefix}[VALIDATION] Complete: missing={len(missing)}, extra={len(extra)}, qty_mismatch={len(quantity_mismatch)}")
    logger.debug(f"{tx_prefix}[VALIDATION] Full result: {validation_result}")
    
    return validation_result
