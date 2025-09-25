import json
import os
from vlm import save_to_minio
from config import logger
from rapidfuzz import fuzz
from collections import defaultdict

class OrderValidator:
    # Static file paths
    ORDERS_FILE = "./agent_configs/orders.json"
    ADDONS_FILE = "./agent_configs/addons.json"
    MINIO_BUCKET = "order-accuracy-validate-results"

    def __init__(self):
        # Load configs at init
        with open(self.ORDERS_FILE, "r") as f:
            self.orders_payload = json.load(f)

        with open(self.ADDONS_FILE, "r") as f:
            raw_addons = json.load(f)
            # Normalize addons payload for case-insensitive checks
            self.addons_payload = {
                k.strip().lower(): [a.strip().lower() for a in v]
                for k, v in raw_addons.items()
            }

    def validate_order(self, detection_output: dict) -> dict:
        """
        Validate detected output against expected orders and required addons.
        """
        bill_number = str(detection_output.get("BillNumber"))
        detected_items = detection_output.get("Items", [])

        # Convert to lowercase for case-insensitive matching
        detected_items = [
            {"name": item["Name"].lower(), "count": item["Count"]}
            for item in detected_items
        ]

        # Find the expected order
        expected_order = next(
            (order for order in self.orders_payload if order["bill_number"] == bill_number),
            None
        )

        if not expected_order:
            return {"error": f"Bill Number {bill_number} not found in config"}

        expected_items = [
            {"name": item["name"].lower(), "count": item["count"]}
            for item in expected_order.get("items", [])
        ]

        # Compare items
        missing_items, extra_items, quantity_mismatches = [], [], []

        expected_dict = {i["name"]: i["count"] for i in expected_items}
        detected_dict = {i["name"]: i["count"] for i in detected_items}
        
        items_to_replace = []
        # compare two dicts ecpected and detected, if item is not detected in expected. then do a token_set_ratio and replace the item if score goes beyond 95%
        for name, qty in expected_dict.items():
            if name not in detected_dict:
                # If the item is not found, check for similar items
                best_match = max(detected_dict.keys(), key=lambda x: fuzz.token_set_ratio(x, name), default="")
                score = fuzz.token_set_ratio(best_match, name)
                if score > 95:
                    logger.info(f"Replacing expected_dict item '{name}' with detected_dict item '{best_match}' (similarity score: {score}%)")
                    items_to_replace.append((name, best_match, qty))
                else:
                    logger.debug(f"No suitable replacement found for expected_dict item '{name}' (best match: '{best_match}', score: {score}%)")
                    
        
        # Apply replacements after iteration
        for old_name, new_name, qty in items_to_replace:
            del expected_dict[old_name]
            expected_dict[new_name] = qty
            logger.info(f"Applied replacement: '{old_name}' -> '{new_name}' with quantity {qty}")
            
        
        normalized_addons = defaultdict(list)
        for addon_key, addon_values in self.addons_payload.items():
            actual_key = addon_key
            if addon_key not in detected_dict:
                # Find closest match in detected_dict
                best_match = max(detected_dict.keys(), key=lambda x: fuzz.token_set_ratio(x, addon_key), default="")
                score = fuzz.token_set_ratio(best_match, addon_key)
                if score > 95:
                    actual_key = best_match
                    normalized_addons[best_match] = addon_values.copy()
                    logger.info(f"Normalized addon key '{addon_key}' to detected item '{best_match}' (similarity score: {score}%)")
                else:
                    continue  # Skip this addon if no good match found
            else:
                # If exact match found, use it
                normalized_addons[addon_key] = addon_values.copy()
                
            # Normalize the addon_values here with same fuzzy logic as keys
            for i, addon_value in enumerate(normalized_addons[actual_key]):
                best_match = max(detected_dict.keys(), key=lambda x: fuzz.token_set_ratio(x, addon_value), default="")
                score = fuzz.token_set_ratio(best_match, addon_value)
                if score > 95:
                    normalized_addons[actual_key][i] = best_match
                    logger.info(f"Normalized addon value '{addon_value}' to detected item '{best_match}' (similarity score: {score}%)")

        # Update self.addons_payload with normalized values
        self.addons_payload = normalized_addons

        # Missing or quantity mismatched
        for name, qty in expected_dict.items():
            if name not in detected_dict:
                missing_items.append(name)
            elif detected_dict[name] != qty:
                quantity_mismatches.append({
                    "item": name.title(),
                    "expected": qty,
                    "detected": detected_dict[name]
                })

        # Build a set of all valid addon items (values from addons_payload)
        all_valid_addons = set()
        for addon_list in self.addons_payload.values():
            all_valid_addons.update(addon_list)

        # Extras - only flag items that user didn't pay for AND are not legitimate addons
        for name in detected_dict:
            if name not in expected_dict and name not in all_valid_addons:
                extra_items.append(name.title())
    

        # Check required addons
        missing_addons = []
        seen_missing = set()  # Prevent duplicates
        for item in detected_dict:
            if item in self.addons_payload:
                required = self.addons_payload[item]
                for addon in required:
                    # Check if the required addon exists in detected items
                    if addon not in detected_dict and (item, addon) not in seen_missing:
                        missing_addons.append({
                            "item": item.title(),
                            "required_addon": addon.title()
                        })
                        seen_missing.add((item, addon))

        validate_payload =  {
            "bill_number": bill_number,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "count_mismatches": quantity_mismatches,
            "missing_addons": missing_addons
        }
        try:
            logger.info(f"Adding validation results to MinIO: {validate_payload}")
            save_to_minio(
                order_id=bill_number,
                data=validate_payload,
                bucket=self.MINIO_BUCKET
            )
        except Exception as e:
            # use logger instead of print for consistency
            logger.error(f"Error saving validation results to MinIO: {e}")
        return validate_payload
