"""
Inventory Narrowing Utility

Narrows the full inventory list to only include items relevant to a specific order.
This reduces VLM cognitive load when catalogs are large (10k+ items).

Architecture:
1. Expected items: Items required by the order (from orders.json)
2. Confusable neighbors: Items in same category or with similar names
3. Aliases: Alternative names from inventory.json metadata
4. Fallback set: Full inventory if narrowing fails or order not found
"""

import json
import logging
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)


def load_inventory_metadata(inventory_json_path: str) -> Dict[str, Dict]:
    """
    Load inventory metadata with aliases and categories.
    
    Args:
        inventory_json_path: Path to inventory.json file
        
    Returns:
        Dict mapping item names to their metadata (category, aliases, id)
        Example: {
            "green apple": {"id": "001", "category": "fruit", "aliases": ["granny smith apple", ...]},
            ...
        }
    """
    try:
        with open(inventory_json_path, "r") as f:
            data = json.load(f)
        
        # Convert items list to name-keyed dict for faster lookup
        metadata = {}
        for item in data.get("items", []):
            name = item.get("name", "").lower()
            if name:
                metadata[name] = {
                    "id": item.get("id", ""),
                    "category": item.get("category", "unknown"),
                    "aliases": [alias.lower() for alias in item.get("aliases", [])]
                }
        
        logger.debug(f"Loaded metadata for {len(metadata)} items from {inventory_json_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load inventory metadata from {inventory_json_path}: {e}")
        return {}


def narrow_inventory(
    full_inventory: List[str],
    expected_items: List[Dict],
    inventory_metadata: Optional[Dict[str, Dict]] = None,
    include_confusable: bool = True,
    max_size: int = 50
) -> List[str]:
    """
    Narrow inventory to items relevant to a specific order.
    
    Strategy:
    1. Include all expected items (exact match from order)
    2. Include confusable neighbors (same category, similar names)
    3. Include aliases (alternative names for expected items)
    4. Cap at max_size to avoid prompt bloat
    
    Args:
        full_inventory: Complete inventory list from config
        expected_items: Items expected in order (from orders.json)
                       Format: [{"name": "apple", "quantity": 2}, ...]
        inventory_metadata: Metadata with aliases and categories (from inventory.json)
        include_confusable: Whether to include items in same category
        max_size: Maximum narrowed inventory size
        
    Returns:
        Narrowed inventory list suitable for VLM prompt
    """
    
    if not expected_items:
        logger.warning("No expected items provided; returning full inventory")
        return full_inventory
    
    # Convert full inventory to lowercase for matching
    full_inventory_lower = {item.lower(): item for item in full_inventory}
    
    # Initialize result with all expected items
    narrowed = set()
    expected_names = set()
    
    # Stage 1: Add all expected items (with fuzzy matching)
    for item in expected_items:
        item_name = item.get("name", "").lower()
        if not item_name:
            continue
            
        expected_names.add(item_name)
        
        # Try exact match
        if item_name in full_inventory_lower:
            narrowed.add(full_inventory_lower[item_name])
        else:
            # Try substring match (e.g., "water" matches "water bottle")
            matches = [
                inv_item for inv_lower, inv_item in full_inventory_lower.items()
                if item_name in inv_lower or inv_lower in item_name
            ]
            if matches:
                narrowed.add(matches[0])  # Add first match
                logger.debug(f"Fuzzy matched '{item_name}' to '{matches[0]}'")
            else:
                logger.warning(f"Expected item '{item_name}' not found in inventory")
    
    # Stage 2: Add confusable neighbors (same category)
    if include_confusable and inventory_metadata:
        for item in expected_items:
            item_name = item.get("name", "").lower()
            if not item_name or item_name not in inventory_metadata:
                continue
            
            category = inventory_metadata[item_name].get("category", "")
            if not category:
                continue
            
            # Find all items in same category
            neighbors = [
                inv_lower for inv_lower, meta in inventory_metadata.items()
                if meta.get("category") == category and inv_lower not in expected_names
            ]
            
            # Add up to 2 neighbors per expected item to limit growth
            for neighbor_lower in neighbors[:2]:
                # Map back to original inventory capitalization
                if neighbor_lower in full_inventory_lower:
                    narrowed.add(full_inventory_lower[neighbor_lower])
    
    # Stage 3: Add aliases for all narrowed items
    if inventory_metadata:
        items_to_check = list(narrowed)
        for item in items_to_check:
            item_lower = item.lower()
            if item_lower in inventory_metadata:
                aliases = inventory_metadata[item_lower].get("aliases", [])
                for alias in aliases:
                    # Try to find alias in full inventory
                    for inv_lower, inv_item in full_inventory_lower.items():
                        if alias in inv_lower:
                            narrowed.add(inv_item)
                            break
    
    # Convert to sorted list for consistent ordering
    result = sorted(list(narrowed))
    
    # Cap size
    if len(result) > max_size:
        logger.warning(f"Narrowed inventory size {len(result)} exceeds max {max_size}; truncating")
        result = result[:max_size]
    
    logger.info(
        f"Narrowed inventory from {len(full_inventory)} to {len(result)} items "
        f"(expected: {len(expected_names)}, confusable: {len(result) - len(expected_names)})"
    )
    logger.debug(f"Narrowed inventory: {result}")
    
    return result


def build_narrowed_inventory_text(
    full_inventory: List[str],
    expected_items: List[Dict],
    inventory_metadata: Optional[Dict[str, Dict]] = None,
    fallback_to_full: bool = True
) -> str:
    """
    Build inventory text for VLM prompt using narrowed list.
    
    Args:
        full_inventory: Complete inventory list
        expected_items: Expected items from order
        inventory_metadata: Optional metadata for aliases
        fallback_to_full: If narrowing fails, use full inventory
        
    Returns:
        Formatted inventory text for prompt (newline-separated list with dashes)
    """
    try:
        # Try to narrow inventory
        if expected_items:
            narrowed = narrow_inventory(full_inventory, expected_items, inventory_metadata)
        else:
            narrowed = full_inventory
        
        # Format as bullet list for prompt
        inventory_text = "\n".join(f"- {item}" for item in narrowed)
        return inventory_text
        
    except Exception as e:
        logger.error(f"Error building narrowed inventory: {e}")
        if fallback_to_full:
            logger.warning("Falling back to full inventory")
            return "\n".join(f"- {item}" for item in full_inventory)
        else:
            raise
