"""
HTTP client for semantic-comparison-service.
Handles semantic matching via external microservice.
"""

import requests
from typing import List, Dict, Optional
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SemanticComparisonClient:
    """Client for semantic-comparison-service API."""
    
    def __init__(self, endpoint: str, timeout: int = 30):
        """
        Initialize semantic comparison client.
        
        Args:
            endpoint: Base URL of semantic-comparison-service (e.g., http://semantic-service:8080)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout
        self.api_base = f"{self.endpoint}/api/v1"
        logger.info(f"SemanticComparisonClient initialized: endpoint={endpoint}, timeout={timeout}s")
    
    def health_check(self) -> bool:
        """
        Check if semantic service is available.
        
        Returns:
            True if service is healthy, False otherwise
        """
        logger.debug(f"Performing health check on {self.endpoint}")
        try:
            response = requests.get(
                f"{self.api_base}/health",
                timeout=5
            )
            is_healthy = response.status_code == 200
            if is_healthy:
                logger.info(f"Semantic service health check passed: {self.endpoint}")
            else:
                logger.warning(f"Semantic service health check failed with status {response.status_code}")
            return is_healthy
        except Exception as e:
            logger.error(f"Semantic service health check exception: {e}")
            return False
    
    def semantic_match(
        self,
        text1: str,
        text2: str,
        context: str = "grocery products"
    ) -> Dict:
        """
        Perform semantic comparison between two text strings.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            context: Context for matching (default: "grocery products")
        
        Returns:
            Dict with keys: match (bool), confidence (float), reasoning (str), match_type (str)
        """
        logger.debug(f"[SEMANTIC-API] Matching: '{text1}' vs '{text2}' (context={context})")
        try:
            response = requests.post(
                f"{self.api_base}/compare/semantic",
                json={
                    "text1": text1,
                    "text2": text2,
                    "context": context
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[SEMANTIC-API] Match result: '{text1}' ↔ '{text2}' = {result.get('match')} (confidence={result.get('confidence', 0.0):.2f})")
                return result
            else:
                logger.error(f"[SEMANTIC-API] Request failed with status {response.status_code}")
                return {
                    "match": False,
                    "confidence": 0.0,
                    "reasoning": f"API error: {response.status_code}",
                    "match_type": "error"
                }
        
        except Exception as e:
            logger.error(f"[SEMANTIC-API] Exception during semantic match: {e}", exc_info=True)
            return {
                "match": False,
                "confidence": 0.0,
                "reasoning": f"Exception: {str(e)}",
                "match_type": "error"
            }
    
    def validate_order(
        self,
        expected_items: List[Dict],
        detected_items: List[Dict],
        use_semantic: bool = True,
        exact_match_first: bool = True
    ) -> Dict:
        """
        Validate order by comparing expected vs detected items.
        
        Args:
            expected_items: List of dicts with 'name' and 'quantity' keys
            detected_items: List of dicts with 'name' and 'quantity' keys
            use_semantic: Enable semantic matching for unmatched items
            exact_match_first: Try exact match before semantic
        
        Returns:
            Dict with validation results including missing, extra, quantity_mismatch, matched items
        """
        try:
            response = requests.post(
                f"{self.api_base}/compare/order",
                json={
                    "expected_items": expected_items,
                    "detected_items": detected_items,
                    "options": {
                        "use_semantic": use_semantic,
                        "exact_match_first": exact_match_first,
                        "case_insensitive": True
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[SemanticClient] Order validation failed: {response.status_code}", flush=True)
                # Return fallback structure
                return {
                    "status": "error",
                    "validation": {
                        "missing": expected_items,
                        "extra": detected_items,
                        "quantity_mismatch": [],
                        "matched": []
                    },
                    "metrics": {
                        "total_expected": len(expected_items),
                        "total_detected": len(detected_items),
                        "exact_matches": 0,
                        "semantic_matches": 0,
                        "processing_time_ms": 0.0
                    }
                }
        
        except Exception as e:
            print(f"[SemanticClient] Order validation exception: {e}", flush=True)
            return {
                "status": "error",
                "validation": {
                    "missing": expected_items,
                    "extra": detected_items,
                    "quantity_mismatch": [],
                    "matched": []
                },
                "metrics": {
                    "total_expected": len(expected_items),
                    "total_detected": len(detected_items),
                    "exact_matches": 0,
                    "semantic_matches": 0,
                    "processing_time_ms": 0.0
                }
            }
    
    def validate_inventory(
        self,
        items: List[str],
        inventory: Optional[List[str]] = None,
        use_semantic: bool = True
    ) -> Dict:
        """
        Validate if items exist in inventory.
        
        Args:
            items: List of item names to validate
            inventory: Inventory list (None to use service's configured inventory)
            use_semantic: Enable semantic matching
        
        Returns:
            Dict with validation results and summary
        """
        try:
            payload = {
                "items": items,
                "options": {
                    "use_semantic": use_semantic,
                    "case_insensitive": True
                }
            }
            
            if inventory is not None:
                payload["inventory"] = inventory
            
            response = requests.post(
                f"{self.api_base}/compare/inventory",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[SemanticClient] Inventory validation failed: {response.status_code}", flush=True)
                return {
                    "results": [],
                    "summary": {
                        "total_items": len(items),
                        "matched": 0,
                        "unmatched": len(items),
                        "processing_time_ms": 0.0
                    }
                }
        
        except Exception as e:
            print(f"[SemanticClient] Inventory validation exception: {e}", flush=True)
            return {
                "results": [],
                "summary": {
                    "total_items": len(items),
                    "matched": 0,
                    "unmatched": len(items),
                    "processing_time_ms": 0.0
                }
            }
