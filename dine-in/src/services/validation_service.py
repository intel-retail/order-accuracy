"""
Validation Service implementing business logic for plate validation.
Uses Strategy pattern for validation algorithms.
"""

import time
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from .vlm_client import VLMClient, VLMResponse
from .semantic_client import SemanticClient, SemanticMatchResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for validation performance tracking"""
    vlm_inference_time_ms: float
    semantic_matching_time_ms: float
    total_validation_time_ms: float
    items_processed: int
    matches_found: int
    items_detected: int = 0
    tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vlm_inference_ms": round(self.vlm_inference_time_ms, 2),
            "semantic_matching_ms": round(self.semantic_matching_time_ms, 2),
            "total_time_ms": round(self.total_validation_time_ms, 2),
            "items_processed": self.items_processed,
            "matches_found": self.matches_found,
            "items_detected": self.items_detected,
            "tps": round(self.tps, 2),
            "tokens": self.prompt_tokens + self.completion_tokens
        }


@dataclass
class ItemMatch:
    """Represents a match between expected and detected items"""
    expected_name: str
    expected_quantity: int
    detected_name: str
    detected_quantity: int
    similarity: float
    is_exact_match: bool
    quantity_match: bool


@dataclass
class ValidationResult:
    """Complete validation result with analysis"""
    image_id: str
    order_complete: bool
    accuracy_score: float
    missing_items: List[Dict[str, Any]] = field(default_factory=list)
    extra_items: List[Dict[str, Any]] = field(default_factory=list)
    quantity_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    matched_items: List[Dict[str, Any]] = field(default_factory=list)
    metrics: ValidationMetrics = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "image_id": self.image_id,
            "order_complete": self.order_complete,
            "accuracy_score": round(self.accuracy_score, 2),
            "missing_items": self.missing_items,
            "extra_items": self.extra_items,
            "quantity_mismatches": self.quantity_mismatches,
            "matched_items": self.matched_items
        }
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        return result


class ValidationStrategy:
    """Abstract validation strategy"""
    
    async def validate(
        self,
        detected_items: List[Dict[str, Any]],
        expected_items: List[Dict[str, Any]],
        semantic_client: SemanticClient
    ) -> Tuple[List[ItemMatch], List[Dict], List[Dict]]:
        """
        Validate detected items against expected items.
        
        Returns:
            Tuple of (matches, missing_items, extra_items)
        """
        raise NotImplementedError


class SemanticValidationStrategy(ValidationStrategy):
    """Validation strategy using semantic matching"""
    
    async def validate(
        self,
        detected_items: List[Dict[str, Any]],
        expected_items: List[Dict[str, Any]],
        semantic_client: SemanticClient
    ) -> Tuple[List[ItemMatch], List[Dict], List[Dict]]:
        """Perform semantic matching between expected and detected items"""
        
        matches: List[ItemMatch] = []
        missing_items: List[Dict] = []
        extra_items: List[Dict] = []
        
        # Track which detected items have been matched
        detected_used = [False] * len(detected_items)
        
        # For each expected item, find best match in detected items
        for expected in expected_items:
            expected_name = expected.get("name", expected.get("item", ""))
            expected_qty = int(expected.get("quantity", 1))
            
            best_match: Optional[Tuple[int, SemanticMatchResult]] = None
            best_similarity = 0.0
            
            # Find best semantic match
            for idx, detected in enumerate(detected_items):
                if detected_used[idx]:
                    continue
                
                # Support both 'name' and 'item' keys (VLM may return either)
                detected_name = detected.get("name", detected.get("item", ""))
                match_result = await semantic_client.match_items(expected_name, detected_name)
                
                if match_result.is_match and match_result.similarity > best_similarity:
                    best_similarity = match_result.similarity
                    best_match = (idx, match_result)
            
            # Process match result
            if best_match:
                idx, match_result = best_match
                detected = detected_items[idx]
                detected_used[idx] = True
                
                detected_qty = int(detected.get("quantity", 1))
                
                match = ItemMatch(
                    expected_name=expected_name,
                    expected_quantity=expected_qty,
                    detected_name=detected.get("name", detected.get("item", "")),
                    detected_quantity=detected_qty,
                    similarity=match_result.similarity,
                    is_exact_match=match_result.similarity >= 0.95,
                    quantity_match=expected_qty == detected_qty
                )
                matches.append(match)
                
                logger.debug(f"Matched: {expected_name} -> {detected.get('name', detected.get('item', ''))} "
                           f"(similarity={match_result.similarity:.2f})")
            else:
                # No match found - item is missing
                missing_items.append({
                    "name": expected_name,
                    "quantity": expected_qty
                })
                logger.debug(f"Missing item: {expected_name} (qty={expected_qty})")
        
        # Remaining detected items are extras
        for idx, detected in enumerate(detected_items):
            if not detected_used[idx]:
                extra_items.append({
                    "name": detected.get("name", detected.get("item", "")),
                    "quantity": int(detected.get("quantity", 1))
                })
                logger.debug(f"Extra item: {detected.get('name', detected.get('item', ''))}")
        
        return matches, missing_items, extra_items


class ValidationService:
    """
    Core validation service orchestrating VLM and semantic matching.
    Implements Facade pattern to simplify complex subsystem interactions.
    """
    
    def __init__(
        self,
        vlm_client: VLMClient,
        semantic_client: SemanticClient,
        strategy: ValidationStrategy = None
    ):
        self.vlm_client = vlm_client
        self.semantic_client = semantic_client
        self.strategy = strategy or SemanticValidationStrategy()
        logger.info("Validation Service initialized")
    
    def _calculate_accuracy(
        self,
        matches: List[ItemMatch],
        expected_count: int,
        detected_count: int
    ) -> float:
        """
        Calculate validation accuracy score.
        
        Formula: (correct_items + correct_quantities) / (2 * total_expected_items)
        Range: 0.0 to 1.0
        """
        if expected_count == 0:
            return 1.0 if detected_count == 0 else 0.0
        
        # Count correct items (semantic matches)
        correct_items = len(matches)
        
        # Count correct quantities
        correct_quantities = sum(1 for m in matches if m.quantity_match)
        
        # Weighted score: 60% for item presence, 40% for quantity correctness
        item_score = correct_items / expected_count
        quantity_score = correct_quantities / expected_count if matches else 0.0
        
        accuracy = 0.6 * item_score + 0.4 * quantity_score
        
        logger.debug(f"Accuracy calculation: items={correct_items}/{expected_count}, "
                    f"quantities={correct_quantities}/{expected_count}, "
                    f"score={accuracy:.2f}")
        
        return min(accuracy, 1.0)
    
    async def validate_plate(
        self,
        image_bytes: bytes,
        order_manifest: Dict[str, Any],
        image_id: str,
        request_id: str = None
    ) -> ValidationResult:
        """
        Validate plate against order manifest using VLM and semantic matching.
        
        Args:
            image_bytes: Raw image data
            order_manifest: Expected order items
            image_id: Unique identifier for this validation
            request_id: Optional unique request identifier for tracking
            
        Returns:
            Complete ValidationResult with metrics
        """
        logger.info(f"Starting plate validation: image_id={image_id}, request_id={request_id}")
        start_time = time.time()
        
        try:
            # Step 1: VLM Inference
            vlm_start = time.time()
            vlm_response: VLMResponse = await self.vlm_client.analyze_plate(image_bytes, request_id=request_id)
            vlm_time_ms = (time.time() - vlm_start) * 1000
            logger.info(f"VLM inference completed in {vlm_time_ms:.2f}ms for {request_id}, "
                       f"detected {len(vlm_response.detected_items)} items")
            
            # Extract expected items from order manifest
            expected_items = order_manifest.get("items", [])
            logger.debug(f"Expected items: {expected_items}")
            logger.debug(f"Detected items: {vlm_response.detected_items}")
            
            # Step 2: Semantic Matching
            semantic_start = time.time()
            matches, missing_items, extra_items = await self.strategy.validate(
                vlm_response.detected_items,
                expected_items,
                self.semantic_client
            )
            semantic_time_ms = (time.time() - semantic_start) * 1000
            logger.info(f"Semantic matching completed in {semantic_time_ms:.2f}ms")
            
            # Step 3: Analyze quantity mismatches
            quantity_mismatches = []
            matched_items_list = []
            
            for match in matches:
                matched_items_list.append({
                    "expected_name": match.expected_name,
                    "detected_name": match.detected_name,
                    "similarity": round(match.similarity, 2),
                    "quantity": match.detected_quantity
                })
                
                if not match.quantity_match:
                    quantity_mismatches.append({
                        "item": match.expected_name,
                        "expected_quantity": match.expected_quantity,
                        "detected_quantity": match.detected_quantity
                    })
            
            # Step 4: Calculate accuracy and determine order completeness
            accuracy_score = self._calculate_accuracy(
                matches,
                len(expected_items),
                len(vlm_response.detected_items)
            )
            
            order_complete = (
                len(missing_items) == 0 and
                len(quantity_mismatches) == 0 and
                len(extra_items) == 0
            )
            
            # Step 5: Build metrics with VLM performance data
            total_time_ms = (time.time() - start_time) * 1000
            
            # Extract VLM performance metrics if available
            vlm_perf = getattr(vlm_response, 'performance_metadata', {}) or {}
            
            metrics = ValidationMetrics(
                vlm_inference_time_ms=vlm_time_ms,
                semantic_matching_time_ms=semantic_time_ms,
                total_validation_time_ms=total_time_ms,
                items_processed=len(expected_items),
                matches_found=len(matches),
                items_detected=len(vlm_response.detected_items),
                tps=vlm_perf.get('tokens_per_second', 0.0),
                prompt_tokens=vlm_perf.get('prompt_tokens', 0),
                completion_tokens=vlm_perf.get('completion_tokens', 0)
            )
            
            result = ValidationResult(
                image_id=image_id,
                order_complete=order_complete,
                accuracy_score=accuracy_score,
                missing_items=missing_items,
                extra_items=extra_items,
                quantity_mismatches=quantity_mismatches,
                matched_items=matched_items_list,
                metrics=metrics
            )
            
            logger.info(f"Validation completed: image_id={image_id}, "
                       f"accuracy={accuracy_score:.2f}, "
                       f"complete={order_complete}, "
                       f"total_time={total_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.exception(f"Validation failed for image_id={image_id}: {e}")
            raise
