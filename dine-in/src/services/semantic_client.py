"""
Semantic Matching Client Service.
Provides fuzzy matching capabilities for item comparison.
"""

import logging
from typing import Dict, Optional
import httpx

logger = logging.getLogger(__name__)


class SemanticMatchResult:
    """Value object for semantic match results"""
    
    def __init__(self, similarity: float, is_match: bool, metadata: Optional[Dict] = None):
        self.similarity = similarity
        self.is_match = is_match
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"SemanticMatchResult(similarity={self.similarity:.2f}, is_match={self.is_match})"


class SemanticClient:
    """
    Semantic matching client for item comparison.
    Uses external semantic service for fuzzy string matching.
    """
    
    def __init__(self, endpoint: str, timeout: int = 10, similarity_threshold: float = 0.7):
        self.endpoint = endpoint
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.compare_endpoint = f"{endpoint}/api/v1/compare/semantic"
        logger.info(f"Semantic Client initialized: endpoint={endpoint}, threshold={similarity_threshold}")
    
    async def match_items(self, expected_item: str, detected_item: str) -> SemanticMatchResult:
        """
        Compare two item names using semantic similarity.
        
        Args:
            expected_item: Expected item name from order
            detected_item: Detected item name from VLM
            
        Returns:
            SemanticMatchResult with similarity score and match decision
        """
        logger.debug(f"Matching items: expected='{expected_item}' vs detected='{detected_item}'")
        
        try:
            payload = {
                "text1": expected_item.lower().strip(),
                "text2": detected_item.lower().strip()
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.compare_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                similarity = result.get("similarity", 0.0)
                is_match = similarity >= self.similarity_threshold
                
                logger.debug(f"Semantic match result: similarity={similarity:.2f}, is_match={is_match}")
                
                return SemanticMatchResult(
                    similarity=similarity,
                    is_match=is_match,
                    metadata=result
                )
                
        except httpx.HTTPError as e:
            logger.warning(f"Semantic service error (falling back to exact match): {e}")
            # Fallback to simple string matching
            exact_match = expected_item.lower().strip() == detected_item.lower().strip()
            return SemanticMatchResult(
                similarity=1.0 if exact_match else 0.0,
                is_match=exact_match,
                metadata={"fallback": True, "error": str(e)}
            )
        except Exception as e:
            logger.exception(f"Unexpected error in semantic matching: {e}")
            # Conservative fallback
            return SemanticMatchResult(similarity=0.0, is_match=False, metadata={"error": str(e)})
    
    async def health_check(self) -> bool:
        """Check if semantic service is available"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.endpoint}/health")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Semantic service health check failed: {e}")
            return False
