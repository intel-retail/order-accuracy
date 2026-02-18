"""
Semantic Matching Client Service.
Provides fuzzy matching capabilities for item comparison.
Includes connection pooling and circuit breaker for reliability.
"""

import logging
import asyncio
from typing import Dict, Optional
from enum import Enum
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class SemanticCircuitBreaker:
    """Circuit breaker for semantic service with configurable thresholds."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 15.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._success_count = 0
                        return True
                return False
            return True  # HALF_OPEN
    
    async def record_success(self):
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= 2:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)
    
    async def record_failure(self):
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN


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
    Semantic matching client with connection pooling and circuit breaker.
    Uses external semantic service for fuzzy string matching.
    
    Features:
    - Shared HTTP connection pool
    - Circuit breaker for fault tolerance
    """
    
    # Class-level HTTP client pool
    _http_client: Optional[httpx.AsyncClient] = None
    _client_lock = asyncio.Lock()
    
    def __init__(self, endpoint: str, timeout: int = 10, similarity_threshold: float = 0.7):
        self.endpoint = endpoint
        self.timeout = timeout
        self.similarity_threshold = similarity_threshold
        self.compare_endpoint = f"{endpoint}/api/v1/compare/semantic"
        
        # Circuit breaker with faster recovery for semantic service
        self._circuit_breaker = SemanticCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=15.0
        )
        
        logger.info(f"Semantic Client initialized: endpoint={endpoint}, threshold={similarity_threshold}, circuit_breaker=enabled")
    
    @classmethod
    async def get_http_client(cls, timeout: int = 10) -> httpx.AsyncClient:
        """Get or create shared HTTP client with connection pooling."""
        if cls._http_client is None or cls._http_client.is_closed:
            async with cls._client_lock:
                if cls._http_client is None or cls._http_client.is_closed:
                    limits = httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                        keepalive_expiry=30.0
                    )
                    cls._http_client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout
                    )
                    logger.info("Created shared HTTP client for semantic service")
        return cls._http_client
    
    @classmethod
    async def close_http_client(cls):
        """Close the shared HTTP client."""
        if cls._http_client is not None:
            await cls._http_client.aclose()
            cls._http_client = None
    
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
        
        # Check circuit breaker
        if not await self._circuit_breaker.can_execute():
            logger.warning("Semantic service circuit breaker is OPEN, using fallback")
            return self._fallback_match(expected_item, detected_item, "circuit_breaker_open")
        
        try:
            payload = {
                "text1": expected_item.lower().strip(),
                "text2": detected_item.lower().strip()
            }
            
            client = await self.get_http_client(self.timeout)
            response = await client.post(
                self.compare_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Record success
            await self._circuit_breaker.record_success()
            
            result = response.json()
            # Use 'confidence' key from semantic service response (not 'similarity')
            # Also use the 'match' field directly if available
            similarity = result.get("confidence", result.get("similarity", 0.0))
            is_match = result.get("match", similarity >= self.similarity_threshold)
            
            logger.debug(f"Semantic match result: similarity={similarity:.2f}, is_match={is_match}")
            
            return SemanticMatchResult(
                similarity=similarity,
                is_match=is_match,
                metadata=result
            )
            
        except httpx.HTTPError as e:
            await self._circuit_breaker.record_failure()
            logger.warning(f"Semantic service error (falling back to exact match): {e}")
            return self._fallback_match(expected_item, detected_item, str(e))
        except Exception as e:
            await self._circuit_breaker.record_failure()
            logger.exception(f"Unexpected error in semantic matching: {e}")
            return SemanticMatchResult(similarity=0.0, is_match=False, metadata={"error": str(e)})
    
    def _fallback_match(self, expected: str, detected: str, reason: str) -> SemanticMatchResult:
        """Fallback to simple string matching when service is unavailable."""
        exact_match = expected.lower().strip() == detected.lower().strip()
        return SemanticMatchResult(
            similarity=1.0 if exact_match else 0.0,
            is_match=exact_match,
            metadata={"fallback": True, "reason": reason}
        )
    
    async def health_check(self) -> bool:
        """Check if semantic service is available."""
        try:
            client = await self.get_http_client(timeout=5)
            response = await client.get(f"{self.endpoint}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Semantic service health check failed: {e}")
            return False
