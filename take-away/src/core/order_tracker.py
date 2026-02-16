"""
Order Tracker Module - Robust Frame Accumulation and Order Finalization

This module implements intelligent order tracking with:
- Configurable timeouts and collection windows
- Grace period frame collection after OCR stops detecting
- Quality-based frame scoring (not hard filtering)
- Diversity-based frame selection strategy
- Thread-safe implementation for async workers


Date: 2026-02-16
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import heapq

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OrderTrackerConfig:
    """
    Configuration for order tracking behavior.
    
    Attributes:
        inactivity_timeout_seconds: Time without new frames before considering finalization
        minimum_collection_time_seconds: Minimum time to track an order before allowing finalization
        grace_frame_count: Number of additional frames to collect after OCR stops detecting order
        top_k_frames: Maximum number of frames to select for VLM processing
        min_frames_soft_requirement: Soft minimum - warn if below, but don't block processing
        min_frames_hard_requirement: Hard minimum - skip orders with fewer frames
        occlusion_penalty: Quality score penalty for frames with hands/persons (0.0-1.0)
        blur_penalty: Quality score penalty for blurry frames (0.0-1.0)
        item_count_weight: Weight for item count in quality scoring (higher = more items preferred)
    """
    inactivity_timeout_seconds: float = 5.0
    minimum_collection_time_seconds: float = 2.0
    grace_frame_count: int = 3
    top_k_frames: int = 3
    min_frames_soft_requirement: int = 3
    min_frames_hard_requirement: int = 1
    occlusion_penalty: float = 0.3
    blur_penalty: float = 0.2
    item_count_weight: float = 0.1
    
    @classmethod
    def from_dict(cls, config: dict) -> 'OrderTrackerConfig':
        """Create config from dictionary (e.g., from YAML)"""
        return cls(
            inactivity_timeout_seconds=config.get('inactivity_timeout_seconds', 5.0),
            minimum_collection_time_seconds=config.get('minimum_collection_time_seconds', 2.0),
            grace_frame_count=config.get('grace_frame_count', 3),
            top_k_frames=config.get('top_k_frames', 3),
            min_frames_soft_requirement=config.get('min_frames_soft_requirement', 3),
            min_frames_hard_requirement=config.get('min_frames_hard_requirement', 1),
            occlusion_penalty=config.get('occlusion_penalty', 0.3),
            blur_penalty=config.get('blur_penalty', 0.2),
            item_count_weight=config.get('item_count_weight', 0.1),
        )


class OrderState(Enum):
    """Order lifecycle states"""
    COLLECTING = "collecting"       # Actively receiving frames
    GRACE_PERIOD = "grace_period"   # OCR stopped detecting, collecting grace frames
    READY = "ready"                 # Ready for finalization
    FINALIZED = "finalized"         # Processing complete
    SKIPPED = "skipped"             # Skipped due to insufficient frames


@dataclass
class FrameData:
    """
    Captured frame with metadata for quality scoring.
    
    Attributes:
        key: Storage key (e.g., MinIO object key)
        timestamp: Capture timestamp
        quality_score: Computed quality score (0.0-1.0, higher is better)
        has_occlusion: True if hands/persons detected
        item_count: Number of detected items (from YOLO)
        is_blurry: True if frame is blurry
        
    Note: raw_image is NOT stored to prevent memory leaks. Frames should be
    loaded from MinIO when needed for processing.
    """
    key: str
    timestamp: float
    quality_score: float = 1.0
    has_occlusion: bool = False
    item_count: int = 0
    is_blurry: bool = False
    
    def __lt__(self, other):
        """For heap operations - higher quality comes first"""
        return self.quality_score > other.quality_score


@dataclass
class TrackedOrder:
    """
    State for a tracked order.
    
    Maintains full history of frames and timing information
    for intelligent finalization decisions.
    """
    order_id: str
    station_id: str
    state: OrderState = OrderState.COLLECTING
    
    # Timing
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    last_ocr_detection_timestamp: float = field(default_factory=time.time)
    
    # Frame collection
    frames: List[FrameData] = field(default_factory=list)
    grace_frames_remaining: int = 0
    
    # Statistics
    total_frames_received: int = 0
    clean_frames_count: int = 0
    occluded_frames_count: int = 0
    
    @property
    def frame_count(self) -> int:
        return len(self.frames)
    
    @property
    def collection_duration(self) -> float:
        return self.last_seen_timestamp - self.first_seen_timestamp
    
    @property
    def time_since_last_ocr(self) -> float:
        return time.time() - self.last_ocr_detection_timestamp
    
    @property
    def time_since_last_frame(self) -> float:
        return time.time() - self.last_seen_timestamp


# ============================================================================
# Frame Quality Scoring
# ============================================================================

def compute_frame_quality(
    has_occlusion: bool,
    is_blurry: bool,
    item_count: int,
    config: OrderTrackerConfig
) -> float:
    """
    Compute quality score for a frame.
    
    Quality scoring strategy:
    - Base score: 1.0
    - Subtract penalties for occlusion and blur
    - Add bonus for higher item count
    
    Args:
        has_occlusion: True if hands/persons detected in frame
        is_blurry: True if frame is blurry
        item_count: Number of detected items
        config: Tracker configuration
        
    Returns:
        Quality score between 0.0 and 1.0+
    """
    score = 1.0
    
    # Apply penalties
    if has_occlusion:
        score -= config.occlusion_penalty
    
    if is_blurry:
        score -= config.blur_penalty
    
    # Bonus for item count (capped to avoid runaway scores)
    item_bonus = min(item_count * config.item_count_weight, 0.3)
    score += item_bonus
    
    # Clamp to reasonable range
    return max(0.0, min(score, 1.5))


def detect_blur(image) -> bool:
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        image: numpy array (BGR or grayscale)
        
    Returns:
        True if image is blurry
    """
    try:
        import cv2
        if image is None:
            return False
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Threshold determined empirically - lower variance = more blur
        return laplacian_var < 100
        
    except Exception:
        return False


# ============================================================================
# Order Tracker
# ============================================================================

class OrderTracker:
    """
    Thread-safe order tracker with intelligent finalization logic.
    
    Features:
    - Configurable timeouts and collection windows
    - Grace period frame collection
    - Quality-based frame scoring
    - Diversity-based frame selection
    
    Usage:
        config = OrderTrackerConfig(
            inactivity_timeout_seconds=5.0,
            minimum_collection_time_seconds=2.0,
            grace_frame_count=3,
            top_k_frames=3
        )
        tracker = OrderTracker("station_1", config)
        
        # When frame arrives
        tracker.update_order("384", frame_key, image, item_count, has_occlusion)
        
        # Check if order is ready
        if tracker.should_finalize_order("384"):
            frames = tracker.select_best_frames("384")
            tracker.mark_finalized("384")
    """
    
    def __init__(self, station_id: str, config: Optional[OrderTrackerConfig] = None):
        """
        Initialize order tracker.
        
        Args:
            station_id: Station identifier for logging
            config: Optional configuration (uses defaults if not provided)
        """
        self.station_id = station_id
        self.config = config or OrderTrackerConfig()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Order tracking
        self._orders: Dict[str, TrackedOrder] = {}
        self._finalized_orders: set = set()
        
        # Statistics
        self._total_orders_processed = 0
        self._total_orders_skipped = 0
        
        logger.info(
            f"[{station_id}] OrderTracker initialized: "
            f"inactivity={self.config.inactivity_timeout_seconds}s, "
            f"min_collection={self.config.minimum_collection_time_seconds}s, "
            f"grace_frames={self.config.grace_frame_count}, "
            f"top_k={self.config.top_k_frames}"
        )
    
    def update_order(
        self,
        order_id: str,
        frame_key: str,
        image: Optional[Any] = None,
        item_count: int = 0,
        has_occlusion: bool = False,
        is_blurry: Optional[bool] = None
    ) -> TrackedOrder:
        """
        Update order with new frame data.
        
        This is the main entry point when a frame arrives.
        Handles order creation, state transitions, and frame storage.
        
        Args:
            order_id: Order identifier from OCR
            frame_key: Storage key for frame (e.g., MinIO path)
            image: Optional numpy array of frame
            item_count: Number of items detected by YOLO
            has_occlusion: True if hands/persons detected
            is_blurry: Override blur detection (auto-detects if None)
            
        Returns:
            Updated TrackedOrder
        """
        with self._lock:
            now = time.time()
            
            # Auto-detect blur if not provided
            if is_blurry is None and image is not None:
                is_blurry = detect_blur(image)
            is_blurry = is_blurry or False
            
            # Compute quality score
            quality = compute_frame_quality(
                has_occlusion=has_occlusion,
                is_blurry=is_blurry,
                item_count=item_count,
                config=self.config
            )
            
            # Create frame data (no raw_image stored to prevent memory leak)
            frame = FrameData(
                key=frame_key,
                timestamp=now,
                quality_score=quality,
                has_occlusion=has_occlusion,
                item_count=item_count,
                is_blurry=is_blurry
            )
            
            # Get or create order
            if order_id not in self._orders:
                self._orders[order_id] = TrackedOrder(
                    order_id=order_id,
                    station_id=self.station_id,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                    last_ocr_detection_timestamp=now,
                    grace_frames_remaining=self.config.grace_frame_count
                )
                logger.info(f"[{self.station_id}][TRACKER] New order started: {order_id}")
            
            order = self._orders[order_id]
            
            # Update timestamps
            order.last_seen_timestamp = now
            order.last_ocr_detection_timestamp = now
            order.total_frames_received += 1
            
            # Reset grace period since we got OCR detection
            if order.state == OrderState.GRACE_PERIOD:
                order.state = OrderState.COLLECTING
                order.grace_frames_remaining = self.config.grace_frame_count
                logger.debug(f"[{self.station_id}][TRACKER] Order {order_id} returned to COLLECTING")
            
            # Store frame
            order.frames.append(frame)
            
            # Update statistics
            if has_occlusion:
                order.occluded_frames_count += 1
            else:
                order.clean_frames_count += 1
            
            logger.debug(
                f"[{self.station_id}][TRACKER] Order {order_id}: "
                f"frame #{order.frame_count}, quality={quality:.2f}, "
                f"occluded={has_occlusion}, items={item_count}"
            )
            
            return order
    
    def add_grace_frame(
        self,
        order_id: str,
        frame_key: str,
        image: Optional[Any] = None,
        item_count: int = 0,
        has_occlusion: bool = False
    ) -> bool:
        """
        Add a grace period frame (when OCR no longer detects this order).
        
        Grace frames continue to be collected even after OCR moves to
        a new order, to ensure we capture the complete order.
        
        Args:
            order_id: Order identifier
            frame_key: Storage key for frame
            image: Optional numpy array
            item_count: Number of items detected
            has_occlusion: True if hands/persons detected
            
        Returns:
            True if grace frame was added, False if grace period exhausted
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            
            # Check if grace period exhausted
            if order.grace_frames_remaining <= 0:
                return False
            
            # Transition to grace period if not already
            if order.state == OrderState.COLLECTING:
                order.state = OrderState.GRACE_PERIOD
                logger.info(
                    f"[{self.station_id}][TRACKER] Order {order_id} entering grace period "
                    f"({order.grace_frames_remaining} frames remaining)"
                )
            
            # Compute quality and create frame (no raw_image stored)
            is_blurry = detect_blur(image) if image is not None else False
            quality = compute_frame_quality(
                has_occlusion=has_occlusion,
                is_blurry=is_blurry,
                item_count=item_count,
                config=self.config
            )
            
            frame = FrameData(
                key=frame_key,
                timestamp=time.time(),
                quality_score=quality,
                has_occlusion=has_occlusion,
                item_count=item_count,
                is_blurry=is_blurry
            )
            
            order.frames.append(frame)
            order.last_seen_timestamp = time.time()
            order.grace_frames_remaining -= 1
            order.total_frames_received += 1
            
            if has_occlusion:
                order.occluded_frames_count += 1
            else:
                order.clean_frames_count += 1
            
            logger.debug(
                f"[{self.station_id}][TRACKER] Order {order_id}: grace frame "
                f"({order.grace_frames_remaining} remaining)"
            )
            
            return True
    
    def should_finalize_order(self, order_id: str) -> bool:
        """
        Check if order should be finalized.
        
        Finalization criteria:
        1. Order exists and is not already finalized
        2. No new frames for inactivity_timeout_seconds
        3. Order tracked for at least minimum_collection_time_seconds
        4. Grace period exhausted (if in grace period)
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order should be finalized
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            
            # Already finalized or skipped
            if order.state in (OrderState.FINALIZED, OrderState.SKIPPED):
                return False
            
            # Check minimum collection time
            if order.collection_duration < self.config.minimum_collection_time_seconds:
                return False
            
            # Check inactivity timeout
            if order.time_since_last_frame < self.config.inactivity_timeout_seconds:
                return False
            
            # If in grace period, check if exhausted
            if order.state == OrderState.GRACE_PERIOD:
                if order.grace_frames_remaining > 0:
                    return False
            
            # Mark as ready
            order.state = OrderState.READY
            return True
    
    def force_finalize(self, order_id: str) -> bool:
        """
        Force immediate finalization (e.g., when new order detected).
        
        Bypasses normal timeout checks but still respects minimum
        collection time for reliability.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order was marked ready
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            
            if order.state in (OrderState.FINALIZED, OrderState.SKIPPED):
                return False
            
            # Still require minimum collection time
            if order.collection_duration < self.config.minimum_collection_time_seconds:
                logger.warning(
                    f"[{self.station_id}][TRACKER] Force finalize {order_id} "
                    f"with short collection time: {order.collection_duration:.1f}s"
                )
            
            order.state = OrderState.READY
            logger.info(
                f"[{self.station_id}][TRACKER] Order {order_id} force finalized "
                f"with {order.frame_count} frames"
            )
            return True
    
    def select_best_frames(self, order_id: str) -> List[FrameData]:
        """
        Select best frames for VLM processing using diversity strategy.
        
        Selection strategy:
        1. Highest quality frame (best overall)
        2. First captured frame (establishes context)
        3. Last captured frame (final state)
        4. Fill remaining slots with highest quality
        
        Prefers clean frames but includes occluded frames if necessary.
        
        Args:
            order_id: Order identifier
            
        Returns:
            List of selected FrameData (up to top_k_frames)
        """
        with self._lock:
            if order_id not in self._orders:
                return []
            
            order = self._orders[order_id]
            
            if not order.frames:
                return []
            
            # Separate clean and occluded frames
            clean_frames = [f for f in order.frames if not f.has_occlusion]
            occluded_frames = [f for f in order.frames if f.has_occlusion]
            
            # Sort by quality (descending)
            clean_frames.sort(key=lambda f: f.quality_score, reverse=True)
            occluded_frames.sort(key=lambda f: f.quality_score, reverse=True)
            
            selected: List[FrameData] = []
            used_keys: set = set()
            
            def add_frame(frame: FrameData) -> bool:
                if frame.key not in used_keys and len(selected) < self.config.top_k_frames:
                    selected.append(frame)
                    used_keys.add(frame.key)
                    return True
                return False
            
            # Strategy 1: Highest quality clean frame
            if clean_frames:
                add_frame(clean_frames[0])
            
            # Strategy 2: First captured frame
            first_frame = min(order.frames, key=lambda f: f.timestamp)
            add_frame(first_frame)
            
            # Strategy 3: Last captured frame
            last_frame = max(order.frames, key=lambda f: f.timestamp)
            add_frame(last_frame)
            
            # Strategy 4: Fill with remaining high-quality clean frames
            for frame in clean_frames:
                if len(selected) >= self.config.top_k_frames:
                    break
                add_frame(frame)
            
            # Strategy 5: If still need frames, use best occluded frames
            if len(selected) < self.config.min_frames_hard_requirement:
                for frame in occluded_frames:
                    if len(selected) >= self.config.top_k_frames:
                        break
                    add_frame(frame)
                
                if occluded_frames and len(selected) > len(clean_frames):
                    logger.warning(
                        f"[{self.station_id}][TRACKER] Order {order_id}: "
                        f"using {len(selected) - len(clean_frames)} occluded frames as fallback"
                    )
            
            # Warn if below soft requirement
            if len(selected) < self.config.min_frames_soft_requirement:
                logger.warning(
                    f"[{self.station_id}][TRACKER] Order {order_id}: "
                    f"only {len(selected)} frames selected "
                    f"(soft requirement: {self.config.min_frames_soft_requirement})"
                )
            
            logger.info(
                f"[{self.station_id}][TRACKER] Order {order_id}: "
                f"selected {len(selected)}/{order.frame_count} frames "
                f"(clean={order.clean_frames_count}, occluded={order.occluded_frames_count})"
            )
            
            return selected
    
    def mark_finalized(self, order_id: str) -> bool:
        """
        Mark order as finalized after processing.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if order was marked finalized
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            order.state = OrderState.FINALIZED
            self._finalized_orders.add(order_id)
            self._total_orders_processed += 1
            
            logger.info(
                f"[{self.station_id}][TRACKER] Order {order_id} finalized: "
                f"{order.frame_count} frames, {order.collection_duration:.1f}s duration"
            )
            
            return True
    
    def mark_skipped(self, order_id: str, reason: str = "insufficient frames") -> bool:
        """
        Mark order as skipped (won't be processed).
        
        Args:
            order_id: Order identifier
            reason: Reason for skipping
            
        Returns:
            True if order was marked skipped
        """
        with self._lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            order.state = OrderState.SKIPPED
            self._total_orders_skipped += 1
            
            logger.warning(
                f"[{self.station_id}][TRACKER] Order {order_id} skipped: {reason} "
                f"({order.frame_count} frames)"
            )
            
            return True
    
    def get_order(self, order_id: str) -> Optional[TrackedOrder]:
        """Get order by ID"""
        with self._lock:
            return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[str]:
        """Get list of active (non-finalized) order IDs"""
        with self._lock:
            return [
                oid for oid, order in self._orders.items()
                if order.state not in (OrderState.FINALIZED, OrderState.SKIPPED)
            ]
    
    def get_orders_ready_to_finalize(self) -> List[str]:
        """Get list of orders that should be finalized"""
        with self._lock:
            ready = []
            for order_id in self._orders:
                if self.should_finalize_order(order_id):
                    ready.append(order_id)
            return ready
    
    def cleanup_finalized(self, max_age_seconds: float = 300) -> int:
        """
        Remove old finalized orders to free memory.
        
        Args:
            max_age_seconds: Remove orders older than this
            
        Returns:
            Number of orders removed
        """
        with self._lock:
            now = time.time()
            to_remove = []
            
            for order_id, order in self._orders.items():
                if order.state in (OrderState.FINALIZED, OrderState.SKIPPED):
                    age = now - order.last_seen_timestamp
                    if age > max_age_seconds:
                        to_remove.append(order_id)
            
            for order_id in to_remove:
                del self._orders[order_id]
                self._finalized_orders.discard(order_id)
            
            if to_remove:
                logger.debug(
                    f"[{self.station_id}][TRACKER] Cleaned up {len(to_remove)} old orders"
                )
            
            return len(to_remove)
    
    def is_finalized(self, order_id: str) -> bool:
        """Check if order has been finalized"""
        with self._lock:
            return order_id in self._finalized_orders
    
    def reset(self):
        """Reset tracker state (e.g., for new video stream)"""
        with self._lock:
            self._orders.clear()
            self._finalized_orders.clear()
            logger.info(f"[{self.station_id}][TRACKER] State reset")
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        with self._lock:
            active = [o for o in self._orders.values() 
                     if o.state not in (OrderState.FINALIZED, OrderState.SKIPPED)]
            
            return {
                'station_id': self.station_id,
                'total_processed': self._total_orders_processed,
                'total_skipped': self._total_orders_skipped,
                'active_orders': len(active),
                'tracked_orders': len(self._orders),
                'config': {
                    'inactivity_timeout': self.config.inactivity_timeout_seconds,
                    'min_collection_time': self.config.minimum_collection_time_seconds,
                    'grace_frames': self.config.grace_frame_count,
                    'top_k': self.config.top_k_frames,
                }
            }


# ============================================================================
# Integration Helper
# ============================================================================

class OrderTrackerManager:
    """
    Manager for multiple station trackers.
    
    Provides centralized access to per-station trackers
    for multi-worker deployments.
    """
    
    def __init__(self, config: Optional[OrderTrackerConfig] = None):
        self._config = config or OrderTrackerConfig()
        self._trackers: Dict[str, OrderTracker] = {}
        self._lock = threading.Lock()
    
    def get_tracker(self, station_id: str) -> OrderTracker:
        """Get or create tracker for station"""
        with self._lock:
            if station_id not in self._trackers:
                self._trackers[station_id] = OrderTracker(station_id, self._config)
            return self._trackers[station_id]
    
    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all trackers"""
        with self._lock:
            return {
                station_id: tracker.get_statistics()
                for station_id, tracker in self._trackers.items()
            }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = OrderTrackerConfig(
        inactivity_timeout_seconds=5.0,
        minimum_collection_time_seconds=2.0,
        grace_frame_count=3,
        top_k_frames=3,
        min_frames_soft_requirement=3,
        min_frames_hard_requirement=1,
        occlusion_penalty=0.3
    )
    
    # Create tracker
    tracker = OrderTracker("station_1", config)
    
    # Simulate frame arrivals
    import numpy as np
    
    print("\n=== Simulating Order 384 ===")
    
    # Frame 1: Clean frame with 2 items
    tracker.update_order(
        order_id="384",
        frame_key="station_1/384/frame_1.jpg",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        item_count=2,
        has_occlusion=False
    )
    
    # Frame 2: Occluded frame (hand detected)
    tracker.update_order(
        order_id="384",
        frame_key="station_1/384/frame_2.jpg",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        item_count=1,
        has_occlusion=True
    )
    
    # Frame 3: Clean frame with 2 items
    tracker.update_order(
        order_id="384",
        frame_key="station_1/384/frame_3.jpg",
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        item_count=2,
        has_occlusion=False
    )
    
    # Check order state
    order = tracker.get_order("384")
    print(f"Order state: {order.state}")
    print(f"Frame count: {order.frame_count}")
    print(f"Clean frames: {order.clean_frames_count}")
    print(f"Occluded frames: {order.occluded_frames_count}")
    
    # Simulate new order detected (force finalize previous)
    print("\n=== New Order Detected - Force Finalize 384 ===")
    tracker.force_finalize("384")
    
    # Select best frames
    selected = tracker.select_best_frames("384")
    print(f"\nSelected {len(selected)} frames:")
    for i, frame in enumerate(selected):
        print(f"  {i+1}. {frame.key} (quality={frame.quality_score:.2f}, occluded={frame.has_occlusion})")
    
    # Mark finalized
    tracker.mark_finalized("384")
    
    # Print statistics
    print("\n=== Tracker Statistics ===")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
