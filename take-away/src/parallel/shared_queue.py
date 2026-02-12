"""
Shared Queue Abstraction for Inter-Process Communication

Provides unified interface for task queues between station workers and VLM scheduler.
Supports both multiprocessing.Queue and Redis backends.
"""

import multiprocessing as mp
from typing import Optional, Any, Dict
from enum import Enum
import json
import time
import logging

logger = logging.getLogger(__name__)


class QueueBackend(Enum):
    """Supported queue backend types"""
    MULTIPROCESSING = "multiprocessing"
    REDIS = "redis"


class VLMRequest:
    """VLM inference request from station worker"""
    
    def __init__(
        self,
        station_id: str,
        order_id: str,
        frames: list,  # List of frame data (numpy arrays or base64)
        timestamp: float,
        priority: int = 0
    ):
        self.station_id = station_id
        self.order_id = order_id
        self.frames = frames
        self.timestamp = timestamp
        self.priority = priority
        self.request_id = f"{station_id}_{order_id}_{int(timestamp * 1000)}"
    
    def to_dict(self) -> Dict:
        """Serialize for queue transmission"""
        return {
            'station_id': self.station_id,
            'order_id': self.order_id,
            'frames': self.frames,  # Already serialized
            'timestamp': self.timestamp,
            'priority': self.priority,
            'request_id': self.request_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VLMRequest':
        """Deserialize from queue"""
        req = cls(
            station_id=data['station_id'],
            order_id=data['order_id'],
            frames=data['frames'],
            timestamp=data['timestamp'],
            priority=data.get('priority', 0)
        )
        req.request_id = data['request_id']
        return req


class VLMResponse:
    """VLM inference response to station worker"""
    
    def __init__(
        self,
        request_id: str,
        station_id: str,
        order_id: str,
        detected_items: list,
        inference_time: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.request_id = request_id
        self.station_id = station_id
        self.order_id = order_id
        self.detected_items = detected_items
        self.inference_time = inference_time
        self.success = success
        self.error = error
    
    def to_dict(self) -> Dict:
        """Serialize for queue transmission"""
        return {
            'request_id': self.request_id,
            'station_id': self.station_id,
            'order_id': self.order_id,
            'detected_items': self.detected_items,
            'inference_time': self.inference_time,
            'success': self.success,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VLMResponse':
        """Deserialize from queue"""
        return cls(
            request_id=data['request_id'],
            station_id=data['station_id'],
            order_id=data['order_id'],
            detected_items=data['detected_items'],
            inference_time=data['inference_time'],
            success=data.get('success', True),
            error=data.get('error')
        )


class SharedQueue:
    """
    Unified queue interface for inter-process communication.
    
    Supports both multiprocessing.Queue (fast, single-node)
    and Redis (distributed, persistent).
    """
    
    def __init__(
        self,
        name: str,
        backend: QueueBackend = QueueBackend.MULTIPROCESSING,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        maxsize: int = 1000
    ):
        self.name = name
        self.backend = backend
        self.maxsize = maxsize
        
        if backend == QueueBackend.MULTIPROCESSING:
            self._queue = mp.Queue(maxsize=maxsize)
            logger.info(f"Initialized multiprocessing queue: {name} (maxsize={maxsize})")
        
        elif backend == QueueBackend.REDIS:
            try:
                import redis
                self._redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self._redis.ping()
                logger.info(f"Initialized Redis queue: {name} at {redis_host}:{redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported queue backend: {backend}")
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None):
        """Put item into queue"""
        if self.backend == QueueBackend.MULTIPROCESSING:
            self._queue.put(item, block=block, timeout=timeout)
        
        elif self.backend == QueueBackend.REDIS:
            # Serialize to JSON
            serialized = json.dumps(item)
            self._redis.rpush(self.name, serialized)
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Get item from queue"""
        if self.backend == QueueBackend.MULTIPROCESSING:
            return self._queue.get(block=block, timeout=timeout)
        
        elif self.backend == QueueBackend.REDIS:
            if block:
                # Blocking pop with timeout
                result = self._redis.blpop(self.name, timeout=timeout or 0)
                if result:
                    _, serialized = result
                    return json.loads(serialized)
                return None
            else:
                # Non-blocking pop
                serialized = self._redis.lpop(self.name)
                if serialized:
                    return json.loads(serialized)
                return None
    
    def put_nowait(self, item: Any):
        """Non-blocking put"""
        self.put(item, block=False)
    
    def get_nowait(self) -> Any:
        """Non-blocking get"""
        return self.get(block=False)
    
    def qsize(self) -> int:
        """Get approximate queue size"""
        if self.backend == QueueBackend.MULTIPROCESSING:
            return self._queue.qsize()
        
        elif self.backend == QueueBackend.REDIS:
            return self._redis.llen(self.name)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.qsize() == 0
    
    def clear(self):
        """Clear all items from queue"""
        if self.backend == QueueBackend.MULTIPROCESSING:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except:
                    break
        
        elif self.backend == QueueBackend.REDIS:
            self._redis.delete(self.name)


class QueueManager:
    """
    Manages all shared queues in the system.
    
    Queues:
    - vlm_request_queue: Station workers → VLM scheduler
    - vlm_response_queue_{station_id}: VLM scheduler → Station workers
    - metrics_queue: All components → Metrics collector
    - control_queue: Station manager → Workers (shutdown signals)
    """
    
    def __init__(
        self,
        backend: QueueBackend = QueueBackend.MULTIPROCESSING,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        station_ids: list = None
    ):
        self.backend = backend
        self.redis_host = redis_host
        self.redis_port = redis_port
        
        # Central VLM request queue
        self.vlm_request_queue = SharedQueue(
            name="vlm_requests",
            backend=backend,
            redis_host=redis_host,
            redis_port=redis_port,
            maxsize=500
        )
        
        # Per-station response queues - CREATE ALL UPFRONT to share across processes
        self.response_queues: Dict[str, SharedQueue] = {}
        if station_ids:
            logger.info(f"Pre-creating response queues for {len(station_ids)} stations")
            for station_id in station_ids:
                self.response_queues[station_id] = SharedQueue(
                    name=f"vlm_response_{station_id}",
                    backend=backend,
                    redis_host=redis_host,
                    redis_port=redis_port,
                    maxsize=100
                )
        
        # Metrics collection queue
        self.metrics_queue = SharedQueue(
            name="metrics",
            backend=backend,
            redis_host=redis_host,
            redis_port=redis_port,
            maxsize=1000
        )
        
        # Control queue for shutdown signals
        self.control_queue = SharedQueue(
            name="control",
            backend=backend,
            redis_host=redis_host,
            redis_port=redis_port,
            maxsize=100
        )
        
        logger.info(f"QueueManager initialized with {backend.value} backend")
    
    def get_response_queue(self, station_id: str) -> SharedQueue:
        """Get or create response queue for station"""
        if station_id not in self.response_queues:
            self.response_queues[station_id] = SharedQueue(
                name=f"vlm_response_{station_id}",
                backend=self.backend,
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                maxsize=100
            )
        return self.response_queues[station_id]
    
    def shutdown(self):
        """Clean up all queues"""
        logger.info("Shutting down queue manager...")
        
        # Clear all queues
        self.vlm_request_queue.clear()
        self.metrics_queue.clear()
        self.control_queue.clear()
        
        for queue in self.response_queues.values():
            queue.clear()
        
        logger.info("Queue manager shutdown complete")
