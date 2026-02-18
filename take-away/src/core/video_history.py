"""
Video Processing History Tracker
Tracks uploaded videos and their associated order results
"""
import os
import json
import logging
from collections import deque
from threading import Lock
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', '/results'))
HISTORY_FILE = RESULTS_DIR / 'video_history.json'
MAX_HISTORY = 50  # Keep last 50 videos


@dataclass
class VideoEntry:
    """Represents a processed video entry"""
    video_id: str
    filename: str
    upload_time: str
    status: str  # 'processing', 'completed', 'failed'
    path: str
    station_id: str = 'station_1'
    orders_detected: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    completed_time: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoEntry':
        return cls(**data)


class VideoHistoryTracker:
    """Singleton tracker for video processing history"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._history: deque = deque(maxlen=MAX_HISTORY)
        self._video_map: Dict[str, VideoEntry] = {}
        self._current_video_id: Optional[str] = None
        self._lock = Lock()
        
        # Load existing history from file
        self._load_history()
        
        self._initialized = True
        logger.info(f"[VIDEO-HISTORY] Initialized with {len(self._history)} existing entries")
    
    def _load_history(self):
        """Load history from disk"""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get('videos', []):
                        entry = VideoEntry.from_dict(entry_data)
                        self._history.append(entry)
                        self._video_map[entry.video_id] = entry
                logger.info(f"[VIDEO-HISTORY] Loaded {len(self._history)} entries from {HISTORY_FILE}")
        except Exception as e:
            logger.error(f"[VIDEO-HISTORY] Failed to load history: {e}")
    
    def _save_history(self):
        """Save history to disk"""
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_videos': len(self._history),
                'videos': [entry.to_dict() for entry in self._history]
            }
            with open(HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"[VIDEO-HISTORY] Saved {len(self._history)} entries to {HISTORY_FILE}")
        except Exception as e:
            logger.error(f"[VIDEO-HISTORY] Failed to save history: {e}")
    
    def start_video(self, video_id: str, filename: str, path: str, station_id: str = 'station_1') -> VideoEntry:
        """Register a new video upload"""
        with self._lock:
            entry = VideoEntry(
                video_id=video_id,
                filename=filename,
                upload_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                status='processing',
                path=path,
                station_id=station_id
            )
            
            self._history.appendleft(entry)
            self._video_map[video_id] = entry
            self._current_video_id = video_id
            
            self._save_history()
            
            logger.info(f"[VIDEO-HISTORY] Started tracking video: {video_id} ({filename})")
            return entry
    
    def add_order_to_video(self, video_id: str, order_id: str):
        """Add an order ID to a video's detected orders"""
        with self._lock:
            if video_id in self._video_map:
                entry = self._video_map[video_id]
                if order_id not in entry.orders_detected:
                    entry.orders_detected.append(order_id)
                    logger.debug(f"[VIDEO-HISTORY] Added order {order_id} to video {video_id}")
            elif self._current_video_id and self._current_video_id in self._video_map:
                # Fallback to current video if video_id not found
                entry = self._video_map[self._current_video_id]
                if order_id not in entry.orders_detected:
                    entry.orders_detected.append(order_id)
                    logger.debug(f"[VIDEO-HISTORY] Added order {order_id} to current video {self._current_video_id}")
    
    def add_result_to_video(self, video_id: str, result: Dict[str, Any]):
        """Add a processing result to a video"""
        with self._lock:
            target_video_id = video_id
            if video_id not in self._video_map:
                target_video_id = self._current_video_id
            
            if target_video_id and target_video_id in self._video_map:
                entry = self._video_map[target_video_id]
                entry.results.append(result)
                
                order_id = result.get('order_id')
                if order_id and order_id not in entry.orders_detected:
                    entry.orders_detected.append(order_id)
                
                logger.debug(f"[VIDEO-HISTORY] Added result to video {target_video_id}: order={order_id}")
    
    def complete_video(self, video_id: str, results: Optional[List[Dict]] = None):
        """Mark a video as completed with its results"""
        with self._lock:
            target_video_id = video_id
            if video_id not in self._video_map:
                target_video_id = self._current_video_id
            
            if target_video_id and target_video_id in self._video_map:
                entry = self._video_map[target_video_id]
                entry.status = 'completed'
                entry.completed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if results:
                    entry.results = results
                    entry.orders_detected = list(set(
                        entry.orders_detected + [r.get('order_id') for r in results if r.get('order_id')]
                    ))
                
                self._save_history()
                
                logger.info(f"[VIDEO-HISTORY] Completed video {target_video_id}: {len(entry.results)} results")
            
            # Don't clear current_video_id here - VLM results may still be coming in
            # It will be cleared when a new video is started
    
    def fail_video(self, video_id: str, error_message: str):
        """Mark a video as failed"""
        with self._lock:
            if video_id in self._video_map:
                entry = self._video_map[video_id]
                entry.status = 'failed'
                entry.error_message = error_message
                entry.completed_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save_history()
                logger.error(f"[VIDEO-HISTORY] Video failed: {video_id} - {error_message}")
    
    def get_video(self, video_id: str) -> Optional[Dict]:
        """Get a specific video entry"""
        with self._lock:
            if video_id in self._video_map:
                return self._video_map[video_id].to_dict()
            return None
    
    def get_current_video_id(self) -> Optional[str]:
        """Get the current processing video ID"""
        return self._current_video_id
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """Get video processing history"""
        with self._lock:
            return [entry.to_dict() for entry in list(self._history)[:limit]]
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        with self._lock:
            completed = sum(1 for e in self._history if e.status == 'completed')
            processing = sum(1 for e in self._history if e.status == 'processing')
            failed = sum(1 for e in self._history if e.status == 'failed')
            total_orders = sum(len(e.orders_detected) for e in self._history)
            
            return {
                'total_videos': len(self._history),
                'completed': completed,
                'processing': processing,
                'failed': failed,
                'total_orders_processed': total_orders,
                'current_video_id': self._current_video_id
            }
    
    def clear_history(self) -> Dict:
        """Clear all video history"""
        with self._lock:
            count = len(self._history)
            self._history.clear()
            self._video_map.clear()
            self._current_video_id = None
            
            # Delete history file
            try:
                if HISTORY_FILE.exists():
                    HISTORY_FILE.unlink()
                    logger.info(f"[VIDEO-HISTORY] Deleted history file: {HISTORY_FILE}")
            except Exception as e:
                logger.error(f"[VIDEO-HISTORY] Failed to delete history file: {e}")
            
            logger.info(f"[VIDEO-HISTORY] Cleared {count} videos from history")
            return {
                'status': 'success',
                'cleared_count': count
            }


# Singleton instance
_tracker: Optional[VideoHistoryTracker] = None


def get_video_tracker() -> VideoHistoryTracker:
    """Get the singleton video history tracker"""
    global _tracker
    if _tracker is None:
        _tracker = VideoHistoryTracker()
    return _tracker


# Convenience functions
def start_video(video_id: str, filename: str, path: str, station_id: str = 'station_1'):
    """Register a new video upload"""
    return get_video_tracker().start_video(video_id, filename, path, station_id)


def add_order_to_video(video_id: str, order_id: str):
    """Add an order ID to a video"""
    get_video_tracker().add_order_to_video(video_id, order_id)


def add_result_to_video(video_id: str, result: Dict):
    """Add a result to a video"""
    get_video_tracker().add_result_to_video(video_id, result)


def complete_video(video_id: str, results: Optional[List[Dict]] = None):
    """Mark a video as completed"""
    get_video_tracker().complete_video(video_id, results)


def fail_video(video_id: str, error_message: str):
    """Mark a video as failed"""
    get_video_tracker().fail_video(video_id, error_message)


def get_video(video_id: str) -> Optional[Dict]:
    """Get a specific video entry"""
    return get_video_tracker().get_video(video_id)


def get_video_history(limit: int = 20) -> List[Dict]:
    """Get video processing history"""
    return get_video_tracker().get_history(limit)


def get_video_summary() -> Dict:
    """Get summary statistics"""
    return get_video_tracker().get_summary()


def get_current_video_id() -> Optional[str]:
    """Get the current processing video ID"""
    return get_video_tracker().get_current_video_id()


def clear_video_history() -> Dict:
    """Clear all video history"""
    return get_video_tracker().clear_history()
