from collections import deque
from threading import Lock
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import video history tracking (lazy to avoid circular imports)
_video_history = None

def _get_video_history():
    """Lazy import video history module"""
    global _video_history
    if _video_history is None:
        try:
            from core.video_history import add_result_to_video, get_current_video_id
            _video_history = {'add_result': add_result_to_video, 'get_current': get_current_video_id}
            logger.info("[RESULTS] Video history tracking enabled")
        except ImportError as e:
            logger.warning(f"[RESULTS] Video history tracking not available: {e}")
            _video_history = {'add_result': lambda r: None, 'get_current': lambda: None}
    return _video_history

# Default Station ID from environment (for backwards compatibility)
DEFAULT_STATION_ID = os.environ.get('STATION_ID', 'station_1')

# Results directory
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', '/results'))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RESULTS = 200  # Keep results across many video loops for run comparison

# Per-station state tracking
class StationResults:
    """Tracks results for a single station"""
    def __init__(self, station_id: str):
        self.station_id = station_id
        self.results = deque(maxlen=MAX_RESULTS)
        self.total_processed = 0
        self.total_validated = 0
        self.total_mismatch = 0
        self.order_run_counts: Dict[str, int] = {}  # order_id -> number of times processed
        self.lock = Lock()
        
        # File paths for this station
        self.results_file = RESULTS_DIR / f"{station_id}_results.jsonl"
        self.summary_file = RESULTS_DIR / f"{station_id}_summary.json"
        self.report_file = RESULTS_DIR / f"{station_id}_report.md"
        
        logger.info(f"[RESULTS] StationResults initialized for {station_id}")

# Global registry of station states
_station_registry: Dict[str, StationResults] = {}
_registry_lock = Lock()

def _get_station(station_id: str) -> StationResults:
    """Get or create station state"""
    with _registry_lock:
        if station_id not in _station_registry:
            _station_registry[station_id] = StationResults(station_id)
            logger.info(f"[RESULTS] Created new station tracker: {station_id}")
        return _station_registry[station_id]

logger.info(f"Order results storage initialized: default_station={DEFAULT_STATION_ID}, max_results={MAX_RESULTS}")
logger.info(f"Results directory: {RESULTS_DIR}")

def _update_summary(station: StationResults):
    """Update station summary file with statistics"""
    try:
        summary = {
            'station_id': station.station_id,
            'last_updated': datetime.now().isoformat(),
            'total_processed': station.total_processed,
            'total_validated': station.total_validated,
            'total_mismatch': station.total_mismatch,
            'validation_rate': (station.total_validated / station.total_processed * 100) if station.total_processed > 0 else 0,
            'recent_results': [
                {
                    'order_id': r.get('order_id'),
                    'run_number': r.get('run_number', '?'),
                    'status': r.get('status'),
                    'inference_time': r.get('inference_time_sec'),
                    'completed_at': r.get('completed_at', 'N/A')
                }
                for r in list(station.results)
            ]
        }
        
        with open(station.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.debug(f"[RESULTS] Summary updated for {station.station_id}: {station.total_processed} processed")
    except Exception as e:
        logger.error(f"[RESULTS] Failed to update summary: {e}")


def _update_readable_report(station: StationResults):
    """Update human-readable markdown report for this station"""
    try:
        lines = [
            f"# {station.station_id.replace('_', ' ').title()} Results Report",
            f"",
            f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Summary",
            f"- Total Processed: {station.total_processed}",
            f"- Validated: {station.total_validated} ({'%.1f' % ((station.total_validated / station.total_processed * 100) if station.total_processed > 0 else 0)}%)",
            f"- Mismatch: {station.total_mismatch}",
            f"",
            f"---",
            f"",
            f"## Recent Orders",
            f""
        ]
        
        for result in list(station.results):
            order_id = result.get('order_id', 'unknown')
            status = result.get('status', 'unknown')
            completed_at = result.get('completed_at', 'N/A')
            inference_time = result.get('inference_time_sec', 0)
            validation = result.get('validation', {})
            
            missing = validation.get('missing', [])
            extra = validation.get('extra', [])
            qty_mismatch = validation.get('quantity_mismatch', [])
            
            status_emoji = "✅ VALIDATED" if status == "validated" else "❌ MISMATCH"
            run_number = result.get('run_number', '?')

            lines.extend([
                f"### Order {order_id} — Run #{run_number}",
                f"- **Completed At:** {completed_at}",
                f"- **Inference Time:** {inference_time:.2f}s",
                f"- Status: {status_emoji}",
                f"- Missing: {missing if missing else 'None'}",
                f"- Extra: {extra if extra else 'None'}",
                f"- Quantity Mismatch: {qty_mismatch if qty_mismatch else 'None'}",
                f""
            ])
        
        with open(station.report_file, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.debug(f"[RESULTS] Report updated: {station.report_file}")
    except Exception as e:
        logger.error(f"[RESULTS] Failed to update report: {e}")


def add_result(result: dict, station_id: Optional[str] = None):
    """
    Add a result for a station.
    Always appends results (even duplicates) to track all detections across video loops.
    
    Args:
        result: The result dictionary
        station_id: Optional station ID (defaults to DEFAULT_STATION_ID)
    """
    # Determine station ID from result or parameter or default
    if station_id is None:
        station_id = result.get('station_id', DEFAULT_STATION_ID)
    
    station = _get_station(station_id)
    
    order_id = result.get('order_id', 'unknown')
    status = result.get('status', 'unknown')
    logger.info(f"[RESULTS] Adding result for {station_id}: order_id={order_id}, status={status}")

    result['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Critical section: in-memory mutations only (no I/O inside lock) ──
    with station.lock:
        station.order_run_counts[order_id] = station.order_run_counts.get(order_id, 0) + 1
        result['run_number'] = station.order_run_counts[order_id]
        station.results.appendleft(result)
        station.total_processed += 1
        if status == 'validated':
            station.total_validated += 1
        elif status == 'mismatch':
            station.total_mismatch += 1
        # Take a snapshot for file writes so we can release lock before I/O
        station_snapshot = station

    # ── File I/O outside lock so writers never block readers ──────────────
    _update_summary(station_snapshot)
    _update_readable_report(station_snapshot)

    # Track in video history (also outside lock)
    try:
        vh = _get_video_history()
        current_video_id = vh['get_current']()
        if current_video_id:
            vh['add_result'](current_video_id, result)
            logger.debug(f"[RESULTS] Result added to video history for order_id={order_id}, video_id={current_video_id}")
    except Exception as e:
        logger.debug(f"[RESULTS] Could not add to video history: {e}")

    logger.info(f"[RESULTS] {station_id} stats: {station.total_processed} processed, {station.total_validated} validated, {station.total_mismatch} mismatch")


def get_results(station_id: Optional[str] = None):
    """Get results for a station"""
    if station_id is None:
        station_id = DEFAULT_STATION_ID

    station = _get_station(station_id)

    if not station.lock.acquire(blocking=True, timeout=2.0):
        logger.warning(f"[RESULTS] get_results lock timeout for {station_id}, returning empty list")
        return []
    try:
        result_list = list(station.results)
        logger.debug(f"[RESULTS] Retrieved {len(result_list)} results for {station_id}")
        return result_list
    finally:
        station.lock.release()


def get_statistics(station_id: Optional[str] = None):
    """Get processing statistics for a station (non-blocking, 2s timeout)"""
    if station_id is None:
        station_id = DEFAULT_STATION_ID

    station = _get_station(station_id)

    if not station.lock.acquire(blocking=True, timeout=2.0):
        logger.warning(f"[RESULTS] get_statistics lock timeout for {station_id}, returning partial stats")
        # Return what we can read without the lock (ints are atomic-enough for a status read)
        return {
            'station_id': station_id,
            'total_processed': station.total_processed,
            'total_validated': station.total_validated,
            'total_mismatch': station.total_mismatch,
            'validation_rate': 0,
            'note': 'partial — lock busy'
        }
    try:
        return {
            'station_id': station.station_id,
            'total_processed': station.total_processed,
            'total_validated': station.total_validated,
            'total_mismatch': station.total_mismatch,
            'validation_rate': (station.total_validated / station.total_processed * 100) if station.total_processed > 0 else 0
        }
    finally:
        station.lock.release()


def get_all_stations():
    """Get list of all active stations"""
    with _registry_lock:
        return list(_station_registry.keys())


def get_all_statistics():
    """Get statistics for all stations"""
    with _registry_lock:
        return {
            station_id: get_statistics(station_id)
            for station_id in _station_registry.keys()
        }


def clear_all_results():
    """Clear all results and statistics for all stations"""
    with _registry_lock:
        count = 0
        for station_id, station in _station_registry.items():
            with station.lock:
                count += len(station.results)
                station.results.clear()
                station.total_processed = 0
                station.total_validated = 0
                station.total_mismatch = 0
                
                # Clear files
                try:
                    if station.results_file.exists():
                        station.results_file.unlink()
                    if station.summary_file.exists():
                        station.summary_file.unlink()
                    if station.report_file.exists():
                        station.report_file.unlink()
                except Exception as e:
                    logger.error(f"[RESULTS] Failed to clear files for {station_id}: {e}")
        
        logger.info(f"[RESULTS] Cleared {count} results from all stations")
        return {'cleared_count': count}
