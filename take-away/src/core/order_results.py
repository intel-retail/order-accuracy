from collections import deque
from threading import Lock
import logging
import json
import os
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Station ID from environment
STATION_ID = os.environ.get('STATION_ID', 'station_1')

# Results directory
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', '/results'))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Station-specific results file
STATION_RESULTS_FILE = RESULTS_DIR / f"{STATION_ID}_results.jsonl"
STATION_SUMMARY_FILE = RESULTS_DIR / f"{STATION_ID}_summary.json"

MAX_RESULTS = 3

_results = deque(maxlen=MAX_RESULTS)
_lock = Lock()
_total_processed = 0
_total_validated = 0
_total_mismatch = 0

logger.info(f"Order results storage initialized: station={STATION_ID}, max_results={MAX_RESULTS}")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Station results file: {STATION_RESULTS_FILE}")

def _write_result_to_file(result: dict):
    """Write result to station-specific JSONL file"""
    try:
        # Add timestamp and station ID
        result_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'station_id': STATION_ID,
            **result
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(STATION_RESULTS_FILE, 'a') as f:
            f.write(json.dumps(result_with_meta) + '\n')
        
        logger.debug(f"[RESULTS] Written to file: {STATION_RESULTS_FILE}")
    except Exception as e:
        logger.error(f"[RESULTS] Failed to write to file: {e}")

def _update_summary():
    """Update station summary file with statistics"""
    global _total_processed, _total_validated, _total_mismatch
    
    try:
        summary = {
            'station_id': STATION_ID,
            'last_updated': datetime.now().isoformat(),
            'total_processed': _total_processed,
            'total_validated': _total_validated,
            'total_mismatch': _total_mismatch,
            'validation_rate': (_total_validated / _total_processed * 100) if _total_processed > 0 else 0,
            'recent_results': [
                {
                    'order_id': r.get('order_id'),
                    'status': r.get('status'),
                    'inference_time': r.get('inference_time_sec')
                }
                for r in list(_results)
            ]
        }
        
        with open(STATION_SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.debug(f"[RESULTS] Summary updated: {_total_processed} processed, {_total_validated} validated, {_total_mismatch} mismatch")
    except Exception as e:
        logger.error(f"[RESULTS] Failed to update summary: {e}")

def add_result(result: dict):
    global _total_processed, _total_validated, _total_mismatch
    
    with _lock:
        order_id = result.get('order_id', 'unknown')
        status = result.get('status', 'unknown')
        
        logger.info(f"[RESULTS] Adding result: order_id={order_id}, status={status}")
        logger.debug(f"[RESULTS] Result details: {result}")
        
        _results.appendleft(result)
        
        # Update statistics
        _total_processed += 1
        if status == 'validated':
            _total_validated += 1
        elif status == 'mismatch':
            _total_mismatch += 1
        
        # Write to file
        _write_result_to_file(result)
        _update_summary()
        
        logger.debug(f"[RESULTS] Current result count: {len(_results)}/{MAX_RESULTS}")
        logger.info(f"[RESULTS] Station {STATION_ID} stats: {_total_processed} processed, {_total_validated} validated, {_total_mismatch} mismatch")

def get_results():
    with _lock:
        result_list = list(_results)
        logger.debug(f"[RESULTS] Retrieved {len(result_list)} results")
        return result_list

def get_statistics():
    """Get processing statistics for this station"""
    with _lock:
        return {
            'station_id': STATION_ID,
            'total_processed': _total_processed,
            'total_validated': _total_validated,
            'total_mismatch': _total_mismatch,
            'validation_rate': (_total_validated / _total_processed * 100) if _total_processed > 0 else 0
        }
