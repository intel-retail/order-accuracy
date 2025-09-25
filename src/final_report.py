"""
Final Report Module - Handles session metrics tracking
"""
import os
import json
from datetime import datetime
from config import logger

# Persistent metrics tracking (reset on app start)
METRICS_FILE = "session_metrics.json"
total_videos = 0
successful_videos = 0
failed_videos = 0
failed_orders_list = []

def reset_metrics_file():
    """Delete metrics file and reset counters on app start"""
    global total_videos, successful_videos, failed_videos, failed_orders_list

    # Delete existing metrics file
    if os.path.exists(METRICS_FILE):
        try:
            os.remove(METRICS_FILE)
            logger.info("Previous session metrics file deleted")
        except Exception as e:
            logger.warning(f"Failed to delete metrics file: {e}")
    
    # Reset counters
    total_videos = 0
    successful_videos = 0
    failed_videos = 0
    failed_orders_list= []
    
    # Create fresh metrics file
    save_metrics_to_file()
    logger.info("Fresh session metrics initialized")

def save_metrics_to_file():
    """Save current metrics to file"""
    try:
        data = {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'failed_orders_list': failed_orders_list,
            'last_updated': datetime.now().isoformat(),
            'session_start': datetime.now().isoformat() if total_videos == 0 else None
        }
        with open(METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

def load_metrics_from_file():
    """Load metrics from file (for browser refresh during same session)"""
    global total_videos, successful_videos, failed_videos, failed_orders_list
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                data = json.load(f)
                total_videos = data.get('total_videos', 0)
                successful_videos = data.get('successful_videos', 0)
                failed_videos = data.get('failed_videos', 0)
                failed_orders_list = data.get('failed_orders_list', [])
                logger.info(f"Loaded session metrics: Total={total_videos}, Success={successful_videos}, Failed={failed_videos}")
        else:
            # File doesn't exist, start fresh
            reset_metrics_file()
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        reset_metrics_file()

def update_metrics(success=True, failed_order_id=None):
    """Update metrics and save to file"""
    global total_videos, successful_videos, failed_videos, failed_orders_list
    total_videos += 1
    if success:
        successful_videos += 1
    else:
        failed_videos += 1
        failed_orders_list.append(failed_order_id)  # Track failed order by its ID

    # Save to file immediately after update
    save_metrics_to_file()


def get_metrics_dict():
    """Get metrics as dictionary for programmatic use"""
    success_rate = (successful_videos / max(total_videos, 1)) * 100 if total_videos > 0 else 0
    if failed_orders_list:
        order_summary = {
            '📦 Total Orders': total_videos,
            '✅ Successful Orders': successful_videos,
            '❌       Failed Orders Count': failed_videos,
            '📋       Failed Orders': ",".join(failed_orders_list),
            '📊 Success Rate': f"{success_rate:.1f}%"
        }
    else:
        order_summary = {
            '📦 Total Orders': total_videos,
            '✅ Successful Orders': successful_videos,
            '❌       Failed Orders Count': failed_videos,
            '📊 Success Rate': f"{success_rate:.1f}%"
        }
    return order_summary