"""Dedicated JSONL debug logging for VLM predictions and validation results."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _base_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _log_path() -> Path:
    results_dir = Path(os.getenv("CONTAINER_RESULTS_PATH") or (_base_dir() / "results"))
    return results_dir / "logs" / "vlm_predictions.log"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_prediction_debug(record: Dict[str, Any]) -> None:
    """Append one structured debug record to the persistent JSONL log."""
    try:
        log_path = _log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_json_safe(record),
        }

        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write prediction debug log: {exc}")