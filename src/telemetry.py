"""
Structured telemetry logger for request-level JSONL events.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import orjson


class TelemetryLogger:
    """Append telemetry events as JSONL records."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log_event(self, event: Dict[str, Any]) -> None:
        """Write a single telemetry event."""
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        output_path = self.log_dir / f"telemetry_{date.today().isoformat()}.jsonl"
        line = orjson.dumps(payload).decode("utf-8")

        with self._lock:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
