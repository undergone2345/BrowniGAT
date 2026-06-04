from datetime import datetime, timezone
from pathlib import Path
import json


def append_event(output_path, event_type, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_events(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return 0
    with output_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
