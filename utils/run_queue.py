from datetime import datetime, timezone
from pathlib import Path
import json


def build_run_queue(stage_plan, workspace_dir, scheduler_cfg):
    workspace_dir = Path(workspace_dir)
    queue = []
    max_retries = int(scheduler_cfg.get("failure_recovery", {}).get("max_retries", 0))
    enforce_stage_order = bool(scheduler_cfg.get("enforce_stage_order", True))
    for stage_index, stage_spec in enumerate(stage_plan, start=1):
        stage_workspace = workspace_dir / "scheduled_runs" / stage_spec["stage_id"]
        queue.append(
            {
                "queue_id": f"queue_{stage_index:02d}",
                "stage_id": stage_spec["stage_id"],
                "stage_name": stage_spec["stage_name"],
                "epochs": int(stage_spec["epochs"]),
                "active_tasks": stage_spec["active_tasks"],
                "stage_kind": stage_spec.get("stage_kind"),
                "priority": int(stage_spec.get("priority", 0)),
                "partition_budget": int(stage_spec["partition_budget"]),
                "resource_request": stage_spec.get("resource_request", {}),
                "workspace_dir": str(stage_workspace),
                "status": "pending",
                "retry_count": 0,
                "max_retries": max_retries,
                "resume_from_checkpoint": None,
                "depends_on": stage_spec.get("depends_on")
                if stage_spec.get("depends_on") is not None
                else (f"queue_{stage_index - 1:02d}" if enforce_stage_order and stage_index > 1 else None),
                "promotion_decision": None,
            }
        )
    return queue


def save_run_queue(queue, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(queue, handle, indent=2, ensure_ascii=False)


def save_scheduler_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def append_scheduler_event(output_path, event_type, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
