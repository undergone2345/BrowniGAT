from datetime import datetime, timezone
from pathlib import Path
import json


def build_checkpoint_payload(
    epoch,
    global_step,
    metrics,
    config_snapshot,
    manifest_path,
    sampling_plan,
    optimizer_state=None,
    scheduler_state=None,
    resume_metadata=None,
    scaler_state=None,
    engine_state=None,
    validation_metrics=None,
    is_best=False,
):
    return {
        "checkpoint_version": 1,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics,
        "config_snapshot": config_snapshot,
        "manifest_path": str(manifest_path),
        "sampling_plan": sampling_plan,
        "optimizer_state": optimizer_state or {},
        "scheduler_state": scheduler_state or {},
        "resume_metadata": resume_metadata or {},
        "scaler_state": scaler_state or {},
        "engine_state": engine_state or {},
        "validation_metrics": validation_metrics or {},
        "is_best": bool(is_best),
    }


def load_checkpoint_payload(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
