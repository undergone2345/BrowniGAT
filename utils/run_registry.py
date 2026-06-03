from datetime import datetime, timezone
from pathlib import Path
import json


def build_run_record(experiment_manifest, training_summary, latest_checkpoint):
    mode = (
        experiment_manifest.get("config_snapshot", {})
        .get("experiment", {})
        .get("mode", "research")
    )
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_manifest["experiment_name"],
        "mode": mode,
        "backbone": experiment_manifest["backbone"]["name"],
        "epochs": training_summary["epochs"],
        "global_steps": training_summary["global_steps"],
        "final_loss_mean": training_summary["final_loss_mean"],
        "latest_checkpoint": str(latest_checkpoint),
    }


def append_run_record(output_dir, run_record):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "run_registry.jsonl"
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run_record, ensure_ascii=False) + "\n")
