from datetime import datetime, timezone
from pathlib import Path
import json


def build_experiment_manifest(foundation_cfg, workspace_dir, workspace_summary):
    return {
        "experiment_name": foundation_cfg["experiment"]["name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_dir": str(Path(workspace_dir).resolve()),
        "backbone": foundation_cfg["backbone"],
        "optimizer": foundation_cfg["optimizer"],
        "scheduler": foundation_cfg["scheduler"],
        "training": foundation_cfg["training"],
        "workspace_summary": workspace_summary,
    }


def save_experiment_manifest(manifest, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
