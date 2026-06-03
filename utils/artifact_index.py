from pathlib import Path
import json


def _normalize_workspace_path(workspace_dir, candidate_path):
    if not candidate_path:
        return None
    candidate = Path(candidate_path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(workspace_dir).as_posix()
    except ValueError:
        return str(candidate)


def build_artifact_index(workspace_dir, training_summary):
    workspace_dir = Path(workspace_dir)
    checkpoints_dir = workspace_dir / "checkpoints"
    checkpoint_files = sorted(
        [path.name for path in checkpoints_dir.glob("*.json")] if checkpoints_dir.exists() else []
    )

    index = {
        "workspace_dir": str(workspace_dir.resolve()),
        "history_file": "training_history.csv" if (workspace_dir / "training_history.csv").exists() else None,
        "summary_file": "training_summary.json" if (workspace_dir / "training_summary.json").exists() else None,
        "experiment_manifest": "experiment_manifest.json" if (workspace_dir / "experiment_manifest.json").exists() else None,
        "run_registry": "run_registry.jsonl" if (workspace_dir / "run_registry.jsonl").exists() else None,
        "checkpoint_files": checkpoint_files,
        "best_checkpoint": _normalize_workspace_path(
            workspace_dir,
            training_summary.get("best_checkpoint")
            or ("checkpoints/best.json" if "best.json" in checkpoint_files else None),
        ),
        "final_checkpoint": _normalize_workspace_path(
            workspace_dir,
            training_summary.get("latest_checkpoint"),
        ),
        "completed_epochs": int(training_summary.get("completed_epochs", training_summary.get("epochs", 0))),
    }
    return index


def save_artifact_index(index, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, ensure_ascii=False)
