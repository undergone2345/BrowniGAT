from pathlib import Path
import json


def update_checkpoint_catalog(workspace_dir, checkpoint_record):
    workspace_dir = Path(workspace_dir)
    catalog_path = workspace_dir / "checkpoints" / "checkpoint_index.json"
    existing = load_checkpoint_catalog(catalog_path)
    existing.append(checkpoint_record)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, ensure_ascii=False)
    return existing


def load_checkpoint_catalog(catalog_path):
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return []
    with catalog_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
