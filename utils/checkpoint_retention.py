from pathlib import Path


def apply_stage_checkpoint_retention(workspace_dir, retention_cfg, checkpoint_catalog):
    workspace_dir = Path(workspace_dir)
    keep_last = max(1, int(retention_cfg.get("keep_last_per_phase", 1)))
    keep_best = bool(retention_cfg.get("keep_best", True))

    retained = []
    grouped = {}
    for item in checkpoint_catalog:
        grouped.setdefault(item.get("phase_name", "default"), []).append(item)

    for phase_items in grouped.values():
        sorted_items = sorted(phase_items, key=lambda item: int(item.get("epoch", 0)))
        retained.extend(sorted_items[-keep_last:])

    if keep_best:
        retained.extend(item for item in checkpoint_catalog if item.get("is_best"))

    retained_paths = {item["checkpoint_path"] for item in retained}
    pruned_paths = []
    for item in checkpoint_catalog:
        relative_path = item["checkpoint_path"]
        if relative_path in retained_paths:
            continue
        checkpoint_path = workspace_dir / relative_path
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            pruned_paths.append(relative_path)

    return {
        "retained_paths": sorted(retained_paths),
        "pruned_paths": sorted(pruned_paths),
    }
