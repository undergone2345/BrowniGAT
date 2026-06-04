from pathlib import Path


def build_recovery_decision(queue_item, policy_cfg, error_message=None):
    max_retries = int(policy_cfg.get("max_retries", 0))
    retry_count = int(queue_item.get("retry_count", 0))
    if retry_count >= max_retries:
        return {
            "action": "fail",
            "resume_from_checkpoint": None,
            "reason": error_message or "retry_budget_exhausted",
        }

    stage_workspace = Path(queue_item["workspace_dir"])
    resume_strategy = policy_cfg.get("resume_strategy", "latest")
    resume_checkpoint = _resolve_resume_checkpoint(stage_workspace, resume_strategy)
    return {
        "action": "retry",
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "reason": error_message or "retry_allowed",
    }


def _resolve_resume_checkpoint(stage_workspace, resume_strategy):
    checkpoints_dir = stage_workspace / "checkpoints"
    if resume_strategy == "best":
        best_path = checkpoints_dir / "best.json"
        if best_path.exists():
            return best_path

    candidates = sorted(checkpoints_dir.glob("epoch_*.json"))
    if candidates:
        return candidates[-1]

    best_path = checkpoints_dir / "best.json"
    if best_path.exists():
        return best_path
    return None
