from copy import deepcopy
from pathlib import Path

from utils.experiment_registry import build_experiment_manifest, save_experiment_manifest
from utils.failure_recovery import build_recovery_decision
from utils.foundation_workspace import build_foundation_workspace
from utils.run_queue import (
    append_scheduler_event,
    build_run_queue,
    save_run_queue,
    save_scheduler_summary,
)
from utils.stage_planner import build_manifest_aware_stage_plan


def schedule_foundation_experiments(foundation_cfg, workspace_dir, runner=None):
    workspace_dir = Path(workspace_dir)
    scheduler_cfg = foundation_cfg.get("experiment_scheduler", {})
    workspace = build_foundation_workspace(foundation_cfg, workspace_dir / "scheduler_workspace")
    stage_plan = build_manifest_aware_stage_plan(workspace["manifest"], foundation_cfg)
    run_queue = build_run_queue(stage_plan, workspace_dir, scheduler_cfg)

    scheduler_dir = workspace_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    save_run_queue(run_queue, scheduler_dir / "run_queue.json")
    save_scheduler_summary(
        {
            "num_stages": len(stage_plan),
            "stage_plan": stage_plan,
            "queue_size": len(run_queue),
        },
        scheduler_dir / "scheduler_plan.json",
    )

    experiment_manifest = build_experiment_manifest(foundation_cfg, workspace_dir, workspace["summary"])
    save_experiment_manifest(experiment_manifest, scheduler_dir / "scheduler_manifest.json")

    if not scheduler_cfg.get("execute_queue", True):
        return {
            "workspace": workspace,
            "stage_plan": stage_plan,
            "run_queue": run_queue,
            "summary": {"status": "planned", "scheduled_runs": 0, "failed_runs": 0},
        }

    if runner is None:
        from utils.foundation_trainer import run_foundation_training as runner

    event_log_path = scheduler_dir / "scheduler_events.jsonl"
    queue_state = deepcopy(run_queue)
    completed_runs = []
    failed_runs = []

    for queue_index, queue_item in enumerate(queue_state):
        append_scheduler_event(
            event_log_path,
            "run_started",
            {"queue_id": queue_item["queue_id"], "stage_name": queue_item["stage_name"]},
        )
        while True:
            try:
                stage_cfg = build_stage_run_config(foundation_cfg, queue_item)
                result = runner(stage_cfg, queue_item["workspace_dir"])
                queue_item["status"] = "completed"
                queue_item["latest_checkpoint"] = result["summary"].get("latest_checkpoint")
                completed_runs.append(
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "workspace_dir": queue_item["workspace_dir"],
                        "latest_checkpoint": queue_item.get("latest_checkpoint"),
                    }
                )
                append_scheduler_event(
                    event_log_path,
                    "run_completed",
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "latest_checkpoint": queue_item.get("latest_checkpoint"),
                    },
                )
                if queue_index + 1 < len(queue_state) and not queue_state[queue_index + 1].get("resume_from_checkpoint"):
                    queue_state[queue_index + 1]["resume_from_checkpoint"] = queue_item.get("latest_checkpoint")
                break
            except Exception as exc:  # noqa: BLE001
                policy_cfg = scheduler_cfg.get("failure_recovery", {})
                recovery = build_recovery_decision(queue_item, policy_cfg, error_message=str(exc))
                queue_item["last_error"] = str(exc)
                if recovery["action"] != "retry":
                    queue_item["status"] = "failed"
                    failed_runs.append(
                        {
                            "queue_id": queue_item["queue_id"],
                            "stage_name": queue_item["stage_name"],
                            "error": str(exc),
                        }
                    )
                    append_scheduler_event(
                        event_log_path,
                        "run_failed",
                        {
                            "queue_id": queue_item["queue_id"],
                            "stage_name": queue_item["stage_name"],
                            "error": str(exc),
                        },
                    )
                    break

                queue_item["retry_count"] = int(queue_item.get("retry_count", 0)) + 1
                queue_item["status"] = "retrying"
                queue_item["resume_from_checkpoint"] = recovery["resume_from_checkpoint"]
                append_scheduler_event(
                    event_log_path,
                    "run_retrying",
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "retry_count": queue_item["retry_count"],
                        "resume_from_checkpoint": queue_item["resume_from_checkpoint"],
                    },
                )

    save_run_queue(queue_state, scheduler_dir / "run_queue.json")
    scheduler_summary = {
        "status": "completed" if not failed_runs else "completed_with_failures",
        "scheduled_runs": len(completed_runs),
        "failed_runs": len(failed_runs),
        "completed_runs": completed_runs,
        "failed_run_records": failed_runs,
    }
    save_scheduler_summary(scheduler_summary, scheduler_dir / "scheduler_summary.json")
    return {
        "workspace": workspace,
        "stage_plan": stage_plan,
        "run_queue": queue_state,
        "summary": scheduler_summary,
    }


def build_stage_run_config(foundation_cfg, queue_item):
    stage_cfg = deepcopy(foundation_cfg)
    stage_cfg["experiment"] = dict(foundation_cfg["experiment"])
    stage_cfg["experiment"]["name"] = (
        f"{foundation_cfg['experiment']['name']}_{queue_item['stage_id']}_{queue_item['stage_name']}"
    )
    stage_cfg["training"] = dict(foundation_cfg["training"])
    stage_cfg["training"]["epochs"] = int(queue_item["epochs"])
    stage_cfg["training"]["resume_from_checkpoint"] = queue_item.get("resume_from_checkpoint")
    stage_cfg["curriculum"] = {
        "phases": [
            {
                "name": queue_item["stage_name"],
                "start_epoch": 1,
                "epochs": int(queue_item["epochs"]),
                "tasks": queue_item["active_tasks"],
            }
        ]
    }
    return stage_cfg
