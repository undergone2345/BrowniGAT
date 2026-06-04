from copy import deepcopy
from pathlib import Path

from utils.experiment_registry import build_experiment_manifest, save_experiment_manifest
from utils.failure_recovery import build_recovery_decision
from utils.foundation_workspace import build_foundation_workspace
from utils.promotion_policy import evaluate_promotion
from utils.resource_scheduler import (
    allocate_resources,
    build_resource_state,
    can_allocate_resources,
    prioritize_queue_items,
    release_resources,
)
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
    skipped_runs = []
    completed_queue_ids = set()
    halt_requested = False
    resource_state = build_resource_state(scheduler_cfg.get("resources", {}))

    while True:
        pending_items = [
            item for item in queue_state
            if item.get("status", "pending") in {"pending", "retrying"}
        ]
        if not pending_items:
            break

        if halt_requested:
            for queue_item in pending_items:
                queue_item["status"] = "skipped_promotion"
                skipped_runs.append(
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "reason": "promotion_gate_halted_scheduler",
                    }
                )
            break

        progress_made = False
        for queue_item in prioritize_queue_items(pending_items):
            dependency_id = queue_item.get("depends_on")
            if dependency_id and dependency_id not in completed_queue_ids:
                continue

            if not can_allocate_resources(resource_state, queue_item.get("resource_request", {})):
                continue

            allocation = allocate_resources(resource_state, queue_item.get("resource_request", {}))
            queue_item["resource_allocation"] = allocation
            append_scheduler_event(
                event_log_path,
                "run_started",
                {
                    "queue_id": queue_item["queue_id"],
                    "stage_name": queue_item["stage_name"],
                    "priority": queue_item.get("priority"),
                    "resource_allocation": allocation,
                },
            )
            while True:
                try:
                    stage_cfg = build_stage_run_config(foundation_cfg, queue_item)
                    result = runner(stage_cfg, queue_item["workspace_dir"])
                    queue_item["status"] = "completed"
                    queue_item["latest_checkpoint"] = result["summary"].get("latest_checkpoint")
                    queue_item["training_summary"] = result["summary"]
                    queue_item["promotion_decision"] = evaluate_promotion(
                        result["summary"],
                        scheduler_cfg.get("promotion_policy", {}),
                    )
                    completed_runs.append(
                        {
                            "queue_id": queue_item["queue_id"],
                            "stage_name": queue_item["stage_name"],
                            "workspace_dir": queue_item["workspace_dir"],
                            "latest_checkpoint": queue_item.get("latest_checkpoint"),
                            "promotion_decision": queue_item["promotion_decision"],
                        }
                    )
                    append_scheduler_event(
                        event_log_path,
                        "run_completed",
                        {
                            "queue_id": queue_item["queue_id"],
                            "stage_name": queue_item["stage_name"],
                            "latest_checkpoint": queue_item.get("latest_checkpoint"),
                            "promotion_decision": queue_item["promotion_decision"],
                        },
                    )
                    completed_queue_ids.add(queue_item["queue_id"])
                    for downstream_item in queue_state:
                        if (
                            downstream_item.get("depends_on") == queue_item["queue_id"]
                            and not downstream_item.get("resume_from_checkpoint")
                        ):
                            downstream_item["resume_from_checkpoint"] = queue_item.get("latest_checkpoint")
                    if not queue_item["promotion_decision"]["passed"]:
                        halt_requested = queue_item["promotion_decision"]["action"] in {"halt", "rollback"}
                        if queue_item["promotion_decision"]["action"] == "rollback":
                            queue_item["rollback_to_checkpoint"] = queue_item.get("resume_from_checkpoint")
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
            release_resources(resource_state, allocation)
            progress_made = True
            break

        if progress_made:
            continue

        unresolved = prioritize_queue_items(pending_items)
        for queue_item in unresolved:
            if queue_item.get("status") not in {"pending", "retrying"}:
                continue
            dependency_id = queue_item.get("depends_on")
            if dependency_id and dependency_id not in completed_queue_ids:
                queue_item["status"] = "skipped_dependency"
                skipped_runs.append(
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "reason": f"dependency_not_completed:{dependency_id}",
                    }
                )
            else:
                queue_item["status"] = "blocked_resources"
                failed_runs.append(
                    {
                        "queue_id": queue_item["queue_id"],
                        "stage_name": queue_item["stage_name"],
                        "error": "insufficient_resources",
                    }
                )
        break

    save_run_queue(queue_state, scheduler_dir / "run_queue.json")
    scheduler_summary = {
        "status": "completed" if not failed_runs else "completed_with_failures",
        "scheduled_runs": len(completed_runs),
        "failed_runs": len(failed_runs),
        "skipped_runs": len(skipped_runs),
        "completed_runs": completed_runs,
        "failed_run_records": failed_runs,
        "skipped_run_records": skipped_runs,
        "resource_state": resource_state,
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
