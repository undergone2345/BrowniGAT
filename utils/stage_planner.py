def build_manifest_aware_stage_plan(manifest, foundation_cfg):
    scheduler_cfg = foundation_cfg.get("experiment_scheduler", {})
    curriculum_phases = foundation_cfg.get("curriculum", {}).get("phases", [])
    enabled_tasks = [task_spec["name"] for task_spec in manifest.get("enabled_tasks", [])]
    partition_summary = manifest.get("partition_summary", {})

    if not curriculum_phases:
        curriculum_phases = [
            {
                "name": "default_stage",
                "start_epoch": 1,
                "tasks": enabled_tasks,
            }
        ]

    default_epochs = int(scheduler_cfg.get("default_stage_epochs", 1))
    stage_plan = []
    for phase_index, phase in enumerate(curriculum_phases, start=1):
        active_tasks = phase.get("tasks", enabled_tasks)
        epoch_span = _resolve_phase_epochs(phase, default_epochs)
        stage_plan.append(
            {
                "stage_id": f"stage_{phase_index:02d}",
                "stage_name": phase.get("name", f"stage_{phase_index:02d}"),
                "active_tasks": active_tasks,
                "epochs": epoch_span,
                "priority": int(phase.get("priority", _infer_stage_priority(active_tasks))),
                "stage_kind": _infer_stage_kind(active_tasks),
                "partition_budget": _estimate_partition_budget(active_tasks, partition_summary),
                "partition_summary": partition_summary,
                "resource_request": _infer_resource_request(active_tasks, phase.get("resource_request", {})),
                "depends_on": phase.get("depends_on"),
            }
        )
    return stage_plan


def _resolve_phase_epochs(phase, default_epochs):
    if "epochs" in phase:
        return max(1, int(phase["epochs"]))
    start_epoch = int(phase.get("start_epoch", 1))
    end_epoch = phase.get("end_epoch")
    if end_epoch is not None:
        return max(1, int(end_epoch) - start_epoch + 1)
    return max(1, int(default_epochs))


def _estimate_partition_budget(active_tasks, partition_summary):
    structural_tasks = {"masked_node_modeling", "masked_edge_modeling"}
    if set(active_tasks).issubset(structural_tasks):
        return int(partition_summary.get("node_partitions", 0) + partition_summary.get("edge_partitions", 0))
    return int(partition_summary.get("total_partitions", 0))


def _infer_stage_kind(active_tasks):
    structural_tasks = {"masked_node_modeling", "masked_edge_modeling"}
    return "structural_warmup" if set(active_tasks).issubset(structural_tasks) else "multimodal_expansion"


def _infer_stage_priority(active_tasks):
    return 40 if _infer_stage_kind(active_tasks) == "structural_warmup" else 80


def _infer_resource_request(active_tasks, resource_override):
    request = {
        "cpu_slots": 1 if _infer_stage_kind(active_tasks) == "structural_warmup" else 2,
        "gpu_slots": 0 if _infer_stage_kind(active_tasks) == "structural_warmup" else 1,
    }
    request.update(resource_override)
    return request
