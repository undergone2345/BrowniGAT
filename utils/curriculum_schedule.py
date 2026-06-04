def build_curriculum_state(enabled_tasks, curriculum_cfg, epoch):
    enabled_tasks = list(enabled_tasks)
    phases = list(curriculum_cfg.get("phases", []))
    if not phases:
        return {
            "active_tasks": enabled_tasks,
            "phase_name": "all_tasks",
            "phase_index": 0,
        }

    selected_phase = phases[-1]
    selected_index = len(phases) - 1
    for index, phase in enumerate(phases):
        start_epoch = int(phase.get("start_epoch", 1))
        end_epoch = phase.get("end_epoch")
        if epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > int(end_epoch):
            continue
        selected_phase = phase
        selected_index = index

    phase_tasks = selected_phase.get("tasks", enabled_tasks)
    active_tasks = [task for task in enabled_tasks if task in set(phase_tasks)]
    return {
        "active_tasks": active_tasks or enabled_tasks,
        "phase_name": selected_phase.get("name", f"phase_{selected_index + 1}"),
        "phase_index": selected_index,
    }
