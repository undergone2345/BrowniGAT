import math


def build_sampling_plan(enabled_tasks, task_batch_sizes, steps_per_epoch, temperature=1.0):
    if not enabled_tasks:
        return []

    weights = []
    for task_spec in enabled_tasks:
        batch_size = int(task_batch_sizes.get(task_spec["name"], 128))
        weights.append(max(batch_size, 1) ** float(temperature))

    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    plan = []
    remaining_steps = steps_per_epoch
    for idx, task_spec in enumerate(enabled_tasks):
        allocated = max(1, int(round(normalized_weights[idx] * steps_per_epoch)))
        plan.append(
            {
                "task_name": task_spec["name"],
                "batch_size": int(task_batch_sizes.get(task_spec["name"], 128)),
                "steps": allocated,
            }
        )
        remaining_steps -= allocated

    step_index = 0
    while remaining_steps != 0 and plan:
        target_idx = step_index % len(plan)
        if remaining_steps > 0:
            plan[target_idx]["steps"] += 1
            remaining_steps -= 1
        elif plan[target_idx]["steps"] > 1:
            plan[target_idx]["steps"] -= 1
            remaining_steps += 1
        step_index += 1

    return sorted(plan, key=lambda item: (-item["steps"], item["task_name"]))
