def build_task_sampling_sequence(sampling_plan, repeat_strategy="round_robin"):
    sequence = []
    if repeat_strategy == "round_robin":
        max_steps = max((item["steps"] for item in sampling_plan), default=0)
        for step_idx in range(max_steps):
            for item in sampling_plan:
                if step_idx < item["steps"]:
                    sequence.append(
                        {
                            "task_name": item["task_name"],
                            "step_index": step_idx,
                            "batch_size": item["batch_size"],
                        }
                    )
        return sequence

    for item in sampling_plan:
        for step_idx in range(item["steps"]):
            sequence.append(
                {
                    "task_name": item["task_name"],
                    "step_index": step_idx,
                    "batch_size": item["batch_size"],
                }
            )
    return sequence
