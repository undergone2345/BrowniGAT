def build_task_sampling_sequence(
    sampling_plan,
    repeat_strategy="round_robin",
    worker_index=0,
    num_workers=1,
):
    sequence = []
    num_workers = max(1, int(num_workers))
    worker_index = max(0, int(worker_index))
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
        return _apply_worker_stride(sequence, worker_index, num_workers)

    for item in sampling_plan:
        for step_idx in range(item["steps"]):
            sequence.append(
                {
                    "task_name": item["task_name"],
                    "step_index": step_idx,
                    "batch_size": item["batch_size"],
                }
            )
    return _apply_worker_stride(sequence, worker_index, num_workers)


def _apply_worker_stride(sequence, worker_index, num_workers):
    if num_workers <= 1:
        return sequence
    return [item for idx, item in enumerate(sequence) if idx % num_workers == worker_index]
