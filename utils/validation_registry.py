import pandas as pd


def build_validation_batches(dataset, enabled_tasks, max_batches_per_task=2):
    validation_batches = []
    for task_spec in enabled_tasks:
        task_name = task_spec["name"]
        task_samples = dataset.get_task_samples(task_name)
        if not task_samples:
            continue
        half = max(1, len(task_samples) // 5)
        val_split = task_samples[-half:]
        if not val_split:
            continue
        batch_size = max(1, min(len(val_split), max(1, len(val_split) // max_batches_per_task)))
        for start_idx in range(0, len(val_split), batch_size):
            validation_batches.append(
                {
                    "task_name": task_name,
                    "batch": val_split[start_idx : start_idx + batch_size],
                }
            )
    return validation_batches


def summarize_validation_outputs(validation_rows):
    if not validation_rows:
        return {"overall_validation_loss": 0.0, "task_validation_losses": []}
    frame = pd.DataFrame(validation_rows)
    grouped = frame.groupby("task_name", as_index=False)["loss"].mean()
    return {
        "overall_validation_loss": float(frame["loss"].mean()),
        "task_validation_losses": grouped.to_dict(orient="records"),
    }
