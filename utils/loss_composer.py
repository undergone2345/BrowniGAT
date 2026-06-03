def compose_multitask_loss(task_name, task_output, loss_weights):
    base_loss = float(task_output["loss"])
    task_weight = float(loss_weights.get(task_name, 1.0))
    weighted_loss = base_loss * task_weight
    return {
        "base_loss": base_loss,
        "task_weight": task_weight,
        "weighted_loss": weighted_loss,
    }


def summarize_loss_components(rows):
    if not rows:
        return {"mean_weighted_loss": 0.0, "mean_base_loss": 0.0}
    mean_weighted_loss = sum(row["weighted_loss"] for row in rows) / len(rows)
    mean_base_loss = sum(row["base_loss"] for row in rows) / len(rows)
    return {
        "mean_weighted_loss": float(mean_weighted_loss),
        "mean_base_loss": float(mean_base_loss),
    }
