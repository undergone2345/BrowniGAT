def build_optimizer_stub(optimizer_cfg, backbone_cfg):
    return {
        "type": optimizer_cfg["name"],
        "learning_rate": float(optimizer_cfg["lr"]),
        "weight_decay": float(optimizer_cfg["weight_decay"]),
        "parameter_groups": backbone_cfg["num_layers"],
    }


def build_scheduler_stub(scheduler_cfg, total_steps):
    return {
        "type": scheduler_cfg["name"],
        "warmup_steps": int(scheduler_cfg["warmup_steps"]),
        "total_steps": int(total_steps),
        "min_lr_ratio": float(scheduler_cfg["min_lr_ratio"]),
    }


def apply_gradient_accumulation(step_metrics, accumulation_steps):
    if accumulation_steps <= 1:
        return step_metrics

    accumulated = []
    chunk = []
    for row in step_metrics:
        chunk.append(row)
        if len(chunk) == accumulation_steps:
            accumulated.append(_merge_chunk(chunk))
            chunk = []
    if chunk:
        accumulated.append(_merge_chunk(chunk))
    return accumulated


def _merge_chunk(chunk):
    merged = dict(chunk[-1])
    merged["loss"] = sum(item["loss"] for item in chunk) / len(chunk)
    merged["micro_steps"] = len(chunk)
    return merged
