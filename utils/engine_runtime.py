import random

from utils.foundation_dataset import collate_foundation_batch


def build_runtime_context(foundation_cfg, workspace, dataset, task_heads, backbone, optimizer_state, scheduler_state):
    return {
        "config": foundation_cfg,
        "workspace": workspace,
        "dataset": dataset,
        "task_heads": task_heads,
        "backbone": backbone,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
    }


def maybe_shuffle_epoch_batches(epoch_batches, shuffle_enabled, seed):
    if not shuffle_enabled:
        return epoch_batches
    rng = random.Random(seed)
    batches = list(epoch_batches)
    rng.shuffle(batches)
    return batches


def execute_training_step(runtime_context, batch_item):
    task_name = batch_item["task_name"]
    head = runtime_context["task_heads"][task_name]
    collated_batch = collate_foundation_batch(batch_item["batch"])
    encoded_batch = runtime_context["backbone"].encode_batch(collated_batch["raw_batch"])
    task_output = head.forward(collated_batch["raw_batch"], encoded_batch=encoded_batch)
    return {
        "task_name": task_name,
        "batch_size": int(collated_batch["batch_size"]),
        "encoded_batch": encoded_batch,
        "task_output": task_output,
    }
