from pathlib import Path

import pandas as pd

from model.foundation_backbone import build_foundation_backbone
from model.foundation_task_heads import build_task_head
from utils.checkpoint_schema import build_checkpoint_payload, save_checkpoint
from utils.experiment_registry import build_experiment_manifest, save_experiment_manifest
from utils.foundation_dataset import (
    MultimodalPretrainingDataset,
    build_epoch_batches,
    collate_foundation_batch,
)
from utils.foundation_workspace import build_foundation_workspace
from utils.training_components import (
    apply_gradient_accumulation,
    build_optimizer_stub,
    build_scheduler_stub,
)


def run_foundation_training(foundation_cfg, workspace_dir):
    workspace_dir = Path(workspace_dir)
    workspace = build_foundation_workspace(foundation_cfg, workspace_dir)
    dataset = MultimodalPretrainingDataset(workspace_dir)
    backbone = build_foundation_backbone(foundation_cfg["backbone"])
    task_heads = {
        task_spec["name"]: build_task_head(task_spec)
        for task_spec in workspace["manifest"]["enabled_tasks"]
    }
    experiment_manifest = build_experiment_manifest(
        foundation_cfg,
        workspace_dir,
        workspace["summary"],
    )
    save_experiment_manifest(experiment_manifest, workspace_dir / "experiment_manifest.json")

    training_cfg = foundation_cfg["training"]
    total_steps = int(training_cfg["epochs"]) * sum(item["steps"] for item in workspace["sampling_plan"])
    optimizer_state = build_optimizer_stub(foundation_cfg["optimizer"], foundation_cfg["backbone"])
    scheduler_state = build_scheduler_stub(foundation_cfg["scheduler"], total_steps=total_steps)
    history = []
    global_step = 0
    resume_metadata = {"resumed_from": foundation_cfg["training"].get("resume_from_checkpoint")}

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        epoch_batches = build_epoch_batches(dataset, workspace["sampling_plan"])
        epoch_metrics = []
        raw_step_metrics = []
        for batch_item in epoch_batches:
            task_name = batch_item["task_name"]
            head = task_heads[task_name]
            collated_batch = collate_foundation_batch(batch_item["batch"])
            encoded_batch = backbone.encode_batch(collated_batch["raw_batch"])
            output = head.forward(collated_batch["raw_batch"], encoded_batch=encoded_batch)
            global_step += 1
            epoch_metrics.append({"task_name": task_name, **output})
            raw_step_metrics.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "task_name": task_name,
                    "backbone_name": backbone.backbone_name,
                    "loss": float(output["loss"]),
                    "batch_size": int(collated_batch["batch_size"]),
                    **{k: float(v) for k, v in output.items() if k != "loss"},
                }
            )
        aggregated_step_metrics = apply_gradient_accumulation(
            raw_step_metrics,
            accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        )
        history.extend(aggregated_step_metrics)

        summary_metrics = summarize_epoch_metrics(epoch_metrics)
        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            global_step=global_step,
            metrics=summary_metrics,
            config_snapshot=foundation_cfg,
            manifest_path=workspace_dir / "pretraining_manifest.json",
            sampling_plan=workspace["sampling_plan"],
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            resume_metadata=resume_metadata,
        )
        if epoch % int(training_cfg["checkpoint_every"]) == 0:
            save_checkpoint(
                checkpoint_payload,
                workspace_dir / "checkpoints" / f"epoch_{epoch:03d}.json",
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(workspace_dir / "training_history.csv", index=False)

    summary = {
        **workspace["summary"],
        "backbone_name": foundation_cfg["backbone"]["name"],
        "epochs": int(training_cfg["epochs"]),
        "global_steps": int(global_step),
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "final_loss_mean": float(history_df["loss"].mean()) if not history_df.empty else 0.0,
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
    }
    return {
        "workspace": workspace,
        "history_df": history_df,
        "summary": summary,
    }


def summarize_epoch_metrics(epoch_metrics):
    metric_table = pd.DataFrame(epoch_metrics)
    grouped = metric_table.groupby("task_name", as_index=False)["loss"].mean()
    return {
        "task_mean_losses": grouped.to_dict(orient="records"),
        "overall_loss_mean": float(metric_table["loss"].mean()) if not metric_table.empty else 0.0,
    }
