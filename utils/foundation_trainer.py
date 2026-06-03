from pathlib import Path

import pandas as pd

from model.foundation_task_heads import build_task_head
from utils.batch_sampler import build_sampling_plan
from utils.checkpoint_schema import build_checkpoint_payload, save_checkpoint
from utils.foundation_dataset import MultimodalPretrainingDataset, build_epoch_batches
from utils.foundation_workspace import build_foundation_workspace


def run_foundation_training(foundation_cfg, workspace_dir):
    workspace_dir = Path(workspace_dir)
    workspace = build_foundation_workspace(foundation_cfg, workspace_dir)
    dataset = MultimodalPretrainingDataset(workspace_dir)
    task_heads = {
        task_spec["name"]: build_task_head(task_spec)
        for task_spec in workspace["manifest"]["enabled_tasks"]
    }

    training_cfg = foundation_cfg["training"]
    history = []
    global_step = 0

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        epoch_batches = build_epoch_batches(dataset, workspace["sampling_plan"])
        epoch_metrics = []
        for batch_item in epoch_batches:
            task_name = batch_item["task_name"]
            head = task_heads[task_name]
            output = head.forward(batch_item["batch"])
            global_step += 1
            epoch_metrics.append({"task_name": task_name, **output})
            history.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "task_name": task_name,
                    "loss": float(output["loss"]),
                    **{k: float(v) for k, v in output.items() if k != "loss"},
                }
            )

        summary_metrics = summarize_epoch_metrics(epoch_metrics)
        checkpoint_payload = build_checkpoint_payload(
            epoch=epoch,
            global_step=global_step,
            metrics=summary_metrics,
            config_snapshot=foundation_cfg,
            manifest_path=workspace_dir / "pretraining_manifest.json",
            sampling_plan=workspace["sampling_plan"],
        )
        save_checkpoint(
            checkpoint_payload,
            workspace_dir / "checkpoints" / f"epoch_{epoch:03d}.json",
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(workspace_dir / "training_history.csv", index=False)

    summary = {
        **workspace["summary"],
        "epochs": int(training_cfg["epochs"]),
        "global_steps": int(global_step),
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "final_loss_mean": float(history_df["loss"].mean()) if not history_df.empty else 0.0,
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
