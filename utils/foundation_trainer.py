from pathlib import Path

import pandas as pd

from model.foundation_backbone import build_foundation_backbone
from model.foundation_task_heads import build_task_head
from utils.checkpoint_schema import (
    build_checkpoint_payload,
    load_checkpoint_payload,
    save_checkpoint,
)
from utils.engine_runtime import (
    build_runtime_context,
    execute_training_step,
    maybe_shuffle_epoch_batches,
)
from utils.experiment_registry import build_experiment_manifest, save_experiment_manifest
from utils.foundation_dataset import (
    MultimodalPretrainingDataset,
    build_epoch_batches,
)
from utils.loss_composer import compose_multitask_loss, summarize_loss_components
from utils.run_registry import append_run_record, build_run_record
from utils.foundation_workspace import build_foundation_workspace
from utils.training_components import (
    apply_gradient_accumulation,
    apply_amp_stub,
    apply_gradient_clipping_stub,
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
    start_epoch = 1
    scaler_state = {"amp_enabled": bool(training_cfg.get("use_amp", False))}
    resume_metadata = {"resumed_from": foundation_cfg["training"].get("resume_from_checkpoint")}

    if training_cfg.get("resume_from_checkpoint"):
        checkpoint = load_checkpoint_payload(training_cfg["resume_from_checkpoint"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        optimizer_state = checkpoint.get("optimizer_state", optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state", scheduler_state)
        scaler_state = checkpoint.get("scaler_state", scaler_state)
        resume_metadata = {
            "resumed_from": training_cfg["resume_from_checkpoint"],
            "resume_epoch": checkpoint.get("epoch", 0),
        }

    runtime_context = build_runtime_context(
        foundation_cfg,
        workspace,
        dataset,
        task_heads,
        backbone,
        optimizer_state,
        scheduler_state,
    )

    for epoch in range(start_epoch, int(training_cfg["epochs"]) + 1):
        epoch_batches = build_epoch_batches(dataset, workspace["sampling_plan"])
        epoch_batches = maybe_shuffle_epoch_batches(
            epoch_batches,
            shuffle_enabled=foundation_cfg["sampling"].get("shuffle_tasks_each_epoch", False),
            seed=epoch,
        )
        epoch_metrics = []
        raw_step_metrics = []
        loss_component_rows = []
        for batch_item in epoch_batches:
            step_result = execute_training_step(runtime_context, batch_item)
            task_name = step_result["task_name"]
            output = step_result["task_output"]
            loss_components = compose_multitask_loss(
                task_name,
                output,
                foundation_cfg["tasks"].get("loss_weights", {}),
            )
            clipped = apply_gradient_clipping_stub(
                loss_components["weighted_loss"],
                training_cfg.get("grad_clip_norm"),
            )
            amp_scaled = apply_amp_stub(
                clipped["clipped_loss"],
                enabled=training_cfg.get("use_amp", False),
            )
            global_step += 1
            epoch_metrics.append({"task_name": task_name, **output})
            loss_component_rows.append(loss_components)
            raw_step_metrics.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "task_name": task_name,
                    "backbone_name": backbone.backbone_name,
                    "loss": float(amp_scaled["scaled_loss"]),
                    "base_loss": float(loss_components["base_loss"]),
                    "task_weight": float(loss_components["task_weight"]),
                    "weighted_loss": float(loss_components["weighted_loss"]),
                    "batch_size": int(step_result["batch_size"]),
                    "amp_enabled": bool(amp_scaled["amp_enabled"]),
                    "clip_applied": bool(clipped["clip_applied"]),
                    **{k: float(v) for k, v in output.items() if k != "loss"},
                }
            )
        aggregated_step_metrics = apply_gradient_accumulation(
            raw_step_metrics,
            accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        )
        history.extend(aggregated_step_metrics)

        summary_metrics = summarize_epoch_metrics(epoch_metrics)
        summary_metrics["loss_components"] = summarize_loss_components(loss_component_rows)
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
            scaler_state=scaler_state,
            engine_state={
                "mode": foundation_cfg["experiment"].get("mode", "research"),
                "backbone_name": foundation_cfg["backbone"]["name"],
            },
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
        "mode": foundation_cfg["experiment"].get("mode", "research"),
        "backbone_name": foundation_cfg["backbone"]["name"],
        "epochs": int(training_cfg["epochs"]),
        "global_steps": int(global_step),
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "final_loss_mean": float(history_df["loss"].mean()) if not history_df.empty else 0.0,
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "resume_from_checkpoint": training_cfg.get("resume_from_checkpoint"),
    }
    latest_checkpoint = workspace_dir / "checkpoints" / f"epoch_{int(training_cfg['epochs']):03d}.json"
    run_record = build_run_record(
        experiment_manifest=experiment_manifest,
        training_summary=summary,
        latest_checkpoint=latest_checkpoint,
    )
    append_run_record(workspace_dir, run_record)
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
