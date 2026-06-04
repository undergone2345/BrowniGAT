from pathlib import Path

import pandas as pd

from model.foundation_backbone import build_foundation_backbone
from utils.artifact_index import build_artifact_index, save_artifact_index
from utils.checkpoint_catalog import load_checkpoint_catalog
from utils.checkpoint_catalog import update_checkpoint_catalog
from utils.checkpoint_retention import apply_stage_checkpoint_retention
from utils.curriculum_schedule import build_curriculum_state
from utils.data_sharding import build_data_shard
from utils.event_log import append_event, count_events
from model.foundation_task_heads import build_task_head
from utils.checkpoint_schema import (
    build_checkpoint_payload,
    load_checkpoint_payload,
    save_checkpoint,
)
from utils.engine_dataloader import build_engine_dataloader
from utils.engine_dataloader import describe_dataloader
from utils.early_stopping import initialize_early_stopping, update_early_stopping
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
from utils.reporting import save_run_metadata
from utils.runtime_topology import build_runtime_topology
from utils.run_registry import append_run_record, build_run_record
from utils.sampler_plan import build_task_sampling_sequence
from utils.validation_registry import build_validation_batches, summarize_validation_outputs
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
    runtime_topology = build_runtime_topology(foundation_cfg.get("distributed", {}))
    workspace = build_foundation_workspace(foundation_cfg, workspace_dir)
    dataset = MultimodalPretrainingDataset(workspace_dir, runtime_topology=runtime_topology)
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
    best_validation_loss = float("inf")
    early_stopping_state = initialize_early_stopping(
        patience=foundation_cfg.get("early_stopping", {}).get("patience", 0),
        min_delta=foundation_cfg.get("early_stopping", {}).get("min_delta", 0.0),
    )
    last_completed_epoch = max(0, start_epoch - 1)
    latest_checkpoint_path = None
    best_checkpoint_path = None
    event_log_path = workspace_dir / "events" / "training_events.jsonl"

    history_path = workspace_dir / "training_history.csv"

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
        best_validation_loss = float(
            checkpoint.get("validation_metrics", {}).get("overall_validation_loss", best_validation_loss)
        )
        early_stopping_state = checkpoint.get("early_stopping_state", early_stopping_state)
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient="records")

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
        curriculum_state = build_curriculum_state(
            [task_spec["name"] for task_spec in workspace["manifest"]["enabled_tasks"]],
            foundation_cfg.get("curriculum", {}),
            epoch,
        )
        append_event(
            event_log_path,
            "epoch_start",
            {
                "epoch": epoch,
                "phase_name": curriculum_state["phase_name"],
                "active_tasks": curriculum_state["active_tasks"],
            },
        )
        active_sampling_plan = [
            item for item in workspace["sampling_plan"]
            if item["task_name"] in set(curriculum_state["active_tasks"])
        ]
        task_sequence = build_task_sampling_sequence(
            active_sampling_plan,
            repeat_strategy=foundation_cfg.get("sampling", {}).get("repeat_strategy", "round_robin"),
            worker_index=0,
            num_workers=1,
        )
        epoch_batches = build_epoch_batches(dataset, active_sampling_plan, task_sequence=task_sequence)
        epoch_batches = maybe_shuffle_epoch_batches(
            epoch_batches,
            shuffle_enabled=foundation_cfg["sampling"].get("shuffle_tasks_each_epoch", False),
            seed=epoch,
        )
        train_loader = build_engine_dataloader(epoch_batches, foundation_cfg.get("dataloader", {}))
        epoch_metrics = []
        raw_step_metrics = []
        loss_component_rows = []
        for batch_item in train_loader:
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
            if global_step % int(training_cfg.get("log_every_n_steps", 50)) == 0:
                print(
                    "[{mode}] step={step} task={task} loss={loss:.4f}".format(
                        mode=foundation_cfg["experiment"].get("mode", "research"),
                        step=global_step,
                        task=task_name,
                        loss=float(amp_scaled["scaled_loss"]),
                    )
                )
        aggregated_step_metrics = apply_gradient_accumulation(
            raw_step_metrics,
            accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        )
        history.extend(aggregated_step_metrics)

        summary_metrics = summarize_epoch_metrics(epoch_metrics)
        summary_metrics["loss_components"] = summarize_loss_components(loss_component_rows)
        validation_batches = build_validation_batches(
            dataset,
            [
                task_spec for task_spec in workspace["manifest"]["enabled_tasks"]
                if task_spec["name"] in set(curriculum_state["active_tasks"])
            ],
            max_batches_per_task=int(foundation_cfg.get("validation", {}).get("max_batches_per_task", 2)),
        )
        validation_rows = []
        for batch_item in validation_batches:
            step_result = execute_training_step(runtime_context, batch_item)
            validation_rows.append(
                {
                    "task_name": batch_item["task_name"],
                    "loss": float(step_result["task_output"]["loss"]),
                }
            )
        validation_metrics = summarize_validation_outputs(validation_rows)
        is_best = validation_metrics["overall_validation_loss"] <= best_validation_loss
        best_validation_loss = min(best_validation_loss, validation_metrics["overall_validation_loss"])
        early_stopping_state = update_early_stopping(
            early_stopping_state,
            current_score=validation_metrics["selection_metric"],
            epoch=epoch,
            mode=foundation_cfg.get("early_stopping", {}).get("mode", "min"),
        )
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
                "runtime_topology": runtime_topology,
            },
            validation_metrics=validation_metrics,
            is_best=is_best,
            early_stopping_state=early_stopping_state,
            curriculum_state=curriculum_state,
            data_shard_state={
                "world_size": runtime_topology.get("world_size", 1),
                "global_rank": runtime_topology.get("global_rank", 0),
                "task_shards": dataset.get_shard_summary(),
            },
        )
        if epoch % int(training_cfg["checkpoint_every"]) == 0:
            checkpoint_path = workspace_dir / "checkpoints" / f"epoch_{epoch:03d}.json"
            save_checkpoint(checkpoint_payload, checkpoint_path)
            latest_checkpoint_path = checkpoint_path
            if is_best:
                best_checkpoint_path = workspace_dir / "checkpoints" / "best.json"
                save_checkpoint(checkpoint_payload, best_checkpoint_path)
            checkpoint_catalog = update_checkpoint_catalog(
                workspace_dir,
                {
                    "epoch": epoch,
                    "checkpoint_path": f"checkpoints/epoch_{epoch:03d}.json",
                    "is_best": bool(is_best),
                    "phase_name": curriculum_state["phase_name"],
                },
            )
            retention_summary = apply_stage_checkpoint_retention(
                workspace_dir,
                foundation_cfg.get("checkpoint_retention", {}),
                checkpoint_catalog,
            )
            append_event(
                event_log_path,
                "checkpoint_saved",
                {
                    "epoch": epoch,
                    "checkpoint_path": f"checkpoints/epoch_{epoch:03d}.json",
                    "is_best": bool(is_best),
                    "retention_summary": retention_summary,
                },
            )
        last_completed_epoch = epoch
        append_event(
            event_log_path,
            "epoch_end",
            {
                "epoch": epoch,
                "overall_validation_loss": validation_metrics["overall_validation_loss"],
                "phase_name": curriculum_state["phase_name"],
            },
        )
        if early_stopping_state.get("should_stop", False):
            append_event(
                event_log_path,
                "early_stop_triggered",
                {
                    "epoch": epoch,
                    "best_epoch": early_stopping_state.get("best_epoch"),
                    "best_score": early_stopping_state.get("best_score"),
                },
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    if latest_checkpoint_path is None and last_completed_epoch > 0:
        fallback_checkpoint = workspace_dir / "checkpoints" / f"epoch_{last_completed_epoch:03d}.json"
        if fallback_checkpoint.exists():
            latest_checkpoint_path = fallback_checkpoint
    if best_checkpoint_path is None:
        candidate_best = workspace_dir / "checkpoints" / "best.json"
        if candidate_best.exists():
            best_checkpoint_path = candidate_best

    summary = {
        **workspace["summary"],
        "mode": foundation_cfg["experiment"].get("mode", "research"),
        "backbone_name": foundation_cfg["backbone"]["name"],
        "epochs": int(training_cfg["epochs"]),
        "completed_epochs": int(last_completed_epoch),
        "global_steps": int(global_step),
        "checkpoint_dir": str(workspace_dir / "checkpoints"),
        "final_loss_mean": float(history_df["loss"].mean()) if not history_df.empty else 0.0,
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "resume_from_checkpoint": training_cfg.get("resume_from_checkpoint"),
        "best_validation_loss": float(best_validation_loss) if best_validation_loss != float("inf") else 0.0,
        "best_epoch": early_stopping_state.get("best_epoch"),
        "early_stopped": bool(early_stopping_state.get("should_stop", False)),
        "latest_checkpoint": str(latest_checkpoint_path) if latest_checkpoint_path else None,
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
        "dataloader": describe_dataloader(
            foundation_cfg.get("dataloader", {}),
            total_batches=sum(item["steps"] for item in workspace["sampling_plan"]),
        ),
        "sampling_strategy": foundation_cfg.get("sampling", {}).get("repeat_strategy", "round_robin"),
        "runtime_topology": runtime_topology,
        "data_sharding": {
            "world_size": runtime_topology.get("world_size", 1),
            "global_rank": runtime_topology.get("global_rank", 0),
            "task_shards": dataset.get_shard_summary(),
        },
        "manifest_partition_summary": workspace["manifest"].get("partition_summary", {}),
        "curriculum_phases": foundation_cfg.get("curriculum", {}).get("phases", []),
        "checkpoint_catalog": "checkpoints/checkpoint_index.json",
        "event_log": "events/training_events.jsonl" if event_log_path.exists() else None,
        "event_count": count_events(event_log_path),
    }
    save_run_metadata(summary, workspace_dir / "training_summary.json")
    run_record = build_run_record(
        experiment_manifest=experiment_manifest,
        training_summary=summary,
        latest_checkpoint=summary["latest_checkpoint"] or "",
    )
    append_run_record(workspace_dir, run_record)
    artifact_index = build_artifact_index(workspace_dir, summary)
    save_artifact_index(artifact_index, workspace_dir / "artifact_index.json")
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
