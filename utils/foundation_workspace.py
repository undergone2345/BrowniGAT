from pathlib import Path

from utils.batch_sampler import build_sampling_plan
from utils.pretraining_manifest import build_pretraining_manifest, save_pretraining_manifest
from utils.task_registry import resolve_enabled_tasks


def build_foundation_workspace(foundation_cfg, output_dir):
    source_bundle_dir = Path(foundation_cfg["source_bundle_dir"]).expanduser().resolve()
    enabled_modalities = foundation_cfg["modalities"]["include"]
    enabled_tasks = resolve_enabled_tasks(
        foundation_cfg["tasks"]["enabled"],
        available_modalities=enabled_modalities,
    )
    manifest = build_pretraining_manifest(
        bundle_dir=source_bundle_dir,
        enabled_modalities=enabled_modalities,
        enabled_tasks=enabled_tasks,
        tokenizer_cfg=foundation_cfg["tokenizer"],
    )
    sampling_plan = build_sampling_plan(
        enabled_tasks=enabled_tasks,
        task_batch_sizes=foundation_cfg["tasks"]["batch_size_per_task"],
        steps_per_epoch=int(foundation_cfg["sampling"]["steps_per_epoch"]),
        temperature=float(foundation_cfg["sampling"]["task_temperature"]),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pretraining_manifest(manifest, output_dir / "pretraining_manifest.json")

    summary = {
        "source_bundle_dir": str(source_bundle_dir),
        "num_modalities": len(manifest["available_modalities"]),
        "num_enabled_tasks": len(enabled_tasks),
        "steps_per_epoch": int(foundation_cfg["sampling"]["steps_per_epoch"]),
        "sampling_plan": sampling_plan,
        "vocab_sizes": manifest["vocab_sizes"],
    }
    return {
        "manifest": manifest,
        "sampling_plan": sampling_plan,
        "summary": summary,
    }
