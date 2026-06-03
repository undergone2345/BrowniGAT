import math
from pathlib import Path

import pandas as pd

from utils.pretraining_manifest import load_canonical_bundle


class MultimodalPretrainingDataset:
    def __init__(self, workspace_dir):
        self.workspace_dir = Path(workspace_dir)
        self.manifest = self._load_manifest()
        self.bundle = load_canonical_bundle(self.manifest["bundle_dir"])
        self.samples_by_task = self._build_samples()

    def _load_manifest(self):
        manifest_path = self.workspace_dir / "pretraining_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing pretraining manifest: {manifest_path}")
        return pd.read_json(manifest_path, typ="series").to_dict()

    def _build_samples(self):
        samples = {}
        available_modalities = set(self.manifest["available_modalities"])

        if "masked_node_modeling" in {task["name"] for task in self.manifest["enabled_tasks"]}:
            samples["masked_node_modeling"] = [
                {"node_id": row["node_id"], "node_type": row["node_type"]}
                for _, row in self.bundle["nodes_df"].iterrows()
            ]

        if "masked_edge_modeling" in {task["name"] for task in self.manifest["enabled_tasks"]}:
            samples["masked_edge_modeling"] = [
                {"source": row["source"], "target": row["target"], "relation": row["relation"], "weight": row["weight"]}
                for _, row in self.bundle["edges_df"].iterrows()
            ]

        if "cross_modal_alignment" in {task["name"] for task in self.manifest["enabled_tasks"]}:
            cross_samples = []
            for modality_name in sorted(available_modalities):
                table = self.bundle["tables"][modality_name]
                for _, row in table.iterrows():
                    cross_samples.append({"modality": modality_name, "fields": row.to_dict()})
            samples["cross_modal_alignment"] = cross_samples

        if "perturbation_conditioning" in {task["name"] for task in self.manifest["enabled_tasks"]}:
            perturb_samples = []
            drug_target_table = self.bundle["tables"].get("drug_target", pd.DataFrame())
            disease_gene_table = self.bundle["tables"].get("disease_gene", pd.DataFrame())
            for _, row in drug_target_table.iterrows():
                disease_hits = disease_gene_table[disease_gene_table["gene"] == row["target"]]
                perturb_samples.append(
                    {
                        "drug": row["drug"],
                        "target": row["target"],
                        "action": row["action"],
                        "drug_score": row["score"],
                        "disease_support": float(disease_hits["score"].mean()) if not disease_hits.empty else 0.0,
                    }
                )
            samples["perturbation_conditioning"] = perturb_samples

        if "spatial_context_prediction" in {task["name"] for task in self.manifest["enabled_tasks"]}:
            spatial_table = self.bundle["tables"].get("spatial", pd.DataFrame())
            samples["spatial_context_prediction"] = [
                row.to_dict() for _, row in spatial_table.iterrows()
            ]

        return samples

    def get_task_samples(self, task_name):
        return self.samples_by_task.get(task_name, [])

    def as_torch_dataset(self, task_name):
        samples = self.get_task_samples(task_name)
        try:
            from torch.utils.data import Dataset
        except ModuleNotFoundError:
            return samples

        class _TaskDataset(Dataset):
            def __init__(self, task_samples):
                self.task_samples = task_samples

            def __len__(self):
                return len(self.task_samples)

            def __getitem__(self, idx):
                return self.task_samples[idx]

        return _TaskDataset(samples)


def create_batches(samples, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        samples[idx : idx + batch_size]
        for idx in range(0, len(samples), batch_size)
    ]


def build_epoch_batches(dataset, sampling_plan):
    epoch_batches = []
    for plan_item in sampling_plan:
        task_name = plan_item["task_name"]
        task_samples = dataset.get_task_samples(task_name)
        if not task_samples:
            continue
        base_batches = create_batches(task_samples, plan_item["batch_size"])
        for step_idx in range(plan_item["steps"]):
            epoch_batches.append(
                {
                    "task_name": task_name,
                    "step_index": step_idx,
                    "batch": base_batches[step_idx % len(base_batches)],
                }
            )
    return epoch_batches


def collate_foundation_batch(batch):
    if not batch:
        return {"raw_batch": [], "batch_size": 0}
    all_keys = sorted({key for item in batch for key in item.keys()})
    collated = {key: [item.get(key) for item in batch] for key in all_keys}
    collated["raw_batch"] = batch
    collated["batch_size"] = len(batch)
    return collated
