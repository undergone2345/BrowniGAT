import json
from pathlib import Path

import pandas as pd

from utils.feature_specs import export_feature_specs
from utils.modality_registry import describe_modalities


def load_canonical_bundle(bundle_dir):
    bundle_dir = Path(bundle_dir)
    modalities_dir = bundle_dir / "modalities"
    tables = {}
    for modality_path in modalities_dir.glob("*.tsv"):
        tables[modality_path.stem] = pd.read_csv(modality_path, sep="\t")
    nodes_df = pd.read_csv(bundle_dir / "canonical_nodes.tsv", sep="\t")
    edges_df = pd.read_csv(bundle_dir / "canonical_edges.tsv", sep="\t")
    return {"tables": tables, "nodes_df": nodes_df, "edges_df": edges_df}


def build_pretraining_manifest(bundle_dir, enabled_modalities, enabled_tasks, tokenizer_cfg):
    bundle = load_canonical_bundle(bundle_dir)
    available_modalities = sorted([name for name in enabled_modalities if name in bundle["tables"]])

    vocabularies = _build_vocabularies(bundle["nodes_df"], tokenizer_cfg)
    manifest = {
        "bundle_dir": str(bundle_dir),
        "available_modalities": available_modalities,
        "modality_specs": describe_modalities(available_modalities),
        "feature_specs": export_feature_specs(),
        "tokenizer": tokenizer_cfg,
        "vocab_sizes": {name: len(tokens) for name, tokens in vocabularies.items()},
        "dataset_summary": {
            "num_nodes": int(len(bundle["nodes_df"])),
            "num_edges": int(len(bundle["edges_df"])),
        },
        "enabled_tasks": enabled_tasks,
    }
    return manifest


def save_pretraining_manifest(manifest, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def _build_vocabularies(nodes_df, tokenizer_cfg):
    special_tokens = tokenizer_cfg.get("reserve_special_tokens", [])
    vocabularies = {}
    for node_type in sorted(nodes_df["node_type"].unique().tolist()):
        node_ids = sorted(nodes_df[nodes_df["node_type"] == node_type]["node_id"].astype(str).unique().tolist())
        vocabularies[f"{node_type}_vocab"] = special_tokens + node_ids
    return vocabularies
