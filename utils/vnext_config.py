from pathlib import Path

import yaml


VNEXT_DEFAULT_CONFIG = {
    "dataset": {},
    "encoder": {
        "relation_weights": {},
        "propagation_steps": 2,
        "normalize_embeddings": True,
    },
    "targeting": {
        "target_genes": [],
        "weight_similarity": 0.35,
        "weight_disease_relevance": 0.25,
        "weight_expression_specificity": 0.2,
        "weight_network_support": 0.2,
    },
    "perturbation": {
        "target_state": "reverse_disease",
        "desired_direction": "down",
        "feature_weights": {},
        "uncertainty_temperature": 0.65,
    },
    "spatial": {
        "disease_regions": [],
        "disease_cell_types": [],
        "region_weight": 0.6,
        "cell_type_weight": 0.4,
    },
    "repurposing": {
        "disease_name": "anti_browning",
        "path_bonus": 0.15,
        "target_overlap_weight": 0.45,
        "perturbation_alignment_weight": 0.3,
        "disease_reversal_weight": 0.25,
    },
    "causal": {
        "stability_weight": 0.25,
        "intervention_weight": 0.3,
        "spatial_weight": 0.15,
        "druggability_weight": 0.15,
        "evidence_weight": 0.15,
        "uncertainty_penalty": 0.25,
    },
    "synbio": {
        "top_k_targets": 6,
        "max_targets_per_construct": 3,
        "editor_by_action": {
            "activate": "CRISPRa",
            "repress": "CRISPRi",
        },
        "promoter_by_cell_type": {},
        "default_promoter": "EF1A",
        "delivery_system": "lentiviral_pool",
        "assembly_strategy": "GoldenGate",
        "construct_prefix": "BrowniBuild",
        "uncertainty_penalty": 0.1,
    },
    "reporting": {"top_k": 8},
    "runtime": {"seed": 11, "output_dir": "results_vnext", "output_dir_synbio": "results_synbio"},
}


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_vnext_config(config_path):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(VNEXT_DEFAULT_CONFIG, loaded)
