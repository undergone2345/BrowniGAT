from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "data": {
        "path": "data/string_interactions_short.csv",
        "sep": "\t",
        "source_column": "node1",
        "target_column": "node2",
        "weight_column": "combined_score",
        "min_weight": 0.0,
        "make_undirected": True,
        "add_self_loops": True,
        "feature_mode": "identity",
    },
    "model": {
        "hidden_channels": 128,
        "out_channels": 64,
        "heads": 4,
        "dropout": 0.15,
    },
    "training": {
        "epochs": 120,
        "lr": 0.005,
        "weight_decay": 0.0001,
        "negative_ratio": 1,
        "log_every": 10,
    },
    "scoring": {
        "target_genes": ["TYR", "MITF", "UCP1", "PPARGC1A"],
        "top_k_similarity_targets": 4,
        "weights": {
            "similarity": 0.45,
            "degree": 0.2,
            "pagerank": 0.15,
            "betweenness": 0.1,
            "closeness": 0.1,
        },
        "baseline_weights": {
            "degree": 0.25,
            "weighted_degree": 0.2,
            "pagerank": 0.2,
            "betweenness": 0.15,
            "closeness": 0.1,
            "eigenvector": 0.1,
        },
    },
    "benchmark": {
        "methods": ["gat", "gcn", "graphsage", "centrality"],
        "top_k": 10,
    },
    "visualization": {
        "method": "tsne",
        "perplexity": 5,
    },
    "reporting": {
        "top_k": 20,
    },
    "runtime": {
        "seed": 42,
        "repeats": 3,
        "seed_stride": 17,
        "device": "cpu",
        "output_dir": "results",
    },
}


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULT_CONFIG, loaded)
