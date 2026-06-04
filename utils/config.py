from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "foundation": {
        "source_bundle_dir": "results_real_ingestion",
        "output_dir": "results_foundation_workspace",
        "experiment": {
            "name": "browning_foundation_pretraining_demo",
            "mode": "research",
        },
        "backbone": {
            "name": "hetero_graph_transformer",
            "hidden_dim": 256,
            "num_layers": 6,
        },
        "tokenizer": {
            "gene_symbol_case": "upper",
            "split_characters": False,
            "reserve_special_tokens": ["[PAD]", "[MASK]", "[CLS]"],
        },
        "modalities": {"include": []},
        "tasks": {
            "enabled": [],
            "batch_size_per_task": {},
        },
        "sampling": {
            "steps_per_epoch": 50,
            "task_temperature": 1.0,
            "shuffle_tasks_each_epoch": False,
            "repeat_strategy": "round_robin",
        },
        "dataloader": {
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "prefetch_factor": None,
        },
        "distributed": {
            "world_size": 1,
            "global_rank": 0,
            "local_rank": 0,
            "backend": "single_process",
            "gradient_sync": False,
        },
        "data_partitioning": {
            "max_partition_size": 1000,
        },
        "curriculum": {
            "phases": [],
        },
        "checkpoint_retention": {
            "keep_last_per_phase": 1,
            "keep_best": True,
        },
        "experiment_scheduler": {
            "enabled": False,
            "execute_queue": True,
            "default_stage_epochs": 1,
            "enforce_stage_order": True,
            "resources": {
                "cpu_slots": 4,
                "gpu_slots": 1,
                "max_concurrent_runs": 1,
            },
            "failure_recovery": {
                "max_retries": 1,
                "resume_strategy": "latest",
            },
            "promotion_policy": {
                "enabled": False,
                "metric_name": "best_validation_loss",
                "mode": "min",
                "threshold": 0.2,
                "on_fail": "halt",
            },
        },
        "validation": {
            "max_batches_per_task": 2,
        },
        "early_stopping": {
            "patience": 0,
            "min_delta": 0.0,
            "mode": "min",
        },
        "training": {
            "epochs": 3,
            "checkpoint_every": 1,
            "gradient_accumulation_steps": 1,
            "resume_from_checkpoint": None,
            "grad_clip_norm": None,
            "use_amp": False,
            "log_every_n_steps": 50,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 3e-4,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "name": "cosine_with_warmup",
            "warmup_steps": 20,
            "min_lr_ratio": 0.1,
        },
    },
    "real_data": {
        "output_dir": "results_real_ingestion",
        "entity_normalization": {
            "uppercase_genes": True,
            "trim_whitespace": True,
        },
        "sources": {},
    },
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
