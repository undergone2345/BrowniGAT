import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BrowniGAT and compare graph-learning baselines."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device from config. Example: cpu, cuda.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    return parser.parse_args()


def resolve_device(device_name):
    import torch

    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_method_outputs(
    method_name,
    method_dir,
    result_df,
    summary_df,
    history,
    metadata,
    embeddings,
    graph_metrics_df,
    label_encoder,
    visualization_cfg,
    reporting_cfg,
    seed,
):
    from utils.baselines import build_topology_embeddings
    from utils.reporting import (
        save_run_history,
        save_run_metadata,
        save_summary_markdown,
    )
    from utils.visualize import plot_embedding_projection

    targets_path = method_dir / "core_targets.tsv"
    history_path = method_dir / "training_history.csv"
    metadata_path = method_dir / "run_metadata.json"
    summary_path = method_dir / "summary.md"
    plot_path = method_dir / "embedding_projection.png"

    result_df.to_csv(targets_path, sep="\t", index=False)
    save_run_history(history, history_path)
    save_run_metadata(metadata, metadata_path)
    save_summary_markdown(summary_df, metadata, summary_path)

    projection_embeddings = embeddings
    if projection_embeddings is None:
        projection_embeddings = build_topology_embeddings(graph_metrics_df)

    plot_embedding_projection(
        embeddings=projection_embeddings,
        node_scores=result_df["CompositeScore"].to_numpy(),
        degrees=result_df["Degree"].to_numpy(),
        label_encoder=label_encoder,
        highlight_genes=summary_df["Protein"].tolist(),
        method=visualization_cfg["method"],
        perplexity=visualization_cfg["perplexity"],
        random_state=seed,
        save_path=plot_path,
        title=f"BrowniGAT Projection ({method_name})",
    )

    print(f"Saved {method_name} targets to: {targets_path}")


def run_single_method(
    method_name,
    config,
    data,
    label_encoder,
    graph_df,
    graph_metrics_df,
    network_audit,
    method_dir,
    device,
    seed,
):
    import torch

    from utils.baselines import compute_centrality_baseline_targets
    from utils.benchmark import evaluate_method_recovery
    from utils.core_target import compute_core_targets, summarize_targets
    from utils.model_factory import build_model
    from utils.reporting import build_run_metadata
    from utils.trainer import train_model

    method_name = method_name.lower()
    method_dir.mkdir(parents=True, exist_ok=True)

    if method_name == "centrality":
        result_df = compute_centrality_baseline_targets(
            graph_metrics_df=graph_metrics_df,
            target_genes=config["scoring"]["target_genes"],
            scoring_cfg=config["scoring"],
        )
        trainer_state = {
            "history": [{"epoch": 0, "loss": 0.0, "positive_loss": 0.0, "negative_loss": 0.0}],
            "best_loss": 0.0,
            "epochs": 0,
        }
        embeddings = None
    else:
        model = build_model(
            model_name=method_name,
            in_channels=data.num_node_features,
            model_cfg=config["model"],
        ).to(device)
        trainer_state = train_model(
            model=model,
            data=data,
            training_cfg=config["training"],
            device=device,
            run_name=method_name,
        )
        model.eval()
        with torch.no_grad():
            embeddings = model(data).detach().cpu().numpy()

        result_df = compute_core_targets(
            embeddings=embeddings,
            label_encoder=label_encoder,
            target_genes=config["scoring"]["target_genes"],
            graph_metrics_df=graph_metrics_df,
            scoring_cfg=config["scoring"],
        )

    summary_df = summarize_targets(result_df, top_k=config["reporting"]["top_k"])
    metadata = build_run_metadata(
        config=config,
        data=data,
        graph_df=graph_df,
        trainer_state=trainer_state,
        device=device,
        seed=seed,
        network_audit=network_audit,
        method_name=method_name,
    )
    benchmark_df = evaluate_method_recovery(
        method_name=method_name,
        embeddings=embeddings,
        label_encoder=label_encoder,
        graph_metrics_df=graph_metrics_df,
        scoring_cfg=config["scoring"],
        benchmark_cfg=config["benchmark"],
    )
    benchmark_df.to_csv(method_dir / "benchmark_recovery.tsv", sep="\t", index=False)

    _save_method_outputs(
        method_name=method_name,
        method_dir=method_dir,
        result_df=result_df,
        summary_df=summary_df,
        history=trainer_state["history"],
        metadata=metadata,
        embeddings=embeddings,
        graph_metrics_df=graph_metrics_df,
        label_encoder=label_encoder,
        visualization_cfg=config["visualization"],
        reporting_cfg=config["reporting"],
        seed=seed,
    )

    return {
        "method_name": method_name,
        "result_df": result_df,
        "summary_df": summary_df,
        "metadata": metadata,
        "trainer_state": trainer_state,
        "benchmark_df": benchmark_df,
    }


def run_single_experiment(config, output_dir, device, seed):
    from utils.data_loader import load_ppi_data
    from utils.graph_metrics import compute_graph_metrics
    from utils.network_audit import audit_network, build_network_from_data
    from utils.seed import set_global_seed

    set_global_seed(seed)

    data, label_encoder, graph_df = load_ppi_data(config["data"])
    data = data.to(device)

    graph_metrics_df = compute_graph_metrics(
        edge_index=data.edge_index.detach().cpu(),
        label_encoder=label_encoder,
        edge_weight=data.edge_weight.detach().cpu()
        if hasattr(data, "edge_weight")
        else None,
    )
    graph = build_network_from_data(data, label_encoder)
    network_audit = audit_network(graph, config["scoring"]["target_genes"])

    method_outputs = {}
    for method_name in config["benchmark"]["methods"]:
        method_dir = output_dir / method_name.lower()
        print(f"  -> Running method: {method_name}")
        method_outputs[method_name.lower()] = run_single_method(
            method_name=method_name,
            config=config,
            data=data,
            label_encoder=label_encoder,
            graph_df=graph_df,
            graph_metrics_df=graph_metrics_df,
            network_audit=network_audit,
            method_dir=method_dir,
            device=device,
            seed=seed,
        )

    return method_outputs


def run_pipeline(config_path, device_override=None, output_dir_override=None):
    from utils.aggregation import aggregate_rankings, summarize_aggregate_methods
    from utils.benchmark import summarize_benchmark_results
    from utils.config import load_config
    from utils.reporting import save_method_comparison_markdown

    config = load_config(config_path)
    output_dir = Path(output_dir_override or config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_override or config["runtime"]["device"])

    base_seed = config["runtime"]["seed"]
    repeats = int(config["runtime"]["repeats"])
    seed_stride = int(config["runtime"]["seed_stride"])
    methods = [method.lower() for method in config["benchmark"]["methods"]]

    all_runs = []
    for run_idx in range(repeats):
        run_seed = base_seed + run_idx * seed_stride
        run_dir = output_dir / f"run_{run_idx + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Starting run {run_idx + 1}/{repeats} with seed {run_seed} ===")
        all_runs.append(
            run_single_experiment(
                config=config,
                output_dir=run_dir,
                device=device,
                seed=run_seed,
            )
        )

    method_aggregate_rankings = {}
    benchmark_frames = []
    for method_name in methods:
        method_frames = [run_output[method_name]["result_df"] for run_output in all_runs]
        aggregate_df = aggregate_rankings(
            method_frames,
            top_k=config["reporting"]["top_k"],
        )
        method_aggregate_rankings[method_name] = aggregate_df
        method_dir = output_dir / "method_comparison" / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        aggregate_df.to_csv(method_dir / "aggregate_rankings.tsv", sep="\t", index=False)
        benchmark_frames.extend([run_output[method_name]["benchmark_df"] for run_output in all_runs])

    benchmark_df, benchmark_summary_df = summarize_benchmark_results(benchmark_frames)
    benchmark_dir = output_dir / "method_comparison"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(benchmark_dir / "benchmark_details.tsv", sep="\t", index=False)
    benchmark_summary_df.to_csv(benchmark_dir / "benchmark_summary.tsv", sep="\t", index=False)
    save_method_comparison_markdown(
        benchmark_summary_df,
        benchmark_dir / "benchmark_summary.md",
    )

    aggregate_overview_df = summarize_aggregate_methods(method_aggregate_rankings)
    aggregate_overview_df.to_csv(
        benchmark_dir / "aggregate_overview.tsv",
        sep="\t",
        index=False,
    )

    print(f"\nSaved method comparison outputs to: {benchmark_dir}")
    print("\nMethod benchmark summary:")
    print(benchmark_summary_df.to_string(index=False))
    print("\nAggregate overview:")
    print(aggregate_overview_df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.config,
        device_override=args.device,
        output_dir_override=args.output_dir,
    )
