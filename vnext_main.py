import argparse
from pathlib import Path

from utils.vnext_config import load_vnext_config
from utils.vnext_reporting import (
    save_json,
    save_markdown_report,
    save_table,
    save_vnext_summary_plots,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BrowniGAT vNext multi-capability heterogenous graph pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/vnext_toy.yaml",
        help="Path to the vNext YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def run_vnext_pipeline(config_path, output_dir_override=None):
    from model.causal_head import CausalTargetRanker
    from model.hetero_encoder import HeteroGraphEncoder
    from model.perturbation_decoder import PerturbationEffectDecoder
    from tasks.causal_ranking import run_causal_ranking
    from tasks.drug_repurposing import run_drug_repurposing
    from tasks.perturbation_prediction import run_perturbation_prediction
    from tasks.spatial_targeting import run_spatial_targeting
    from tasks.target_prioritization import run_target_prioritization
    from utils.hetero_graph import load_multimodal_hetero_dataset
    from utils.seed import set_global_seed

    config = load_vnext_config(config_path)
    output_dir = Path(output_dir_override or config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config["runtime"]["seed"])

    dataset = load_multimodal_hetero_dataset(config["dataset"])
    encoder = HeteroGraphEncoder(config["encoder"])
    embeddings, feature_frame = encoder.fit_transform(dataset)

    perturbation_decoder = PerturbationEffectDecoder(config["perturbation"])
    causal_ranker = CausalTargetRanker(config["causal"])

    target_df = run_target_prioritization(dataset, embeddings, feature_frame, config["targeting"])
    perturbation_df = run_perturbation_prediction(
        dataset,
        embeddings,
        feature_frame,
        perturbation_decoder,
        config["perturbation"],
    )
    spatial_df = run_spatial_targeting(dataset, embeddings, feature_frame, config["spatial"])
    repurposing_df = run_drug_repurposing(dataset, embeddings, feature_frame, config["repurposing"])
    causal_df = run_causal_ranking(
        dataset,
        embeddings,
        feature_frame,
        target_df,
        perturbation_df,
        spatial_df,
        repurposing_df,
        causal_ranker,
        config["causal"],
    )

    summary = {
        "num_nodes": int(len(dataset["nodes_df"])),
        "num_edges": int(len(dataset["edges_df"])),
        "num_perturbations": int(len(dataset["perturbation_df"])),
        "num_spatial_regions": int(dataset["spatial_df"]["region"].nunique()),
        "top_target": target_df.iloc[0]["target"],
        "top_drug": repurposing_df.iloc[0]["drug"],
        "top_causal_target": causal_df.iloc[0]["target"],
    }

    save_table(target_df, output_dir / "target_prioritization.tsv")
    save_table(perturbation_df, output_dir / "perturbation_forecast.tsv")
    save_table(spatial_df, output_dir / "spatial_targeting.tsv")
    save_table(repurposing_df, output_dir / "drug_repurposing.tsv")
    save_table(causal_df, output_dir / "causal_ranking.tsv")
    save_json(summary, output_dir / "vnext_summary.json")
    save_markdown_report(
        summary=summary,
        target_df=target_df,
        perturbation_df=perturbation_df,
        spatial_df=spatial_df,
        repurposing_df=repurposing_df,
        causal_df=causal_df,
        output_path=output_dir / "REPORT.md",
    )
    save_vnext_summary_plots(
        target_df=target_df,
        perturbation_df=perturbation_df,
        spatial_df=spatial_df,
        repurposing_df=repurposing_df,
        causal_df=causal_df,
        output_dir=output_dir / "plots",
    )

    print(f"Saved vNext outputs to: {output_dir}")
    print("Top causal targets:")
    print(causal_df.head(config["reporting"]["top_k"]).to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_vnext_pipeline(args.config, output_dir_override=args.output_dir)
