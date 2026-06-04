import argparse
from pathlib import Path

from model.causal_head import CausalTargetRanker
from model.hetero_encoder import HeteroGraphEncoder
from model.perturbation_decoder import PerturbationEffectDecoder
from tasks.causal_ranking import run_causal_ranking
from tasks.drug_repurposing import run_drug_repurposing
from tasks.perturbation_prediction import run_perturbation_prediction
from tasks.spatial_targeting import run_spatial_targeting
from tasks.synthetic_biology import (
    run_construct_blueprints,
    run_gene_program_design,
    run_pathway_rewiring_plan,
)
from tasks.target_prioritization import run_target_prioritization
from utils.hetero_graph import load_multimodal_hetero_dataset
from utils.seed import set_global_seed
from utils.synbio_reporting import save_synbio_markdown_report
from utils.vnext_config import load_vnext_config
from utils.vnext_reporting import save_json, save_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BrowniGAT synthetic biology design layer."
    )
    parser.add_argument(
        "--config",
        default="config/synbio_toy.yaml",
        help="Path to the synthetic biology YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def run_synbio_pipeline(config_path, output_dir_override=None):
    config = load_vnext_config(config_path)
    output_dir = Path(output_dir_override or config["runtime"].get("output_dir_synbio", "results_synbio"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config["runtime"]["seed"])

    dataset = load_multimodal_hetero_dataset(config["dataset"])
    encoder = HeteroGraphEncoder(config["encoder"])
    embeddings, feature_frame = encoder.fit_transform(dataset)

    target_df = run_target_prioritization(dataset, embeddings, feature_frame, config["targeting"])
    perturbation_df = run_perturbation_prediction(
        dataset,
        embeddings,
        feature_frame,
        PerturbationEffectDecoder(config["perturbation"]),
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
        CausalTargetRanker(config["causal"]),
        config["causal"],
    )

    gene_program_df = run_gene_program_design(
        dataset,
        target_df,
        perturbation_df,
        spatial_df,
        config["synbio"],
    )
    pathway_rewiring_df = run_pathway_rewiring_plan(dataset, gene_program_df, config["synbio"])
    construct_df = run_construct_blueprints(gene_program_df, pathway_rewiring_df, config["synbio"])

    summary = {
        "top_design_target": gene_program_df.iloc[0]["target"],
        "top_pathway_program": pathway_rewiring_df.iloc[0]["pathway"],
        "top_construct": construct_df.iloc[0]["construct_name"],
        "num_gene_programs": int(len(gene_program_df)),
        "num_constructs": int(len(construct_df)),
        "upstream_top_causal_target": causal_df.iloc[0]["target"],
    }

    save_table(gene_program_df, output_dir / "gene_program_design.tsv")
    save_table(pathway_rewiring_df, output_dir / "pathway_rewiring.tsv")
    save_table(construct_df, output_dir / "construct_blueprints.tsv")
    save_json(summary, output_dir / "synbio_summary.json")
    save_synbio_markdown_report(
        summary,
        gene_program_df,
        pathway_rewiring_df,
        construct_df,
        output_dir / "SYNBIO_REPORT.md",
    )
    print(f"Saved synthetic biology outputs to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_synbio_pipeline(args.config, output_dir_override=args.output_dir)
