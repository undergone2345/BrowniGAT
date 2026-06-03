import tempfile
import unittest
from pathlib import Path

from model.causal_head import CausalTargetRanker
from model.hetero_encoder import HeteroGraphEncoder
from model.perturbation_decoder import PerturbationEffectDecoder
from tasks.causal_ranking import run_causal_ranking
from tasks.drug_repurposing import run_drug_repurposing
from tasks.perturbation_prediction import run_perturbation_prediction
from tasks.spatial_targeting import run_spatial_targeting
from tasks.target_prioritization import run_target_prioritization
from utils.hetero_graph import load_multimodal_hetero_dataset
from utils.vnext_config import load_vnext_config
from utils.vnext_reporting import save_json, save_markdown_report, save_table, save_vnext_summary_plots


class VNextPipelineTests(unittest.TestCase):
    def test_vnext_toy_pipeline_generates_all_major_outputs(self):
        config = load_vnext_config("config/vnext_toy.yaml")
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

        self.assertFalse(target_df.empty)
        self.assertFalse(perturbation_df.empty)
        self.assertFalse(spatial_df.empty)
        self.assertFalse(repurposing_df.empty)
        self.assertFalse(causal_df.empty)
        self.assertIn("target_priority_score", target_df.columns)
        self.assertIn("predicted_reversal_score", perturbation_df.columns)
        self.assertIn("spatial_targeting_score", spatial_df.columns)
        self.assertIn("repurposing_score", repurposing_df.columns)
        self.assertIn("causal_score_raw", causal_df.columns)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            summary = {
                "num_nodes": len(dataset["nodes_df"]),
                "num_edges": len(dataset["edges_df"]),
                "num_perturbations": len(dataset["perturbation_df"]),
                "num_spatial_regions": dataset["spatial_df"]["region"].nunique(),
                "top_target": target_df.iloc[0]["target"],
                "top_drug": repurposing_df.iloc[0]["drug"],
                "top_causal_target": causal_df.iloc[0]["target"],
            }
            save_table(target_df, outdir / "target_prioritization.tsv")
            save_json(summary, outdir / "summary.json")
            save_markdown_report(
                summary,
                target_df,
                perturbation_df,
                spatial_df,
                repurposing_df,
                causal_df,
                outdir / "REPORT.md",
            )
            save_vnext_summary_plots(
                target_df,
                perturbation_df,
                spatial_df,
                repurposing_df,
                causal_df,
                outdir / "plots",
            )

            self.assertTrue((outdir / "target_prioritization.tsv").exists())
            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "REPORT.md").exists())
            self.assertTrue((outdir / "plots" / "causal_ranking.png").exists())


if __name__ == "__main__":
    unittest.main()
