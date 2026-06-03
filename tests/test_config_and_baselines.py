import unittest

import pandas as pd

from utils.baselines import build_topology_embeddings, compute_centrality_baseline_targets
from utils.benchmark import summarize_benchmark_results
from utils.config import load_config


class ConfigAndBaselineTests(unittest.TestCase):
    def test_toy_config_loads_and_overrides_defaults(self):
        config = load_config("config/toy_config.yaml")

        self.assertEqual(config["data"]["path"], "data/toy_string_interactions.tsv")
        self.assertEqual(config["runtime"]["repeats"], 1)
        self.assertEqual(config["benchmark"]["top_k"], 5)
        self.assertIn("baseline_weights", config["scoring"])

    def test_centrality_baseline_scores_candidates(self):
        graph_metrics_df = pd.DataFrame(
            {
                "Protein": ["TYR", "MITF", "DCT"],
                "Degree": [10, 8, 3],
                "WeightedDegree": [9.8, 7.9, 2.5],
                "PageRank": [0.32, 0.25, 0.08],
                "Betweenness": [0.07, 0.04, 0.01],
                "Closeness": [0.8, 0.72, 0.4],
                "Eigenvector": [0.75, 0.61, 0.15],
            }
        )
        scoring_cfg = load_config("config/config.yaml")["scoring"]

        result_df = compute_centrality_baseline_targets(
            graph_metrics_df=graph_metrics_df,
            target_genes=["TYR", "MITF"],
            scoring_cfg=scoring_cfg,
        )

        self.assertEqual(result_df.iloc[0]["Protein"], "TYR")
        self.assertIn("CompositeScore", result_df.columns)
        self.assertIn("EvidenceTag", result_df.columns)

    def test_topology_embeddings_have_expected_shape(self):
        graph_metrics_df = pd.DataFrame(
            {
                "Protein": ["TYR", "MITF", "DCT"],
                "Degree": [10, 8, 3],
                "WeightedDegree": [9.8, 7.9, 2.5],
                "PageRank": [0.32, 0.25, 0.08],
                "Betweenness": [0.07, 0.04, 0.01],
                "Closeness": [0.8, 0.72, 0.4],
                "Eigenvector": [0.75, 0.61, 0.15],
            }
        )
        embeddings = build_topology_embeddings(graph_metrics_df)

        self.assertEqual(embeddings.shape, (3, 6))

    def test_benchmark_summary_aggregates_per_method(self):
        benchmark_frames = [
            pd.DataFrame(
                [
                    {"Method": "gat", "HeldOutTarget": "TYR", "Rank": 1, "ReciprocalRank": 1.0, "HitAt1": 1, "HitAt5": 1, "HitAt10": 1, "HitAtK": 1},
                    {"Method": "gat", "HeldOutTarget": "MITF", "Rank": 2, "ReciprocalRank": 0.5, "HitAt1": 0, "HitAt5": 1, "HitAt10": 1, "HitAtK": 1},
                ]
            ),
            pd.DataFrame(
                [
                    {"Method": "centrality", "HeldOutTarget": "TYR", "Rank": 3, "ReciprocalRank": 1 / 3, "HitAt1": 0, "HitAt5": 1, "HitAt10": 1, "HitAtK": 1},
                ]
            ),
        ]

        _, summary_df = summarize_benchmark_results(benchmark_frames)

        self.assertEqual(summary_df.iloc[0]["Method"], "gat")
        self.assertIn("MRR", summary_df.columns)
        self.assertIn("EvaluatedTargets", summary_df.columns)


if __name__ == "__main__":
    unittest.main()
