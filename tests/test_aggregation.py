import unittest

import pandas as pd

from utils.aggregation import aggregate_rankings, summarize_aggregate_methods


def _build_result_frame(proteins, composite_scores, core_scores):
    return pd.DataFrame(
        {
            "Protein": proteins,
            "KnownTargetHit": [protein in {"TYR", "MITF"} for protein in proteins],
            "Degree": [10, 8, 4][: len(proteins)],
            "WeightedDegree": [9.5, 7.5, 3.5][: len(proteins)],
            "PageRank": [0.3, 0.2, 0.1][: len(proteins)],
            "Betweenness": [0.05, 0.03, 0.01][: len(proteins)],
            "Closeness": [0.8, 0.6, 0.5][: len(proteins)],
            "Eigenvector": [0.7, 0.5, 0.2][: len(proteins)],
            "MeanTargetSimilarity": [0.9, 0.7, 0.4][: len(proteins)],
            "MaxTargetSimilarity": [0.95, 0.75, 0.42][: len(proteins)],
            "CompositeScore": composite_scores,
            "CoreScore": core_scores,
        }
    )


class AggregationTests(unittest.TestCase):
    def test_aggregate_rankings_computes_stability_columns(self):
        frame_a = _build_result_frame(
            ["TYR", "MITF", "DCT"],
            [0.91, 0.82, 0.31],
            [9.1, 6.5, 1.2],
        )
        frame_b = _build_result_frame(
            ["MITF", "TYR", "DCT"],
            [0.88, 0.87, 0.29],
            [7.0, 8.9, 1.0],
        )

        aggregate_df = aggregate_rankings([frame_a, frame_b], top_k=2)

        self.assertIn("CompositeScoreMean", aggregate_df.columns)
        self.assertIn("RankMean", aggregate_df.columns)
        self.assertEqual(aggregate_df.iloc[0]["Protein"], "TYR")
        self.assertGreaterEqual(aggregate_df.iloc[0]["TopKHitCount"], 1)

    def test_summarize_aggregate_methods_returns_method_level_overview(self):
        aggregate_a = pd.DataFrame(
            [{"Protein": "TYR", "CompositeScoreMean": 0.91, "TopKHitCount": 2, "RankMean": 1.5}]
        )
        aggregate_b = pd.DataFrame(
            [{"Protein": "MITF", "CompositeScoreMean": 0.81, "TopKHitCount": 1, "RankMean": 2.0}]
        )

        overview_df = summarize_aggregate_methods({"gat": aggregate_a, "centrality": aggregate_b})

        self.assertEqual(list(overview_df["Method"]), ["gat", "centrality"])
        self.assertEqual(overview_df.iloc[0]["TopProtein"], "TYR")


if __name__ == "__main__":
    unittest.main()
