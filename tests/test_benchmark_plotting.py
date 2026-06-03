import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.benchmark_plotting import (
    load_aggregate_overview,
    load_benchmark_summary,
    plot_aggregate_overview,
    plot_benchmark_summary,
    prepare_benchmark_metric_frame,
)


class BenchmarkPlottingTests(unittest.TestCase):
    def test_prepare_benchmark_metric_frame_has_expected_metrics(self):
        summary_df = pd.DataFrame(
            [
                {"Method": "gat", "MRR": 0.7, "HitAtK": 1.0, "MeanRank": 1.5},
                {"Method": "gcn", "MRR": 0.5, "HitAtK": 0.8, "MeanRank": 2.1},
            ]
        )
        metric_df = prepare_benchmark_metric_frame(summary_df)

        self.assertEqual(set(metric_df["Metric"]), {"MRR", "HitAtK"})
        self.assertEqual(len(metric_df), 4)

    def test_plotters_write_png_files(self):
        summary_df = pd.DataFrame(
            [
                {"Method": "gat", "MRR": 0.7, "HitAtK": 1.0, "MeanRank": 1.5},
                {"Method": "gcn", "MRR": 0.5, "HitAtK": 0.8, "MeanRank": 2.1},
            ]
        )
        overview_df = pd.DataFrame(
            [
                {"Method": "gat", "TopProtein": "TYR", "TopScoreMean": 0.9, "TopKHitCount": 2, "RankMean": 1.2},
                {"Method": "gcn", "TopProtein": "MITF", "TopScoreMean": 0.8, "TopKHitCount": 1, "RankMean": 1.9},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "benchmark_summary.tsv"
            overview_path = tmpdir_path / "aggregate_overview.tsv"
            summary_png = tmpdir_path / "benchmark_summary.png"
            overview_png = tmpdir_path / "aggregate_overview.png"

            summary_df.to_csv(summary_path, sep="\t", index=False)
            overview_df.to_csv(overview_path, sep="\t", index=False)

            plot_benchmark_summary(summary_path, summary_png)
            plot_aggregate_overview(overview_path, overview_png)

            self.assertTrue(summary_png.exists())
            self.assertTrue(overview_png.exists())
            self.assertGreater(summary_png.stat().st_size, 0)
            self.assertGreater(overview_png.stat().st_size, 0)

            loaded_summary = load_benchmark_summary(summary_path)
            loaded_overview = load_aggregate_overview(overview_path)
            self.assertEqual(list(loaded_summary["Method"]), ["gat", "gcn"])
            self.assertEqual(list(loaded_overview["TopProtein"]), ["TYR", "MITF"])


if __name__ == "__main__":
    unittest.main()
