import argparse
from pathlib import Path

from utils.benchmark_plotting import (
    plot_aggregate_overview,
    plot_benchmark_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benchmark figures from BrowniGAT method comparison tables."
    )
    parser.add_argument(
        "--benchmark-summary",
        default="results/method_comparison/benchmark_summary.tsv",
        help="Path to benchmark_summary.tsv.",
    )
    parser.add_argument(
        "--aggregate-overview",
        default="results/method_comparison/aggregate_overview.tsv",
        help="Path to aggregate_overview.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/method_comparison/plots",
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_plot_path = output_dir / "benchmark_summary.png"
    aggregate_plot_path = output_dir / "aggregate_overview.png"

    plot_benchmark_summary(args.benchmark_summary, benchmark_plot_path)
    plot_aggregate_overview(args.aggregate_overview, aggregate_plot_path)

    print(f"Saved benchmark summary plot to: {benchmark_plot_path}")
    print(f"Saved aggregate overview plot to: {aggregate_plot_path}")


if __name__ == "__main__":
    main()
