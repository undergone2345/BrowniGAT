from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_benchmark_summary(summary_path):
    summary_df = pd.read_csv(summary_path, sep="\t")
    required_columns = {"Method", "MRR", "HitAtK", "MeanRank"}
    missing = required_columns.difference(summary_df.columns)
    if missing:
        raise ValueError(f"Missing required benchmark summary columns: {sorted(missing)}")
    return summary_df


def load_aggregate_overview(overview_path):
    overview_df = pd.read_csv(overview_path, sep="\t")
    required_columns = {"Method", "TopProtein", "TopScoreMean", "TopKHitCount", "RankMean"}
    missing = required_columns.difference(overview_df.columns)
    if missing:
        raise ValueError(f"Missing required aggregate overview columns: {sorted(missing)}")
    return overview_df


def prepare_benchmark_metric_frame(summary_df):
    long_df = summary_df.melt(
        id_vars=["Method"],
        value_vars=["MRR", "HitAtK"],
        var_name="Metric",
        value_name="Score",
    )
    return long_df


def plot_benchmark_summary(summary_path, output_path):
    summary_df = load_benchmark_summary(summary_path)
    metric_df = prepare_benchmark_metric_frame(summary_df)

    methods = summary_df["Method"].tolist()
    metric_names = ["MRR", "HitAtK"]
    colors = {"MRR": "#bf360c", "HitAtK": "#ef6c00"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    positions = list(range(len(methods)))
    width = 0.35
    for metric_index, metric_name in enumerate(metric_names):
        metric_values = (
            metric_df[metric_df["Metric"] == metric_name]
            .set_index("Method")
            .loc[methods, "Score"]
            .tolist()
        )
        shifted_positions = [pos + (metric_index - 0.5) * width for pos in positions]
        axes[0].bar(
            shifted_positions,
            metric_values,
            width=width,
            label=metric_name,
            color=colors[metric_name],
            alpha=0.85,
        )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(methods, rotation=20)
    axes[0].set_ylim(0, max(1.0, summary_df[["MRR", "HitAtK"]].max().max() * 1.15))
    axes[0].set_title("Recovery Metrics")
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False)

    mean_rank_df = summary_df.sort_values("MeanRank", ascending=True)
    axes[1].barh(
        mean_rank_df["Method"],
        mean_rank_df["MeanRank"],
        color="#6d4c41",
        alpha=0.9,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Mean Rank")
    axes[1].set_xlabel("Lower is better")

    fig.suptitle("BrowniGAT Benchmark Summary", fontsize=14)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_overview(overview_path, output_path):
    overview_df = load_aggregate_overview(overview_path).sort_values(
        ["TopKHitCount", "TopScoreMean"],
        ascending=[False, False],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        overview_df["Method"],
        overview_df["TopScoreMean"],
        color="#2e7d32",
        alpha=0.9,
    )
    axes[0].set_title("Top Candidate Score Mean")
    axes[0].set_ylabel("Composite Score Mean")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].scatter(
        overview_df["TopKHitCount"],
        overview_df["RankMean"],
        s=140,
        c="#1565c0",
        alpha=0.85,
    )
    for _, row in overview_df.iterrows():
        axes[1].text(
            row["TopKHitCount"] + 0.02,
            row["RankMean"] + 0.02,
            row["Method"],
            fontsize=8,
        )
    axes[1].set_title("Stability vs Rank")
    axes[1].set_xlabel("TopK Hit Count")
    axes[1].set_ylabel("Rank Mean")

    fig.suptitle("BrowniGAT Aggregate Method Overview", fontsize=14)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
