import json
from pathlib import Path

import matplotlib.pyplot as plt


def save_table(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)


def save_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_markdown_report(
    summary,
    target_df,
    perturbation_df,
    spatial_df,
    repurposing_df,
    causal_df,
    output_path,
):
    lines = [
        "# BrowniGAT vNext Report",
        "",
        f"- Nodes: `{summary['num_nodes']}`",
        f"- Edges: `{summary['num_edges']}`",
        f"- Perturbation records: `{summary['num_perturbations']}`",
        f"- Spatial regions: `{summary['num_spatial_regions']}`",
        f"- Top target: `{summary['top_target']}`",
        f"- Top drug: `{summary['top_drug']}`",
        f"- Top causal target: `{summary['top_causal_target']}`",
        "",
        "## Target Prioritization",
        "",
        _frame_to_markdown_table(target_df.head(8)),
        "",
        "## Perturbation Forecast",
        "",
        _frame_to_markdown_table(perturbation_df.head(8)),
        "",
        "## Spatial Targeting",
        "",
        _frame_to_markdown_table(spatial_df.head(8)),
        "",
        "## Drug Repurposing",
        "",
        _frame_to_markdown_table(repurposing_df.head(8)),
        "",
        "## Causal Ranking",
        "",
        _frame_to_markdown_table(causal_df.head(8)),
        "",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def save_vnext_summary_plots(target_df, perturbation_df, spatial_df, repurposing_df, causal_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_bar(
        names=target_df.head(8)["target"].tolist(),
        values=target_df.head(8)["target_priority_score"].tolist(),
        title="Target Prioritization",
        ylabel="Priority Score",
        output_path=output_dir / "target_prioritization.png",
        color="#8e24aa",
    )
    _plot_bar(
        names=perturbation_df.head(8)["target"].tolist(),
        values=perturbation_df.head(8)["predicted_reversal_score"].tolist(),
        title="Perturbation Forecast",
        ylabel="Predicted Reversal",
        output_path=output_dir / "perturbation_forecast.png",
        color="#ef6c00",
    )
    _plot_bar(
        names=spatial_df.head(8)["target"].tolist(),
        values=spatial_df.head(8)["spatial_targeting_score"].tolist(),
        title="Spatial Targeting",
        ylabel="Spatial Score",
        output_path=output_dir / "spatial_targeting.png",
        color="#039be5",
    )
    _plot_bar(
        names=repurposing_df.head(8)["drug"].tolist(),
        values=repurposing_df.head(8)["repurposing_score"].tolist(),
        title="Drug Repurposing",
        ylabel="Repurposing Score",
        output_path=output_dir / "drug_repurposing.png",
        color="#43a047",
    )
    _plot_bar(
        names=causal_df.head(8)["target"].tolist(),
        values=causal_df.head(8)["causal_score_raw"].tolist(),
        title="Causal Ranking",
        ylabel="Causal Score",
        output_path=output_dir / "causal_ranking.png",
        color="#6d4c41",
    )


def _plot_bar(names, values, title, ylabel, output_path, color):
    plt.figure(figsize=(10, 4.8))
    plt.bar(names, values, color=color, alpha=0.88)
    plt.xticks(rotation=25)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _frame_to_markdown_table(df):
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        formatted = []
        for value in row.tolist():
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator] + rows)
