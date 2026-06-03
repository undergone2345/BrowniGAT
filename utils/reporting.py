import json
from pathlib import Path

import pandas as pd


def save_run_history(history, output_path):
    pd.DataFrame(history).to_csv(output_path, index=False)


def build_run_metadata(
    config,
    data,
    graph_df,
    trainer_state,
    device,
    seed,
    network_audit,
    method_name,
):
    return {
        "method_name": method_name,
        "device": str(device),
        "seed": int(seed),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "input_rows": int(len(graph_df)),
        "best_loss": trainer_state["best_loss"],
        "epochs": trainer_state["epochs"],
        "targets": config["scoring"]["target_genes"],
        "feature_mode": config["data"]["feature_mode"],
        "visualization_method": config["visualization"]["method"],
        "network_audit": network_audit,
    }


def save_run_metadata(metadata, output_path):
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def save_summary_markdown(summary_df, metadata, output_path):
    lines = [
        "# BrowniGAT Run Summary",
        "",
        f"- Method: `{metadata.get('method_name', 'n/a')}`",
        f"- Device: `{metadata['device']}`",
        f"- Seed: `{metadata.get('seed', 'n/a')}`",
        f"- Nodes: `{metadata['num_nodes']}`",
        f"- Edges: `{metadata['num_edges']}`",
        f"- Best loss: `{metadata['best_loss']:.6f}`",
        f"- Target genes: `{', '.join(metadata['targets'])}`",
        f"- Network density: `{metadata['network_audit']['density']:.6f}`",
        f"- Connected components: `{metadata['network_audit']['num_connected_components']}`",
        f"- Largest component: `{metadata['network_audit']['largest_component_size']}`",
        "",
        "## Top Candidates",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## Network Audit",
        "",
        f"- Observed targets: `{metadata['network_audit']['observed_target_count']}`",
        f"- Missing targets: `{', '.join(metadata['network_audit']['missing_targets']) or 'None'}`",
        f"- Mean clustering: `{metadata['network_audit']['mean_clustering']:.6f}`",
        "",
    ]
    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def save_method_comparison_markdown(summary_df, output_path):
    lines = [
        "# BrowniGAT Method Comparison",
        "",
        "This table summarizes leave-one-target-out recovery across methods.",
        "",
        summary_df.to_markdown(index=False),
        "",
    ]
    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
