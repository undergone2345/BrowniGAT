import numpy as np
import pandas as pd


def _min_max_scale(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


def compute_centrality_baseline_targets(graph_metrics_df, target_genes, scoring_cfg):
    weights = scoring_cfg["baseline_weights"]
    result_df = graph_metrics_df.copy()
    result_df["KnownTargetHit"] = result_df["Protein"].isin(target_genes)
    result_df["MeanTargetSimilarity"] = 0.0
    result_df["MaxTargetSimilarity"] = 0.0

    baseline_components = {
        "degree": _min_max_scale(result_df["Degree"]),
        "weighted_degree": _min_max_scale(result_df["WeightedDegree"]),
        "pagerank": _min_max_scale(result_df["PageRank"]),
        "betweenness": _min_max_scale(result_df["Betweenness"]),
        "closeness": _min_max_scale(result_df["Closeness"]),
        "eigenvector": _min_max_scale(result_df["Eigenvector"]),
    }

    baseline_score = np.zeros(len(result_df), dtype=float)
    for name, component in baseline_components.items():
        baseline_score += component * float(weights.get(name, 0.0))

    result_df["CoreScore"] = result_df["Degree"] * 1.0
    result_df["CompositeScore"] = baseline_score
    result_df["EvidenceTag"] = result_df.apply(_build_baseline_evidence_tag, axis=1)
    result_df["BaselineMethod"] = "centrality"
    return result_df.sort_values(
        ["CompositeScore", "Degree", "PageRank"],
        ascending=False,
    ).reset_index(drop=True)


def build_topology_embeddings(graph_metrics_df):
    feature_columns = [
        "Degree",
        "WeightedDegree",
        "PageRank",
        "Betweenness",
        "Closeness",
        "Eigenvector",
    ]
    features = graph_metrics_df[feature_columns].to_numpy(dtype=float)
    scaled_columns = [_min_max_scale(features[:, col_idx]) for col_idx in range(features.shape[1])]
    return np.column_stack(scaled_columns)


def _build_baseline_evidence_tag(row):
    tags = []
    if row["KnownTargetHit"]:
        tags.append("known_target")
    if row["Degree"] >= 10:
        tags.append("hub_like")
    if row["PageRank"] >= 0.02:
        tags.append("high_pagerank")
    if row["Betweenness"] >= 0.01:
        tags.append("bridge_like")
    return "|".join(tags) if tags else "topology_candidate"
