import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _min_max_scale(values):
    values = np.asarray(values, dtype=float)
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


def _resolve_target_ids(label_encoder, target_genes):
    node_names = set(label_encoder.classes_)
    present_targets = [gene for gene in target_genes if gene in node_names]
    if not present_targets:
        raise ValueError(
            "None of the configured target genes were found in the graph: "
            + ", ".join(target_genes)
        )
    return present_targets, label_encoder.transform(present_targets)


def compute_core_targets(embeddings, label_encoder, target_genes, graph_metrics_df, scoring_cfg):
    present_targets, target_ids = _resolve_target_ids(label_encoder, target_genes)
    similarity_matrix = cosine_similarity(embeddings, embeddings[target_ids])
    mean_similarity = similarity_matrix.mean(axis=1)
    max_similarity = similarity_matrix.max(axis=1)

    result_df = graph_metrics_df.copy()
    result_df["MeanTargetSimilarity"] = mean_similarity
    result_df["MaxTargetSimilarity"] = max_similarity
    result_df["KnownTargetHit"] = result_df["Protein"].isin(present_targets)
    for target_idx, target_name in enumerate(present_targets):
        result_df[f"SimilarityTo_{target_name}"] = similarity_matrix[:, target_idx]

    weights = scoring_cfg["weights"]
    score_components = {
        "similarity": _min_max_scale(result_df["MeanTargetSimilarity"]),
        "degree": _min_max_scale(result_df["Degree"]),
        "pagerank": _min_max_scale(result_df["PageRank"]),
        "betweenness": _min_max_scale(result_df["Betweenness"]),
        "closeness": _min_max_scale(result_df["Closeness"]),
    }

    composite_score = np.zeros(len(result_df), dtype=float)
    for key, component in score_components.items():
        composite_score += component * float(weights.get(key, 0.0))

    result_df["CompositeScore"] = composite_score
    result_df["CoreScore"] = (
        result_df["MeanTargetSimilarity"] * result_df["Degree"]
    )
    result_df["EvidenceTag"] = result_df.apply(_build_evidence_tag, axis=1)
    result_df["BaselineMethod"] = "embedding_similarity"

    ordered_columns = [
        "Protein",
        "KnownTargetHit",
        "EvidenceTag",
        "Degree",
        "WeightedDegree",
        "PageRank",
        "Betweenness",
        "Closeness",
        "Eigenvector",
        "MeanTargetSimilarity",
        "MaxTargetSimilarity",
        "CoreScore",
        "CompositeScore",
    ]
    similarity_columns = [f"SimilarityTo_{target_name}" for target_name in present_targets]
    return result_df[ordered_columns].sort_values(
        ["CompositeScore", "CoreScore"],
        ascending=False,
    ).reset_index(drop=True).merge(
        result_df[["Protein"] + similarity_columns],
        on="Protein",
        how="left",
    )


def summarize_targets(result_df, top_k=20):
    return result_df.head(top_k).copy()


def _build_evidence_tag(row):
    tags = []
    if row["KnownTargetHit"]:
        tags.append("known_target")
    if row["MeanTargetSimilarity"] >= 0.8:
        tags.append("high_similarity")
    if row["Degree"] >= 10:
        tags.append("hub_like")
    if row["PageRank"] >= 0.02:
        tags.append("high_pagerank")
    return "|".join(tags) if tags else "emerging_candidate"
