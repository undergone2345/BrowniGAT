import numpy as np
import pandas as pd

from utils.evidence_paths import collect_evidence_paths, summarize_path_strength
from utils.hetero_graph import get_gene_nodes


def _cosine_similarity_matrix(matrix, anchor_matrix):
    matrix = np.asarray(matrix, dtype=float)
    anchor_matrix = np.asarray(anchor_matrix, dtype=float)
    numerator = matrix @ anchor_matrix.T
    denominator = (
        np.linalg.norm(matrix, axis=1, keepdims=True)
        * np.linalg.norm(anchor_matrix, axis=1)[None, :]
    )
    return numerator / np.clip(denominator, 1e-8, None)


def run_target_prioritization(dataset, embeddings, feature_frame, targeting_cfg):
    graph = dataset["graph"]
    disease_df = dataset["disease_gene_df"]
    gene_nodes = get_gene_nodes(dataset["nodes_df"])

    embedding_df = embeddings.set_index("node_id")
    gene_embedding_df = embedding_df.loc[gene_nodes]
    target_genes = [gene for gene in targeting_cfg["target_genes"] if gene in gene_embedding_df.index]
    target_anchor_matrix = gene_embedding_df.loc[target_genes].to_numpy(dtype=float)
    gene_matrix = gene_embedding_df.to_numpy(dtype=float)

    similarities = _cosine_similarity_matrix(gene_matrix, target_anchor_matrix)
    mean_similarity = similarities.mean(axis=1)

    feature_subset = feature_frame.set_index("node_id").loc[gene_nodes]
    disease_relevance = feature_subset["disease_relevance"].to_numpy(dtype=float)
    expression_specificity = feature_subset["cell_type_specificity"].to_numpy(dtype=float)

    disease_support_map = disease_df.set_index("gene")["importance"].to_dict()
    disease_support = np.asarray([disease_support_map.get(gene, 0.0) for gene in gene_nodes], dtype=float)

    evidence_scores = []
    evidence_text = []
    for gene in gene_nodes:
        paths = collect_evidence_paths(graph, target_genes, gene, max_hops=3, limit=4)
        evidence_score, evidence_summary = summarize_path_strength(paths)
        evidence_scores.append(evidence_score)
        evidence_text.append(evidence_summary)

    network_support = np.asarray(evidence_scores, dtype=float)
    composite = (
        mean_similarity * float(targeting_cfg["weight_similarity"])
        + disease_support * float(targeting_cfg["weight_disease_relevance"])
        + expression_specificity * float(targeting_cfg["weight_expression_specificity"])
        + network_support * float(targeting_cfg["weight_network_support"])
    )

    result_df = pd.DataFrame(
        {
            "target": gene_nodes,
            "mean_similarity": mean_similarity,
            "disease_support": disease_support,
            "expression_specificity": expression_specificity,
            "network_support": network_support,
            "target_priority_score": composite,
            "evidence_paths": evidence_text,
        }
    ).sort_values(
        ["target_priority_score", "network_support", "disease_support"],
        ascending=False,
    ).reset_index(drop=True)
    result_df["target_rank"] = result_df.index + 1
    return result_df
