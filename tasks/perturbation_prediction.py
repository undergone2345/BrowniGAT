import pandas as pd

from utils.hetero_graph import get_gene_nodes
from utils.uncertainty import entropy_like_uncertainty, score_dispersion_uncertainty


def run_perturbation_prediction(dataset, embeddings, feature_frame, perturbation_decoder, perturbation_cfg):
    perturbation_df = dataset["perturbation_df"].copy()
    gene_nodes = get_gene_nodes(dataset["nodes_df"])

    perturbation_support_map = perturbation_df.groupby("target")["observed_reversal_score"].mean().to_dict()
    preferred_action_map = {}
    for target, group in perturbation_df.groupby("target"):
        best_idx = group["observed_reversal_score"].idxmax()
        preferred_action_map[target] = group.loc[best_idx, "intervention"]

    feature_subset = feature_frame.set_index("node_id").loc[gene_nodes]
    candidate_frame = pd.DataFrame(
        {
            "target": gene_nodes,
            "disease_relevance": feature_subset["disease_relevance"].to_numpy(dtype=float),
            "perturbation_support": [perturbation_support_map.get(target, 0.25) for target in gene_nodes],
            "pathway_support": feature_subset["pathway_support"].to_numpy(dtype=float),
            "cell_specificity": feature_subset["cell_type_specificity"].to_numpy(dtype=float),
            "preferred_action": [preferred_action_map.get(target, "activate") for target in gene_nodes],
        }
    )
    result_df = perturbation_decoder.predict(candidate_frame)

    uncertainty_scores = []
    for target in result_df["target"]:
        target_rows = perturbation_df[perturbation_df["target"] == target]
        if target_rows.empty:
            uncertainty_scores.append(0.85)
            continue
        support_values = target_rows["observed_reversal_score"].to_numpy(dtype=float)
        uncertainty_scores.append(
            (entropy_like_uncertainty(support_values) + score_dispersion_uncertainty(support_values)) / 2.0
        )
    result_df["uncertainty"] = uncertainty_scores
    result_df = result_df.sort_values(
        ["predicted_reversal_score", "uncertainty"],
        ascending=[False, True],
    ).reset_index(drop=True)
    result_df["perturbation_rank"] = result_df.index + 1
    return result_df
