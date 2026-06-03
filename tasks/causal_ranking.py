import pandas as pd

from utils.uncertainty import score_dispersion_uncertainty


def run_causal_ranking(
    dataset,
    embeddings,
    feature_frame,
    target_df,
    perturbation_df,
    spatial_df,
    repurposing_df,
    causal_ranker,
    causal_cfg,
):
    feature_subset = feature_frame.set_index("node_id")
    target_subset = target_df.set_index("target")
    perturbation_subset = perturbation_df.set_index("target")
    spatial_subset = spatial_df.set_index("target")

    drug_map = {}
    for _, row in dataset["drug_df"].iterrows():
        drug_map.setdefault(row["target"], []).append(row["drug"])

    causal_rows = []
    for target in target_subset.index.tolist():
        stability_score = float(target_subset.loc[target, "mean_similarity"])
        intervention_score = float(perturbation_subset.loc[target, "predicted_reversal_score"]) if target in perturbation_subset.index else 0.2
        spatial_score = float(spatial_subset.loc[target, "spatial_targeting_score"]) if target in spatial_subset.index else 0.1
        druggability_score = float(feature_subset.loc[target, "druggability"])
        evidence_score = float(target_subset.loc[target, "network_support"])
        uncertainty_score = float(perturbation_subset.loc[target, "uncertainty"]) if target in perturbation_subset.index else 0.85

        supporting_drugs = sorted(set(drug_map.get(target, [])))
        causal_rows.append(
            {
                "target": target,
                "stability_score": stability_score,
                "intervention_score": intervention_score,
                "spatial_score": spatial_score,
                "druggability_score": druggability_score,
                "evidence_score": evidence_score,
                "uncertainty_score": uncertainty_score,
                "supporting_drugs": ", ".join(supporting_drugs) if supporting_drugs else "none",
                "causal_hypothesis": _build_causal_hypothesis(
                    target=target,
                    supporting_drugs=supporting_drugs,
                    spatial_score=spatial_score,
                    intervention_score=intervention_score,
                ),
            }
        )

    causal_df = causal_ranker.rank(pd.DataFrame(causal_rows))
    causal_df["score_dispersion"] = score_dispersion_uncertainty(causal_df["causal_score_raw"].to_numpy())
    return causal_df


def _build_causal_hypothesis(target, supporting_drugs, spatial_score, intervention_score):
    location_text = "context-dependent" if spatial_score < 0.6 else "spatially reinforced"
    drug_text = ", ".join(supporting_drugs[:2]) if supporting_drugs else "no linked compounds"
    effect_text = "high predicted reversal" if intervention_score >= 0.75 else "moderate predicted reversal"
    return f"{target} is {location_text}, shows {effect_text}, and links to {drug_text}."
