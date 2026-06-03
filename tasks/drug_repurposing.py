import pandas as pd

from utils.evidence_paths import collect_evidence_paths, summarize_path_strength


def run_drug_repurposing(dataset, embeddings, feature_frame, repurposing_cfg):
    graph = dataset["graph"]
    drug_df = dataset["drug_df"].copy()
    disease_df = dataset["disease_gene_df"].copy()

    disease_targets = disease_df.sort_values("importance", ascending=False)["gene"].tolist()
    perturbation_support = (
        dataset["perturbation_df"].groupby("target")["observed_reversal_score"].mean().to_dict()
    )

    records = []
    for drug_name, drug_group in drug_df.groupby("drug"):
        supported_targets = drug_group["target"].tolist()
        target_overlap_score = disease_df[disease_df["gene"].isin(supported_targets)]["importance"].sum()
        perturbation_alignment = sum(perturbation_support.get(target, 0.2) for target in supported_targets) / max(len(supported_targets), 1)
        reversal_support = drug_group["reversal_support"].mean()
        path_bonus, evidence_text = summarize_path_strength(
            collect_evidence_paths(graph, supported_targets, repurposing_cfg["disease_name"], max_hops=3, limit=4)
        )
        final_score = (
            target_overlap_score * float(repurposing_cfg["target_overlap_weight"])
            + perturbation_alignment * float(repurposing_cfg["perturbation_alignment_weight"])
            + reversal_support * float(repurposing_cfg["disease_reversal_weight"])
            + path_bonus * float(repurposing_cfg["path_bonus"])
        )
        records.append(
            {
                "drug": drug_name,
                "supported_targets": ", ".join(supported_targets),
                "target_overlap_score": target_overlap_score,
                "perturbation_alignment_score": perturbation_alignment,
                "reversal_support_score": reversal_support,
                "path_bonus_score": path_bonus,
                "repurposing_score": final_score,
                "evidence_paths": evidence_text,
            }
        )

    result_df = pd.DataFrame(records).sort_values(
        ["repurposing_score", "target_overlap_score"],
        ascending=False,
    ).reset_index(drop=True)
    result_df["drug_rank"] = result_df.index + 1
    return result_df
