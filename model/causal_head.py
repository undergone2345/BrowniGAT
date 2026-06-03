import pandas as pd


class CausalTargetRanker:
    def __init__(self, causal_cfg):
        self.causal_cfg = causal_cfg

    def rank(self, candidate_df):
        cfg = self.causal_cfg
        candidate_df = candidate_df.copy()
        candidate_df["causal_score_raw"] = (
            candidate_df["stability_score"] * float(cfg["stability_weight"])
            + candidate_df["intervention_score"] * float(cfg["intervention_weight"])
            + candidate_df["spatial_score"] * float(cfg["spatial_weight"])
            + candidate_df["druggability_score"] * float(cfg["druggability_weight"])
            + candidate_df["evidence_score"] * float(cfg["evidence_weight"])
            - candidate_df["uncertainty_score"] * float(cfg["uncertainty_penalty"])
        )
        candidate_df = candidate_df.sort_values(
            ["causal_score_raw", "evidence_score", "intervention_score"],
            ascending=False,
        ).reset_index(drop=True)
        candidate_df["causal_rank"] = candidate_df.index + 1
        return candidate_df
