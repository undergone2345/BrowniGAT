import numpy as np
import pandas as pd


class PerturbationEffectDecoder:
    def __init__(self, perturbation_cfg):
        self.perturbation_cfg = perturbation_cfg

    def predict(self, candidate_frame):
        weights = self.perturbation_cfg["feature_weights"]
        score = (
            candidate_frame["disease_relevance"] * float(weights.get("disease_relevance", 0.0))
            + candidate_frame["perturbation_support"] * float(weights.get("perturbation_support", 0.0))
            + candidate_frame["pathway_support"] * float(weights.get("pathway_support", 0.0))
            + candidate_frame["cell_specificity"] * float(weights.get("cell_specificity", 0.0))
        )
        uncertainty = 1.0 - np.clip(
            candidate_frame["perturbation_support"] * self.perturbation_cfg["uncertainty_temperature"],
            0.0,
            1.0,
        )
        predicted_effect = np.where(
            candidate_frame["preferred_action"] == "activate",
            score * 1.1,
            score,
        )
        return pd.DataFrame(
            {
                "target": candidate_frame["target"],
                "preferred_action": candidate_frame["preferred_action"],
                "predicted_reversal_score": predicted_effect,
                "uncertainty": uncertainty,
            }
        )
