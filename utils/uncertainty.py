import numpy as np


def entropy_like_uncertainty(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 1.0
    clipped = np.clip(values, 1e-8, None)
    probabilities = clipped / clipped.sum()
    entropy = -(probabilities * np.log(probabilities)).sum()
    max_entropy = np.log(len(probabilities))
    if max_entropy == 0:
        return 0.0
    return float(entropy / max_entropy)


def score_dispersion_uncertainty(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0.0
    normalized_std = np.std(values) / max(np.mean(np.abs(values)), 1e-8)
    return float(np.clip(normalized_std, 0.0, 1.0))
