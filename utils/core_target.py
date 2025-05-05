import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd

def compute_core_targets(embeddings, degrees, label_encoder, target_genes):
    num_nodes = embeddings.shape[0]
    target_ids = label_encoder.transform([g for g in target_genes if g in label_encoder.classes_])
    similarity = cosine_similarity(embeddings, embeddings[target_ids])
    core_scores = similarity.mean(axis=1) * degrees.numpy()

    protein_names = label_encoder.inverse_transform(range(num_nodes))
    result_df = pd.DataFrame({
        'Protein': protein_names,
        'Degree': degrees.numpy(),
        'Similarity': similarity.mean(axis=1),
        'CoreScore': core_scores
    }).sort_values('CoreScore', ascending=False)

    return result_df
