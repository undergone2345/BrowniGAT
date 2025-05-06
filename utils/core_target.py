import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd

def compute_core_targets(embeddings, degrees, label_encoder, target_genes):
    """
    计算与褐变相关的核心靶点

    参数:
    embeddings (ndarray): GAT模型生成的节点嵌入
    degrees (tensor): 每个节点的度数
    label_encoder (LabelEncoder): 标签编码器，用于蛋白名称与ID之间的转换
    target_genes (list): 褐变相关的目标基因列表

    返回:
    pd.DataFrame: 包含核心靶点的DataFrame
    """
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
