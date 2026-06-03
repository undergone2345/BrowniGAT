import numpy as np
import pandas as pd

from utils.hetero_graph import get_neighbors_by_relation


class HeteroGraphEncoder:
    def __init__(self, encoder_cfg):
        self.encoder_cfg = encoder_cfg

    def fit_transform(self, dataset):
        nodes_df = dataset["nodes_df"].copy()
        graph = dataset["graph"]

        base_feature_columns = [
            "basal_activity",
            "expression_score",
            "disease_relevance",
            "cell_type_specificity",
            "druggability",
            "pathway_support",
        ]
        feature_frame = nodes_df[["node_id", "node_type"] + base_feature_columns].copy()
        feature_matrix = feature_frame[base_feature_columns].to_numpy(dtype=float)

        relation_names = sorted(dataset["edges_df"]["relation"].unique().tolist())
        relation_weights = self.encoder_cfg["relation_weights"]

        propagated_blocks = [feature_matrix]
        for _ in range(int(self.encoder_cfg["propagation_steps"])):
            relation_block = []
            for relation_name in relation_names:
                block = np.zeros_like(feature_matrix, dtype=float)
                for node_idx, node_id in enumerate(feature_frame["node_id"].tolist()):
                    neighbors = get_neighbors_by_relation(graph, node_id, relation_name)
                    if not neighbors:
                        continue
                    neighbor_indices = [
                        feature_frame.index[feature_frame["node_id"] == neighbor_id][0]
                        for neighbor_id, _ in neighbors
                    ]
                    neighbor_weights = np.asarray([weight for _, weight in neighbors], dtype=float)
                    neighbor_vectors = feature_matrix[neighbor_indices]
                    aggregated = (neighbor_vectors * neighbor_weights[:, None]).sum(axis=0)
                    aggregated /= max(neighbor_weights.sum(), 1e-8)
                    block[node_idx] = aggregated * float(relation_weights.get(relation_name, 1.0))
                relation_block.append(block)
            if relation_block:
                propagated_blocks.append(np.mean(relation_block, axis=0))

        embedding_matrix = np.concatenate(propagated_blocks, axis=1)
        if self.encoder_cfg.get("normalize_embeddings", True):
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            embedding_matrix = embedding_matrix / np.clip(norms, 1e-8, None)

        embedding_columns = [f"emb_{idx}" for idx in range(embedding_matrix.shape[1])]
        embeddings = pd.DataFrame(embedding_matrix, columns=embedding_columns)
        embeddings.insert(0, "node_id", feature_frame["node_id"].tolist())
        return embeddings, feature_frame
