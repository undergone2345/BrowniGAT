import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected


def _build_node_features(num_nodes, feature_mode, edge_index):
    if feature_mode == "identity":
        return torch.eye(num_nodes, dtype=torch.float32)

    if feature_mode == "degree_profile":
        degree = torch.bincount(edge_index.flatten(), minlength=num_nodes).float()
        max_degree = degree.max().clamp(min=1.0)
        normalized_degree = (degree / max_degree).unsqueeze(1)
        bias = torch.ones((num_nodes, 1), dtype=torch.float32)
        return torch.cat([bias, normalized_degree], dim=1)

    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def load_ppi_data(data_cfg):
    df = pd.read_csv(
        data_cfg["path"],
        sep=data_cfg["sep"],
        comment="#",
    )

    required_columns = [data_cfg["source_column"], data_cfg["target_column"]]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    weight_column = data_cfg.get("weight_column")
    if weight_column and weight_column in df.columns:
        filtered_df = df[df[weight_column] >= data_cfg["min_weight"]].copy()
        edge_weight = torch.tensor(
            filtered_df[weight_column].to_numpy(),
            dtype=torch.float32,
        )
    else:
        filtered_df = df.copy()
        edge_weight = torch.ones(len(filtered_df), dtype=torch.float32)

    protein_a = filtered_df[data_cfg["source_column"]].astype(str)
    protein_b = filtered_df[data_cfg["target_column"]].astype(str)
    all_proteins = pd.concat([protein_a, protein_b]).unique()

    label_encoder = LabelEncoder().fit(all_proteins)
    edge_index = torch.tensor(
        [
            label_encoder.transform(protein_a),
            label_encoder.transform(protein_b),
        ],
        dtype=torch.long,
    )

    if data_cfg["make_undirected"]:
        edge_index, edge_weight = to_undirected(
            edge_index,
            edge_attr=edge_weight,
            num_nodes=len(label_encoder.classes_),
        )

    if data_cfg["add_self_loops"]:
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=len(label_encoder.classes_),
        )

    num_nodes = len(label_encoder.classes_)
    x = _build_node_features(num_nodes, data_cfg["feature_mode"], edge_index)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return data, label_encoder, filtered_df
