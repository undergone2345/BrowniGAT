from pathlib import Path

import networkx as nx
import pandas as pd


def load_multimodal_hetero_dataset(dataset_cfg):
    nodes_df = pd.read_csv(dataset_cfg["nodes_path"], sep="\t")
    edges_df = pd.read_csv(dataset_cfg["edges_path"], sep="\t")
    perturbation_df = pd.read_csv(dataset_cfg["perturbation_path"], sep="\t")
    spatial_df = pd.read_csv(dataset_cfg["spatial_path"], sep="\t")
    drug_df = pd.read_csv(dataset_cfg["drug_path"], sep="\t")
    disease_gene_df = pd.read_csv(dataset_cfg["disease_gene_path"], sep="\t")

    graph = nx.MultiDiGraph()
    for _, row in nodes_df.iterrows():
        graph.add_node(row["node_id"], **row.to_dict())

    for _, row in edges_df.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            key=row["relation"],
            relation=row["relation"],
            weight=float(row["weight"]),
        )
        graph.add_edge(
            row["target"],
            row["source"],
            key=f"{row['relation']}_reverse",
            relation=row["relation"],
            weight=float(row["weight"]),
        )

    return {
        "graph": graph,
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "perturbation_df": perturbation_df,
        "spatial_df": spatial_df,
        "drug_df": drug_df,
        "disease_gene_df": disease_gene_df,
    }


def get_gene_nodes(nodes_df):
    return nodes_df[nodes_df["node_type"] == "gene"]["node_id"].tolist()


def get_neighbors_by_relation(graph, node_id, relation_name):
    neighbors = []
    for _, target, edge_data in graph.out_edges(node_id, data=True):
        if edge_data.get("relation") == relation_name:
            neighbors.append((target, float(edge_data.get("weight", 1.0))))
    return neighbors
