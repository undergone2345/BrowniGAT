import networkx as nx
import pandas as pd
import torch


def compute_graph_metrics(edge_index, label_encoder, edge_weight=None):
    graph = nx.Graph()
    node_names = list(label_encoder.classes_)
    graph.add_nodes_from(node_names)

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    weights = (
        edge_weight.tolist()
        if edge_weight is not None
        else [1.0 for _ in range(len(src))]
    )

    for s, d, w in zip(src, dst, weights):
        source_name = label_encoder.inverse_transform([s])[0]
        target_name = label_encoder.inverse_transform([d])[0]
        if source_name == target_name:
            continue
        graph.add_edge(source_name, target_name, weight=float(w))

    degree_dict = dict(graph.degree(weight=None))
    weighted_degree_dict = dict(graph.degree(weight="weight"))
    pagerank_dict = nx.pagerank(graph, weight="weight")
    betweenness_dict = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    closeness_dict = nx.closeness_centrality(graph)

    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(graph, weight="weight")
    except nx.NetworkXException:
        eigenvector_dict = {node: 0.0 for node in graph.nodes}

    metrics_df = pd.DataFrame(
        {
            "Protein": node_names,
            "Degree": [degree_dict.get(node, 0.0) for node in node_names],
            "WeightedDegree": [
                weighted_degree_dict.get(node, 0.0) for node in node_names
            ],
            "PageRank": [pagerank_dict.get(node, 0.0) for node in node_names],
            "Betweenness": [
                betweenness_dict.get(node, 0.0) for node in node_names
            ],
            "Closeness": [closeness_dict.get(node, 0.0) for node in node_names],
            "Eigenvector": [
                eigenvector_dict.get(node, 0.0) for node in node_names
            ],
        }
    )
    return metrics_df.sort_values("Degree", ascending=False).reset_index(drop=True)
