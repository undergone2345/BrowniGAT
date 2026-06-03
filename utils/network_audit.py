import networkx as nx


def build_network_from_data(data, label_encoder):
    graph = nx.Graph()
    graph.add_nodes_from(label_encoder.classes_)

    src = data.edge_index[0].detach().cpu().tolist()
    dst = data.edge_index[1].detach().cpu().tolist()
    weights = (
        data.edge_weight.detach().cpu().tolist()
        if hasattr(data, "edge_weight")
        else [1.0 for _ in src]
    )
    for s, d, w in zip(src, dst, weights):
        source_name = label_encoder.inverse_transform([s])[0]
        target_name = label_encoder.inverse_transform([d])[0]
        graph.add_edge(source_name, target_name, weight=float(w))
    return graph


def audit_network(graph, target_genes):
    components = list(nx.connected_components(graph))
    component_sizes = sorted((len(component) for component in components), reverse=True)
    observed_targets = [gene for gene in target_genes if gene in graph.nodes]

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "num_connected_components": nx.number_connected_components(graph),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "mean_clustering": nx.average_clustering(graph),
        "observed_target_count": len(observed_targets),
        "missing_targets": [gene for gene in target_genes if gene not in graph.nodes],
    }
