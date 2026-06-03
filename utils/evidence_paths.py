import networkx as nx


def collect_evidence_paths(graph, source_nodes, target_node, max_hops=3, limit=5):
    undirected_graph = nx.Graph()
    for source, target, edge_data in graph.edges(data=True):
        undirected_graph.add_edge(
            source,
            target,
            relation=edge_data.get("relation", "related"),
            weight=edge_data.get("weight", 1.0),
        )

    paths = []
    for source_node in source_nodes:
        if source_node not in undirected_graph or target_node not in undirected_graph:
            continue
        try:
            for path in nx.all_simple_paths(undirected_graph, source=source_node, target=target_node, cutoff=max_hops):
                if len(paths) >= limit:
                    break
                edge_labels = []
                for idx in range(len(path) - 1):
                    edge_labels.append(
                        undirected_graph[path[idx]][path[idx + 1]].get("relation", "related")
                    )
                paths.append({"source": source_node, "target": target_node, "path": path, "relations": edge_labels})
        except nx.NetworkXNoPath:
            continue
        if len(paths) >= limit:
            break
    return paths


def summarize_path_strength(paths):
    if not paths:
        return 0.0, "no_support"
    shortest_path_len = min(len(item["path"]) for item in paths)
    path_diversity = len({tuple(item["relations"]) for item in paths})
    evidence_score = min(1.0, 0.35 + 0.15 * len(paths) + 0.08 * path_diversity - 0.05 * shortest_path_len)
    summary_text = "; ".join(
        " -> ".join(item["path"]) for item in paths[:3]
    )
    return evidence_score, summary_text
