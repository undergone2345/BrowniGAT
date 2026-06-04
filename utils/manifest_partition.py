import math


def build_manifest_partitions(bundle, max_partition_size):
    max_partition_size = max(1, int(max_partition_size))
    partitions = {
        "nodes": _partition_table("nodes", int(len(bundle["nodes_df"])), max_partition_size),
        "edges": _partition_table("edges", int(len(bundle["edges_df"])), max_partition_size),
        "modalities": {},
    }
    for modality_name, table in bundle["tables"].items():
        partitions["modalities"][modality_name] = _partition_table(
            modality_name,
            int(len(table)),
            max_partition_size,
        )
    return partitions


def summarize_manifest_partitions(partitions):
    modality_partition_count = sum(len(items) for items in partitions.get("modalities", {}).values())
    return {
        "node_partitions": len(partitions.get("nodes", [])),
        "edge_partitions": len(partitions.get("edges", [])),
        "modality_partitions": int(modality_partition_count),
        "total_partitions": int(
            len(partitions.get("nodes", []))
            + len(partitions.get("edges", []))
            + modality_partition_count
        ),
    }


def _partition_table(table_name, row_count, max_partition_size):
    if row_count <= 0:
        return []
    num_partitions = int(math.ceil(row_count / max_partition_size))
    partitions = []
    for partition_idx in range(num_partitions):
        start_idx = partition_idx * max_partition_size
        end_idx = min(row_count, start_idx + max_partition_size)
        partitions.append(
            {
                "table_name": table_name,
                "partition_id": f"{table_name}_{partition_idx:03d}",
                "start_row": int(start_idx),
                "end_row": int(end_idx),
                "num_rows": int(end_idx - start_idx),
            }
        )
    return partitions
