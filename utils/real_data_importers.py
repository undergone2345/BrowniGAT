from pathlib import Path

import pandas as pd

from utils.schema_validation import validate_schema


def ingest_multimodal_sources(ingestion_cfg):
    normalization_cfg = ingestion_cfg.get("entity_normalization", {})
    source_cfgs = ingestion_cfg["sources"]

    canonical_tables = {}
    schema_reports = {}
    for modality_name, modality_cfg in source_cfgs.items():
        raw_df = pd.read_csv(modality_cfg["path"], sep=modality_cfg.get("sep", "\t"))
        canonical_df = _canonicalize_table(raw_df, modality_cfg["columns"])
        canonical_df = _normalize_entities(canonical_df, normalization_cfg)
        schema_report = validate_schema(canonical_df, modality_cfg["schema"])
        if not schema_report["is_valid"]:
            raise ValueError(
                f"Schema validation failed for modality '{modality_name}': {schema_report['errors']}"
            )
        canonical_tables[modality_name] = canonical_df
        schema_reports[modality_name] = schema_report

    bundle = harmonize_multimodal_bundle(canonical_tables)
    bundle["schema_reports"] = schema_reports
    return bundle


def harmonize_multimodal_bundle(canonical_tables):
    nodes = []
    edges = []

    if "drug_target" in canonical_tables:
        df = canonical_tables["drug_target"]
        nodes.extend([{"node_id": row["drug"], "node_type": "drug"} for _, row in df.iterrows()])
        nodes.extend([{"node_id": row["target"], "node_type": "gene"} for _, row in df.iterrows()])
        edges.extend(
            {
                "source": row["drug"],
                "target": row["target"],
                "relation": "drug_target",
                "weight": float(row["score"]),
                "action": row["action"],
                "source_db": row["source"],
            }
            for _, row in df.iterrows()
        )

    if "disease_gene" in canonical_tables:
        df = canonical_tables["disease_gene"]
        nodes.extend([{"node_id": row["disease"], "node_type": "disease"} for _, row in df.iterrows()])
        nodes.extend([{"node_id": row["gene"], "node_type": "gene"} for _, row in df.iterrows()])
        edges.extend(
            {
                "source": row["gene"],
                "target": row["disease"],
                "relation": "disease_association",
                "weight": float(row["score"]),
                "direction": row["direction"],
                "source_db": row["source"],
            }
            for _, row in df.iterrows()
        )

    if "pathway" in canonical_tables:
        df = canonical_tables["pathway"]
        nodes.extend([{"node_id": row["pathway"], "node_type": "pathway"} for _, row in df.iterrows()])
        nodes.extend([{"node_id": row["gene"], "node_type": "gene"} for _, row in df.iterrows()])
        edges.extend(
            {
                "source": row["gene"],
                "target": row["pathway"],
                "relation": "pathway_membership",
                "weight": float(row["score"]),
                "source_db": row["source"],
            }
            for _, row in df.iterrows()
        )

    if "spatial" in canonical_tables:
        df = canonical_tables["spatial"]
        nodes.extend([{"node_id": row["region"], "node_type": "region"} for _, row in df.iterrows()])
        nodes.extend([{"node_id": row["cell_type"], "node_type": "cell_type"} for _, row in df.iterrows()])
        nodes.extend([{"node_id": row["gene"], "node_type": "gene"} for _, row in df.iterrows()])
        edges.extend(
            {
                "source": row["gene"],
                "target": row["region"],
                "relation": "region_enrichment",
                "weight": float(row["region_score"]),
                "source_db": row["source"],
            }
            for _, row in df.iterrows()
        )
        edges.extend(
            {
                "source": row["gene"],
                "target": row["cell_type"],
                "relation": "celltype_expression",
                "weight": float(row["cell_type_score"]),
                "source_db": row["source"],
            }
            for _, row in df.iterrows()
        )

    nodes_df = pd.DataFrame(nodes).drop_duplicates().reset_index(drop=True)
    edges_df = pd.DataFrame(edges).drop_duplicates().reset_index(drop=True)

    return {
        "tables": canonical_tables,
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "summary": {
            "modalities": list(canonical_tables.keys()),
            "num_nodes": int(len(nodes_df)),
            "num_edges": int(len(edges_df)),
        },
    }


def _canonicalize_table(df, column_mapping):
    missing = [source_col for source_col in column_mapping.values() if source_col not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns required by mapping: {missing}")
    renamed_df = df.rename(columns={source_col: canonical_col for canonical_col, source_col in column_mapping.items()})
    ordered_columns = list(column_mapping.keys())
    return renamed_df[ordered_columns].copy()


def _normalize_entities(df, normalization_cfg):
    df = df.copy()
    object_columns = df.select_dtypes(include=["object"]).columns.tolist()
    if normalization_cfg.get("trim_whitespace", True):
        for column in object_columns:
            df[column] = df[column].astype(str).str.strip()

    if normalization_cfg.get("uppercase_genes", True):
        for gene_column in {"gene", "target"}.intersection(df.columns):
            df[gene_column] = df[gene_column].astype(str).str.upper()

    return df
