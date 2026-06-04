import pandas as pd


def run_gene_program_design(dataset, target_df, perturbation_df, spatial_df, synbio_cfg):
    top_k = int(synbio_cfg.get("top_k_targets", 6))
    max_targets_per_construct = int(synbio_cfg.get("max_targets_per_construct", 3))
    editor_map = synbio_cfg.get(
        "editor_by_action",
        {"activate": "CRISPRa", "repress": "CRISPRi"},
    )
    promoter_map = synbio_cfg.get("promoter_by_cell_type", {})

    merged = (
        target_df.head(top_k)
        .merge(
            perturbation_df[["target", "preferred_action", "predicted_reversal_score", "uncertainty"]],
            on="target",
            how="left",
        )
        .merge(
            spatial_df[["target", "top_region", "top_cell_type", "spatial_targeting_score"]],
            on="target",
            how="left",
        )
        .copy()
    )
    merged["preferred_action"] = merged["preferred_action"].fillna("repress")
    merged["editor_system"] = merged["preferred_action"].map(editor_map).fillna("CRISPRi")
    merged["promoter"] = merged["top_cell_type"].map(promoter_map).fillna(
        synbio_cfg.get("default_promoter", "EF1A")
    )
    merged["delivery_system"] = synbio_cfg.get("delivery_system", "lentiviral_pool")
    merged["design_score"] = (
        merged["target_priority_score"] * 0.45
        + merged["predicted_reversal_score"].fillna(0.0) * 0.35
        + merged["spatial_targeting_score"].fillna(0.0) * 0.2
        - merged["uncertainty"].fillna(0.0) * float(synbio_cfg.get("uncertainty_penalty", 0.1))
    )
    merged = merged.sort_values("design_score", ascending=False).reset_index(drop=True)
    merged["multiplex_module"] = [
        f"module_{idx // max_targets_per_construct + 1:02d}"
        for idx in range(len(merged))
    ]
    merged["design_rationale"] = merged.apply(
        lambda row: (
            f"{row['editor_system']}->{row['target']} in {row['top_cell_type']} / {row['top_region']}"
        ),
        axis=1,
    )
    return merged[
        [
            "target",
            "preferred_action",
            "editor_system",
            "promoter",
            "delivery_system",
            "top_cell_type",
            "top_region",
            "multiplex_module",
            "design_score",
            "design_rationale",
        ]
    ]


def run_pathway_rewiring_plan(dataset, gene_program_df, synbio_cfg):
    edges_df = dataset["edges_df"]
    pathway_edges = edges_df[edges_df["relation"] == "pathway_membership"][["source", "target"]].copy()
    pathway_edges.columns = ["target", "pathway"]
    merged = gene_program_df.merge(pathway_edges, on="target", how="left")
    merged["pathway"] = merged["pathway"].fillna("unassigned_program")

    rewiring = (
        merged.groupby("pathway", as_index=False)
        .agg(
            targets=("target", lambda values: ",".join(sorted(set(values)))),
            dominant_editor=("editor_system", lambda values: values.mode().iloc[0]),
            module_count=("multiplex_module", "nunique"),
            average_design_score=("design_score", "mean"),
        )
        .sort_values("average_design_score", ascending=False)
        .reset_index(drop=True)
    )
    rewiring["rewiring_strategy"] = rewiring.apply(
        lambda row: (
            f"{row['dominant_editor']} program across {row['module_count']} modules"
        ),
        axis=1,
    )
    rewiring["build_priority"] = rewiring.index + 1
    return rewiring


def run_construct_blueprints(gene_program_df, pathway_rewiring_df, synbio_cfg):
    grouped = (
        gene_program_df.groupby("multiplex_module", as_index=False)
        .agg(
            targets=("target", lambda values: ",".join(values)),
            editors=("editor_system", lambda values: ",".join(sorted(set(values)))),
            promoters=("promoter", lambda values: ",".join(sorted(set(values)))),
            preferred_context=("top_cell_type", lambda values: values.mode().iloc[0] if not values.mode().empty else "generic"),
            mean_design_score=("design_score", "mean"),
        )
        .sort_values("mean_design_score", ascending=False)
        .reset_index(drop=True)
    )
    dominant_pathway = pathway_rewiring_df.iloc[0]["pathway"] if not pathway_rewiring_df.empty else "unassigned_program"
    grouped["construct_name"] = [
        f"{synbio_cfg.get('construct_prefix', 'BrowniBuild')}_{idx + 1:02d}"
        for idx in range(len(grouped))
    ]
    grouped["payload_strategy"] = grouped.apply(
        lambda row: f"{row['editors']} on {row['targets']}",
        axis=1,
    )
    grouped["pathway_focus"] = dominant_pathway
    grouped["assembly_strategy"] = synbio_cfg.get("assembly_strategy", "GoldenGate")
    return grouped[
        [
            "construct_name",
            "multiplex_module",
            "targets",
            "editors",
            "promoters",
            "preferred_context",
            "pathway_focus",
            "assembly_strategy",
            "payload_strategy",
            "mean_design_score",
        ]
    ]
