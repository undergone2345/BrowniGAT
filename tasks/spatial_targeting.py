import pandas as pd


def run_spatial_targeting(dataset, embeddings, feature_frame, spatial_cfg):
    spatial_df = dataset["spatial_df"].copy()
    disease_regions = set(spatial_cfg["disease_regions"])
    disease_cell_types = set(spatial_cfg["disease_cell_types"])

    spatial_df["region_match"] = spatial_df["region"].isin(disease_regions).astype(float)
    spatial_df["cell_type_match"] = spatial_df["cell_type"].isin(disease_cell_types).astype(float)
    spatial_df["spatial_score"] = (
        (
            spatial_df["region_score"] * spatial_df["region_match"]
            + spatial_df["disease_hotspot_score"]
        )
        * float(spatial_cfg["region_weight"])
        + (
            spatial_df["cell_type_score"] * spatial_df["cell_type_match"]
        )
        * float(spatial_cfg["cell_type_weight"])
    )

    result_df = (
        spatial_df.groupby("target", as_index=False)
        .agg(
            top_region=("region", "first"),
            top_cell_type=("cell_type", "first"),
            region_score=("region_score", "max"),
            cell_type_score=("cell_type_score", "max"),
            hotspot_score=("disease_hotspot_score", "max"),
            spatial_targeting_score=("spatial_score", "max"),
        )
        .sort_values("spatial_targeting_score", ascending=False)
        .reset_index(drop=True)
    )
    result_df["spatial_rank"] = result_df.index + 1
    return result_df
