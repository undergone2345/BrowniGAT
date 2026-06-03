from pathlib import Path

from utils.reporting import save_run_metadata


def export_foundation_bundle(bundle, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle["nodes_df"].to_csv(output_dir / "canonical_nodes.tsv", sep="\t", index=False)
    bundle["edges_df"].to_csv(output_dir / "canonical_edges.tsv", sep="\t", index=False)

    modalities_dir = output_dir / "modalities"
    modalities_dir.mkdir(parents=True, exist_ok=True)
    for modality_name, table_df in bundle["tables"].items():
        table_df.to_csv(modalities_dir / f"{modality_name}.tsv", sep="\t", index=False)

    manifest = {
        "modality_files": {
            modality_name: f"modalities/{modality_name}.tsv"
            for modality_name in bundle["tables"].keys()
        },
        "canonical_nodes": "canonical_nodes.tsv",
        "canonical_edges": "canonical_edges.tsv",
        "summary": bundle["summary"],
    }
    save_run_metadata(manifest, output_dir / "foundation_manifest.json")
