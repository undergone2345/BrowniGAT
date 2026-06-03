import argparse
from pathlib import Path

from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.real_data_importers import ingest_multimodal_sources
from utils.reporting import save_run_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest real multimodal biological tables into BrowniGAT canonical heterogenous graph format."
    )
    parser.add_argument(
        "--config",
        default="config/real_data_example.yaml",
        help="Path to ingestion config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for the canonical bundle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ingestion_cfg = config["real_data"]
    output_dir = Path(args.output_dir or ingestion_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = ingest_multimodal_sources(ingestion_cfg)
    export_foundation_bundle(bundle, output_dir)
    save_run_metadata(
        {
            "bundle_summary": bundle["summary"],
            "schema_reports": bundle["schema_reports"],
            "modalities": list(bundle["tables"].keys()),
        },
        output_dir / "ingestion_summary.json",
    )
    print(f"Saved canonical bundle to: {output_dir}")


if __name__ == "__main__":
    main()
