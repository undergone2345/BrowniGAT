import argparse
from pathlib import Path

from utils.config import load_config
from utils.foundation_workspace import build_foundation_workspace
from utils.reporting import save_run_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a foundation-model-ready workspace from a canonical BrowniGAT multimodal bundle."
    )
    parser.add_argument(
        "--config",
        default="config/foundation_example.yaml",
        help="Path to foundation workspace config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override for the workspace.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    foundation_cfg = config["foundation"]
    output_dir = Path(args.output_dir or foundation_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = build_foundation_workspace(foundation_cfg, output_dir)
    save_run_metadata(workspace["summary"], output_dir / "workspace_summary.json")
    print(f"Saved foundation workspace to: {output_dir}")


if __name__ == "__main__":
    main()
