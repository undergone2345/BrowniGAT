import argparse
from pathlib import Path

from utils.config import load_config
from utils.foundation_trainer import run_foundation_training
from utils.reporting import save_run_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BrowniGAT foundation pretraining trainer skeleton."
    )
    parser.add_argument(
        "--config",
        default="config/foundation_example.yaml",
        help="Path to foundation training config YAML.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Optional workspace directory override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    foundation_cfg = config["foundation"]
    workspace_dir = Path(args.workspace_dir or foundation_cfg["output_dir"])
    workspace_dir.mkdir(parents=True, exist_ok=True)

    result = run_foundation_training(foundation_cfg, workspace_dir)
    save_run_metadata(result["summary"], workspace_dir / "training_summary.json")
    print(f"Saved training artifacts to: {workspace_dir}")


if __name__ == "__main__":
    main()
