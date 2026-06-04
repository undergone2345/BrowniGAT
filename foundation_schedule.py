import argparse
from pathlib import Path

from utils.config import load_config
from utils.experiment_scheduler import schedule_foundation_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the BrowniGAT foundation experiment scheduler."
    )
    parser.add_argument(
        "--config",
        default="config/foundation_engine_example.yaml",
        help="Path to foundation scheduler config YAML.",
    )
    parser.add_argument(
        "--workspace-dir",
        default=None,
        help="Optional scheduler workspace directory override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    foundation_cfg = config["foundation"]
    workspace_dir = Path(args.workspace_dir or foundation_cfg["output_dir"])
    workspace_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = schedule_foundation_experiments(foundation_cfg, workspace_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Missing canonical bundle input for scheduler. "
            "Run `python ingest_multimodal_data.py --config config/real_data_example.yaml` first, "
            "or point `foundation.source_bundle_dir` to an existing canonical bundle."
        ) from exc
    print(
        "Scheduler status={status} scheduled_runs={scheduled} failed_runs={failed}".format(
            status=result["summary"]["status"],
            scheduled=result["summary"]["scheduled_runs"],
            failed=result["summary"]["failed_runs"],
        )
    )


if __name__ == "__main__":
    main()
