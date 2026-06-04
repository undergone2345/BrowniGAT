import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ai_loop_utils import build_loop_summary, parse_deadline, should_continue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a repeatable BrowniGAT engineering loop until a deadline."
    )
    parser.add_argument(
        "--config",
        default="config/ai_loop_example.yaml",
        help="Path to AI loop YAML config.",
    )
    parser.add_argument(
        "--deadline",
        default=None,
        help="Optional deadline override such as 24:00-style alias 'tonight_24', '23:30', or ISO datetime.",
    )
    return parser.parse_args()


def load_loop_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def append_jsonl(output_path, payload):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_step(step, repo_root):
    cwd_value = step.get("cwd")
    cwd = repo_root / cwd_value if cwd_value else repo_root
    cwd = cwd.resolve()
    started_at = datetime.now()
    completed = subprocess.run(
        step["command"],
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
    )
    finished_at = datetime.now()
    return {
        "name": step["name"],
        "command": step["command"],
        "cwd": str(cwd),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def main():
    args = parse_args()
    config = load_loop_config(args.config)
    repo_root = Path(config.get("repo_root", ".")).resolve()
    output_dir = Path(config.get("output_dir", "results/ai_loop")).resolve()
    log_path = output_dir / "loop_runs.jsonl"
    summary_path = output_dir / "loop_summary.json"
    started_at = datetime.now().astimezone()
    deadline = parse_deadline(args.deadline or config.get("deadline", "tonight_24"), now=started_at)
    sleep_seconds = float(config.get("sleep_seconds", 30))
    max_iterations = config.get("max_iterations")
    stop_on_failure = bool(config.get("stop_on_failure", False))
    steps = config.get("steps", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        log_path,
        {
            "event": "loop_started",
            "started_at": started_at.isoformat(),
            "deadline": deadline.isoformat(),
            "config_path": str(Path(args.config).resolve()),
        },
    )

    iteration = 0
    success_count = 0
    failure_count = 0

    while should_continue(datetime.now().astimezone(), deadline, iteration, max_iterations=max_iterations):
        iteration += 1
        iteration_failed = False
        append_jsonl(
            log_path,
            {
                "event": "iteration_started",
                "iteration": iteration,
                "started_at": datetime.now().astimezone().isoformat(),
            },
        )
        for step in steps:
            result = run_step(step, repo_root)
            result["event"] = "step_finished"
            result["iteration"] = iteration
            append_jsonl(log_path, result)
            if result["returncode"] != 0:
                failure_count += 1
                iteration_failed = True
                if bool(step.get("stop_on_failure", stop_on_failure)):
                    break
        if iteration_failed:
            append_jsonl(
                log_path,
                {
                    "event": "iteration_failed",
                    "iteration": iteration,
                    "recorded_at": datetime.now().astimezone().isoformat(),
                },
            )
            if stop_on_failure:
                break
        else:
            success_count += 1
            append_jsonl(
                log_path,
                {
                    "event": "iteration_succeeded",
                    "iteration": iteration,
                    "recorded_at": datetime.now().astimezone().isoformat(),
                },
            )

        if should_continue(datetime.now().astimezone(), deadline, iteration, max_iterations=max_iterations):
            time.sleep(sleep_seconds)

    summary = build_loop_summary(
        started_at=started_at,
        deadline=deadline,
        iteration_count=iteration,
        success_count=success_count,
        failure_count=failure_count,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    append_jsonl(
        log_path,
        {
            "event": "loop_finished",
            "finished_at": datetime.now().astimezone().isoformat(),
            **summary,
        },
    )
    print(f"Loop finished after {iteration} iterations. Logs: {log_path}")


if __name__ == "__main__":
    main()
