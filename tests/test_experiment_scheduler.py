import json
import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.experiment_scheduler import schedule_foundation_experiments
from utils.foundation_ready import export_foundation_bundle
from utils.real_data_importers import ingest_multimodal_sources


class ExperimentSchedulerTests(unittest.TestCase):
    def test_scheduler_builds_plan_without_execution(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_engine_example.yaml")["foundation"]
        foundation_cfg = dict(foundation_cfg)
        foundation_cfg["experiment_scheduler"] = dict(foundation_cfg["experiment_scheduler"])
        foundation_cfg["experiment_scheduler"]["execute_queue"] = False
        bundle = ingest_multimodal_sources(real_cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "scheduler_workspace"
            export_foundation_bundle(bundle, bundle_dir)
            foundation_cfg["source_bundle_dir"] = str(bundle_dir)

            result = schedule_foundation_experiments(foundation_cfg, workspace_dir)

            self.assertEqual(result["summary"]["status"], "planned")
            self.assertGreaterEqual(len(result["stage_plan"]), 2)
            self.assertTrue((workspace_dir / "scheduler" / "run_queue.json").exists())
            self.assertTrue((workspace_dir / "scheduler" / "scheduler_plan.json").exists())

    def test_scheduler_retries_failed_stage_and_passes_resume_checkpoint(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_engine_example.yaml")["foundation"]
        foundation_cfg = dict(foundation_cfg)
        foundation_cfg["experiment_scheduler"] = dict(foundation_cfg["experiment_scheduler"])
        foundation_cfg["experiment_scheduler"]["execute_queue"] = True
        foundation_cfg["experiment_scheduler"]["failure_recovery"] = {
            "max_retries": 1,
            "resume_strategy": "latest",
        }
        bundle = ingest_multimodal_sources(real_cfg["real_data"])
        runner_calls = []
        attempt_count = {}

        def fake_runner(stage_cfg, stage_workspace):
            stage_workspace = Path(stage_workspace)
            stage_workspace.mkdir(parents=True, exist_ok=True)
            checkpoints_dir = stage_workspace / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            attempt_key = str(stage_workspace)
            attempt_count[attempt_key] = attempt_count.get(attempt_key, 0) + 1
            runner_calls.append(
                {
                    "experiment_name": stage_cfg["experiment"]["name"],
                    "resume_from_checkpoint": stage_cfg["training"].get("resume_from_checkpoint"),
                    "workspace_dir": str(stage_workspace),
                    "attempt": attempt_count[attempt_key],
                }
            )

            if stage_workspace.name == "stage_01" and attempt_count[attempt_key] == 1:
                (checkpoints_dir / "epoch_001.json").write_text("{}", encoding="utf-8")
                raise RuntimeError("simulated transient failure")

            latest_checkpoint = checkpoints_dir / f"epoch_{int(stage_cfg['training']['epochs']):03d}.json"
            latest_checkpoint.write_text("{}", encoding="utf-8")
            return {"summary": {"latest_checkpoint": str(latest_checkpoint)}}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "scheduler_workspace"
            export_foundation_bundle(bundle, bundle_dir)
            foundation_cfg["source_bundle_dir"] = str(bundle_dir)

            result = schedule_foundation_experiments(foundation_cfg, workspace_dir, runner=fake_runner)

            self.assertEqual(result["summary"]["failed_runs"], 0)
            self.assertEqual(result["summary"]["scheduled_runs"], 2)
            self.assertEqual(result["run_queue"][0]["retry_count"], 1)
            self.assertTrue(result["run_queue"][1]["resume_from_checkpoint"].endswith("epoch_001.json"))

            retry_calls = [call for call in runner_calls if call["workspace_dir"].endswith("stage_01")]
            self.assertEqual(len(retry_calls), 2)
            self.assertTrue(retry_calls[1]["resume_from_checkpoint"].endswith("epoch_001.json"))

            summary_path = workspace_dir / "scheduler" / "scheduler_summary.json"
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["scheduled_runs"], 2)


if __name__ == "__main__":
    unittest.main()
