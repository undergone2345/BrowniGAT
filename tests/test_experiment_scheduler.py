import json
import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.experiment_scheduler import schedule_foundation_experiments
from utils.foundation_ready import export_foundation_bundle
from utils.real_data_importers import ingest_multimodal_sources
from utils.run_queue import build_run_queue
from utils.stage_planner import build_manifest_aware_stage_plan


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

    def test_resource_aware_priority_queue_and_promotion_gate(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_engine_example.yaml")["foundation"]
        foundation_cfg = dict(foundation_cfg)
        foundation_cfg["experiment_scheduler"] = dict(foundation_cfg["experiment_scheduler"])
        foundation_cfg["experiment_scheduler"]["execute_queue"] = True
        foundation_cfg["experiment_scheduler"]["enforce_stage_order"] = False
        foundation_cfg["experiment_scheduler"]["promotion_policy"] = {
            "enabled": True,
            "metric_name": "best_validation_loss",
            "mode": "min",
            "threshold": 0.2,
            "on_fail": "halt",
        }
        foundation_cfg["curriculum"] = {
            "phases": [
                {
                    "name": "high_priority_multimodal",
                    "epochs": 1,
                    "priority": 100,
                    "tasks": [
                        "masked_node_modeling",
                        "cross_modal_alignment",
                        "perturbation_conditioning",
                    ],
                },
                {
                    "name": "lower_priority_structural",
                    "epochs": 1,
                    "priority": 20,
                    "tasks": [
                        "masked_node_modeling",
                        "masked_edge_modeling",
                    ],
                },
                {
                    "name": "should_be_skipped_after_gate",
                    "epochs": 1,
                    "priority": 10,
                    "tasks": [
                        "masked_node_modeling",
                        "masked_edge_modeling",
                    ],
                },
            ]
        }
        bundle = ingest_multimodal_sources(real_cfg["real_data"])
        runner_calls = []

        def fake_runner(stage_cfg, stage_workspace):
            runner_calls.append(stage_cfg["experiment"]["name"])
            stage_workspace = Path(stage_workspace)
            stage_workspace.mkdir(parents=True, exist_ok=True)
            checkpoint_path = stage_workspace / "checkpoints" / "epoch_001.json"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("{}", encoding="utf-8")
            metric_value = 0.5 if "high_priority_multimodal" in stage_cfg["experiment"]["name"] else 0.05
            return {
                "summary": {
                    "latest_checkpoint": str(checkpoint_path),
                    "best_validation_loss": metric_value,
                }
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "scheduler_workspace"
            export_foundation_bundle(bundle, bundle_dir)
            foundation_cfg["source_bundle_dir"] = str(bundle_dir)

            result = schedule_foundation_experiments(foundation_cfg, workspace_dir, runner=fake_runner)

            self.assertEqual(result["summary"]["scheduled_runs"], 1)
            self.assertGreaterEqual(result["summary"]["skipped_runs"], 1)
            self.assertIn("high_priority_multimodal", runner_calls[0])
            self.assertEqual(result["run_queue"][0]["priority"], 100)
            self.assertFalse(result["run_queue"][0]["promotion_decision"]["passed"])

    def test_stage_plan_and_run_queue_expose_resource_requests(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_engine_example.yaml")["foundation"]
        bundle = ingest_multimodal_sources(real_cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "workspace"
            export_foundation_bundle(bundle, bundle_dir)
            foundation_cfg = dict(foundation_cfg)
            foundation_cfg["source_bundle_dir"] = str(bundle_dir)

            from utils.foundation_workspace import build_foundation_workspace

            workspace = build_foundation_workspace(foundation_cfg, workspace_dir)
            stage_plan = build_manifest_aware_stage_plan(workspace["manifest"], foundation_cfg)
            run_queue = build_run_queue(stage_plan, workspace_dir, foundation_cfg["experiment_scheduler"])

            self.assertIn("resource_request", stage_plan[0])
            self.assertIn("priority", stage_plan[0])
            self.assertIn("resource_request", run_queue[0])
            self.assertIn("priority", run_queue[0])


if __name__ == "__main__":
    unittest.main()
