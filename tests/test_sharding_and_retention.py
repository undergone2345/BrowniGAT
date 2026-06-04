import json
import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.foundation_workspace import build_foundation_workspace
from utils.real_data_importers import ingest_multimodal_sources
from utils.sampler_plan import build_task_sampling_sequence


class ShardingAndRetentionTests(unittest.TestCase):
    def test_manifest_builds_partition_summary(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_example.yaml")["foundation"]
        bundle = ingest_multimodal_sources(real_cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "workspace"
            export_foundation_bundle(bundle, bundle_dir)

            foundation_cfg = dict(foundation_cfg)
            foundation_cfg["source_bundle_dir"] = str(bundle_dir)
            workspace = build_foundation_workspace(foundation_cfg, workspace_dir)

            self.assertIn("partition_summary", workspace["manifest"])
            self.assertGreater(workspace["manifest"]["partition_summary"]["total_partitions"], 0)

    def test_worker_aware_sampler_and_checkpoint_retention(self):
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
            foundation_cfg["training"] = dict(foundation_cfg["training"])
            foundation_cfg["training"]["epochs"] = 3
            foundation_cfg["checkpoint_retention"] = {
                "keep_last_per_phase": 1,
                "keep_best": True,
            }
            result = run_foundation_training(foundation_cfg, workspace_dir)

            checkpoint_catalog = workspace_dir / "checkpoints" / "checkpoint_index.json"
            self.assertTrue((workspace_dir / "checkpoints" / "epoch_001.json").exists())
            self.assertTrue((workspace_dir / "checkpoints" / "epoch_003.json").exists())
            self.assertFalse((workspace_dir / "checkpoints" / "epoch_002.json").exists())
            self.assertEqual(result["summary"]["data_sharding"]["world_size"], 1)
            self.assertIn("task_shards", result["summary"]["data_sharding"])

            with checkpoint_catalog.open("r", encoding="utf-8") as handle:
                catalog_rows = json.load(handle)
            self.assertEqual(len(catalog_rows), 3)

            sampling_plan = [
                {"task_name": "a", "steps": 3, "batch_size": 2},
                {"task_name": "b", "steps": 3, "batch_size": 2},
            ]
            worker_zero = build_task_sampling_sequence(
                sampling_plan,
                repeat_strategy="round_robin",
                worker_index=0,
                num_workers=2,
            )
            worker_one = build_task_sampling_sequence(
                sampling_plan,
                repeat_strategy="round_robin",
                worker_index=1,
                num_workers=2,
            )
            self.assertEqual(len(worker_zero), 3)
            self.assertEqual(len(worker_one), 3)
            self.assertEqual(worker_zero[0]["task_name"], "a")
            self.assertEqual(worker_one[0]["task_name"], "b")


if __name__ == "__main__":
    unittest.main()
