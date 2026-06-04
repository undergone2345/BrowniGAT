import json
import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_schema import load_checkpoint_payload
from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class ScalableEngineComponentTests(unittest.TestCase):
    def test_curriculum_event_log_and_checkpoint_catalog_are_written(self):
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
            result = run_foundation_training(foundation_cfg, workspace_dir)

            event_log_path = workspace_dir / "events" / "training_events.jsonl"
            checkpoint_catalog_path = workspace_dir / "checkpoints" / "checkpoint_index.json"
            checkpoint_path = workspace_dir / "checkpoints" / "epoch_002.json"

            self.assertTrue(event_log_path.exists())
            self.assertTrue(checkpoint_catalog_path.exists())
            self.assertEqual(result["summary"]["runtime_topology"]["world_size"], 1)
            self.assertGreater(result["summary"]["event_count"], 0)
            self.assertIn("manifest_partition_summary", result["summary"])
            self.assertIn("data_sharding", result["summary"])

            checkpoint_payload = load_checkpoint_payload(checkpoint_path)
            self.assertIn("curriculum_state", checkpoint_payload)
            self.assertIn("runtime_topology", checkpoint_payload["engine_state"])
            self.assertIn("data_shard_state", checkpoint_payload)

            with event_log_path.open("r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(any(event["event_type"] == "epoch_start" for event in events))
            self.assertTrue(any(event["event_type"] == "checkpoint_saved" for event in events))

            with checkpoint_catalog_path.open("r", encoding="utf-8") as handle:
                checkpoint_catalog = json.load(handle)
            self.assertGreaterEqual(len(checkpoint_catalog), 1)
            self.assertEqual(checkpoint_catalog[0]["phase_name"], "graph_warmup")



if __name__ == "__main__":
    unittest.main()
