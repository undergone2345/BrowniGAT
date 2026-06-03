import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_schema import load_checkpoint_payload
from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class ArtifactAndEarlyStoppingTests(unittest.TestCase):
    def test_artifact_index_and_early_stopping_metadata_are_written(self):
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
            foundation_cfg["training"]["epochs"] = 5
            foundation_cfg["early_stopping"] = {
                "patience": 0,
                "min_delta": 1.0,
                "mode": "min",
            }

            result = run_foundation_training(foundation_cfg, workspace_dir)

            artifact_index_path = workspace_dir / "artifact_index.json"
            best_checkpoint_path = workspace_dir / "checkpoints" / "best.json"
            final_checkpoint_path = Path(result["summary"]["latest_checkpoint"])

            self.assertTrue(artifact_index_path.exists())
            self.assertTrue((workspace_dir / "training_summary.json").exists())
            self.assertTrue(best_checkpoint_path.exists())
            self.assertTrue(final_checkpoint_path.exists())
            self.assertLess(result["summary"]["completed_epochs"], foundation_cfg["training"]["epochs"])
            self.assertTrue(result["summary"]["early_stopped"])

            best_payload = load_checkpoint_payload(best_checkpoint_path)
            self.assertIn("early_stopping_state", best_payload)
            self.assertEqual(best_payload["early_stopping_state"]["best_epoch"], 1)

            artifact_payload = load_checkpoint_payload(artifact_index_path)
            self.assertEqual(artifact_payload["best_checkpoint"], "checkpoints/best.json")
            self.assertEqual(
                Path(artifact_payload["final_checkpoint"]).name,
                final_checkpoint_path.name,
            )


if __name__ == "__main__":
    unittest.main()
