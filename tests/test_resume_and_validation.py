import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_schema import load_checkpoint_payload
from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class ResumeAndValidationTests(unittest.TestCase):
    def test_best_checkpoint_and_validation_metrics_are_written(self):
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

            best_path = workspace_dir / "checkpoints" / "best.json"
            self.assertTrue(best_path.exists())
            best_payload = load_checkpoint_payload(best_path)
            self.assertIn("validation_metrics", best_payload)
            self.assertIn("overall_validation_loss", best_payload["validation_metrics"])
            self.assertIn("best_validation_loss", result["summary"])

    def test_resume_from_checkpoint_continues_training(self):
        real_cfg = load_config("config/real_data_example.yaml")
        foundation_cfg = load_config("config/foundation_engine_example.yaml")["foundation"]
        bundle = ingest_multimodal_sources(real_cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bundle_dir = tmpdir_path / "bundle"
            workspace_dir = tmpdir_path / "workspace"
            export_foundation_bundle(bundle, bundle_dir)

            first_cfg = dict(foundation_cfg)
            first_cfg["source_bundle_dir"] = str(bundle_dir)
            first_cfg["training"] = dict(foundation_cfg["training"])
            first_cfg["training"]["epochs"] = 1
            run_foundation_training(first_cfg, workspace_dir)

            resume_cfg = dict(foundation_cfg)
            resume_cfg["source_bundle_dir"] = str(bundle_dir)
            resume_cfg["training"] = dict(foundation_cfg["training"])
            resume_cfg["training"]["epochs"] = 2
            resume_cfg["training"]["resume_from_checkpoint"] = str(
                workspace_dir / "checkpoints" / "epoch_001.json"
            )
            result = run_foundation_training(resume_cfg, workspace_dir)

            self.assertGreaterEqual(result["summary"]["global_steps"], 40)
            self.assertEqual(result["summary"]["resume_from_checkpoint"], str(workspace_dir / "checkpoints" / "epoch_001.json"))


if __name__ == "__main__":
    unittest.main()
