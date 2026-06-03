import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class FoundationTrainerTests(unittest.TestCase):
    def test_training_skeleton_writes_history_and_checkpoints(self):
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
            result = run_foundation_training(foundation_cfg, workspace_dir)

            self.assertTrue((workspace_dir / "training_history.csv").exists())
            self.assertTrue((workspace_dir / "checkpoints" / "epoch_001.json").exists())
            self.assertGreater(result["summary"]["global_steps"], 0)
            self.assertIn("final_loss_mean", result["summary"])


if __name__ == "__main__":
    unittest.main()
