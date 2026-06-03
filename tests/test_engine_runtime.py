import tempfile
import unittest
from pathlib import Path

from utils.checkpoint_schema import load_checkpoint_payload
from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class EngineRuntimeTests(unittest.TestCase):
    def test_engine_mode_writes_run_registry_and_checkpoint_metadata(self):
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

            checkpoint_path = workspace_dir / "checkpoints" / "epoch_002.json"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = load_checkpoint_payload(checkpoint_path)
            self.assertIn("engine_state", checkpoint)
            self.assertIn("scaler_state", checkpoint)
            self.assertTrue((workspace_dir / "run_registry.jsonl").exists())
            self.assertEqual(result["summary"]["mode"], "engine")


if __name__ == "__main__":
    unittest.main()
