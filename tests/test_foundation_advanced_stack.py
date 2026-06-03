import tempfile
import unittest
from pathlib import Path

from model.foundation_backbone import build_foundation_backbone
from utils.config import load_config
from utils.foundation_dataset import MultimodalPretrainingDataset, collate_foundation_batch
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_trainer import run_foundation_training
from utils.real_data_importers import ingest_multimodal_sources


class FoundationAdvancedStackTests(unittest.TestCase):
    def test_backbone_and_dataset_pipeline_work_together(self):
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
            run_foundation_training(foundation_cfg, workspace_dir)

            dataset = MultimodalPretrainingDataset(workspace_dir)
            batch = dataset.get_task_samples("masked_edge_modeling")[:4]
            collated = collate_foundation_batch(batch)
            backbone = build_foundation_backbone(foundation_cfg["backbone"])
            encoded = backbone.encode_batch(collated["raw_batch"])

            self.assertEqual(collated["batch_size"], 4)
            self.assertEqual(encoded["backbone_name"], foundation_cfg["backbone"]["name"])

    def test_training_writes_experiment_manifest(self):
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

            self.assertTrue((workspace_dir / "experiment_manifest.json").exists())
            self.assertIn("backbone_name", result["summary"])
            self.assertIn("gradient_accumulation_steps", result["summary"])


if __name__ == "__main__":
    unittest.main()
