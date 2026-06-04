import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.foundation_workspace import build_foundation_workspace
from utils.pretraining_manifest import load_canonical_bundle
from utils.real_data_importers import ingest_multimodal_sources


class FoundationWorkspaceTests(unittest.TestCase):
    def test_foundation_workspace_builds_manifest_and_sampling_plan(self):
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

            self.assertTrue((workspace_dir / "pretraining_manifest.json").exists())
            self.assertGreater(workspace["summary"]["num_modalities"], 0)
            self.assertGreater(workspace["summary"]["num_enabled_tasks"], 0)
            self.assertGreater(len(workspace["sampling_plan"]), 0)
            self.assertIn("partition_summary", workspace["summary"])

    def test_canonical_bundle_loader_reads_exported_modalities(self):
        real_cfg = load_config("config/real_data_example.yaml")
        bundle = ingest_multimodal_sources(real_cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            export_foundation_bundle(bundle, bundle_dir)
            loaded = load_canonical_bundle(bundle_dir)

            self.assertIn("drug_target", loaded["tables"])
            self.assertGreater(len(loaded["nodes_df"]), 0)
            self.assertGreater(len(loaded["edges_df"]), 0)


if __name__ == "__main__":
    unittest.main()
