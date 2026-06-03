import tempfile
import unittest
from pathlib import Path

from utils.config import load_config
from utils.foundation_ready import export_foundation_bundle
from utils.real_data_importers import ingest_multimodal_sources
from utils.schema_validation import validate_schema


class RealDataIngestionTests(unittest.TestCase):
    def test_schema_validation_accepts_example_drug_target_table(self):
        cfg = load_config("config/real_data_example.yaml")
        bundle = ingest_multimodal_sources(cfg["real_data"])

        report = validate_schema(bundle["tables"]["drug_target"], "drug_target")
        self.assertTrue(report["is_valid"])
        self.assertIn("rows", report)

    def test_ingestion_builds_canonical_bundle(self):
        cfg = load_config("config/real_data_example.yaml")
        bundle = ingest_multimodal_sources(cfg["real_data"])

        self.assertIn("nodes_df", bundle)
        self.assertIn("edges_df", bundle)
        self.assertGreater(len(bundle["nodes_df"]), 0)
        self.assertGreater(len(bundle["edges_df"]), 0)
        self.assertIn("drug_target", bundle["tables"])
        self.assertIn("spatial", bundle["tables"])

    def test_foundation_bundle_export_writes_manifest(self):
        cfg = load_config("config/real_data_example.yaml")
        bundle = ingest_multimodal_sources(cfg["real_data"])

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            export_foundation_bundle(bundle, outdir)

            self.assertTrue((outdir / "canonical_nodes.tsv").exists())
            self.assertTrue((outdir / "canonical_edges.tsv").exists())
            self.assertTrue((outdir / "foundation_manifest.json").exists())
            self.assertTrue((outdir / "modalities" / "drug_target.tsv").exists())


if __name__ == "__main__":
    unittest.main()
