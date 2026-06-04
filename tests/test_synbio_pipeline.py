import tempfile
import unittest
from pathlib import Path

from synbio_main import run_synbio_pipeline


class SyntheticBiologyPipelineTests(unittest.TestCase):
    def test_synbio_pipeline_generates_design_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "synbio"
            run_synbio_pipeline("config/synbio_toy.yaml", output_dir_override=output_dir)

            self.assertTrue((output_dir / "gene_program_design.tsv").exists())
            self.assertTrue((output_dir / "pathway_rewiring.tsv").exists())
            self.assertTrue((output_dir / "construct_blueprints.tsv").exists())
            self.assertTrue((output_dir / "synbio_summary.json").exists())
            self.assertTrue((output_dir / "SYNBIO_REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
