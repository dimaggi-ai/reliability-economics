"""Negative controls for the publication consistency gate."""
from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch

from check_research_claims import ROOT, DOCUMENTS, Config, check, claim_block, update_block


class ResearchClaimsTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".pytest_cache" / "research-claims"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = (ROOT / "results/results.csv").read_text()
        self.summary = (ROOT / "results/summary.md").read_text()
        self.sensitivity = (ROOT / "results/recovery-sensitivity.json").read_text()
        for name in ("results/results.csv", "results/summary.md", "results/recovery-sensitivity.json",
                     "evidence.json", *DOCUMENTS):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text((ROOT / name).read_text())

    def test_current_artifacts_and_prose_pass(self):
        self.assertEqual(check(self.root, self.csv, self.summary, self.sensitivity), [])

    def test_changed_csv_refused_before_prose_can_be_rewritten(self):
        path = self.root / "results/results.csv"
        path.write_text(self.csv.replace("24816774.875", "24816775.875"))
        before = (self.root / "README.md").read_bytes()
        self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity, write=True))
        self.assertEqual((self.root / "README.md").read_bytes(), before)

    def test_changed_generated_summary_refused(self):
        (self.root / "results/summary.md").write_text("stale summary\n")
        self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity))

    def test_changed_sensitivity_refused(self):
        (self.root / "results/recovery-sensitivity.json").write_text("{}\n")
        self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity))

    def test_stale_numeric_manifest_claims_refused_despite_unchanged_hashes(self):
        path = self.root / "evidence.json"
        original = path.read_text()
        for name in ("manual-to-eight-spare difference", "elastic-to-eight-spare difference",
                     "eight-spare nonproductive capacity value"):
            manifest = json.loads(original)
            next(row for row in manifest["results"] if row["name"] == name)["value"] += 1
            path.write_text(json.dumps(manifest))
            before = (self.root / "README.md").read_bytes()
            self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity, write=True), name)
            self.assertEqual((self.root / "README.md").read_bytes(), before)
        path.write_text(original)

    def test_missing_or_duplicate_manifest_claim_refused(self):
        path = self.root / "evidence.json"
        original = json.loads(path.read_text())
        name = "elastic-to-eight-spare difference"
        for duplicate in (False, True):
            manifest = json.loads(json.dumps(original))
            row = next(row for row in manifest["results"] if row["name"] == name)
            if duplicate:
                manifest["results"].append(row)
            else:
                manifest["results"].remove(row)
            path.write_text(json.dumps(manifest))
            self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity))

    def test_changed_assumptions_refuse_even_when_artifact_inputs_agree(self):
        for field, value in (("nodes", 1024), ("horizon_h", 1440),
                             ("dollars_per_gpu_h", 3.0), ("repair_h", 2.0),
                             ("single_rate_per_node_h", 0.0002),
                             ("ckpt_overlap_write_h", 0.002)):
            changed = replace(Config(), **{field: value})
            with patch("check_research_claims.Config", return_value=changed):
                self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity, write=True), field)
        with patch("check_research_claims.NODE_GPUS", 4):
            self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity))

    def test_each_changed_economic_block_refused_and_restored(self):
        for name in DOCUMENTS:
            path = self.root / name
            original = path.read_text()
            path.write_text(original.replace("**$20,953,975.000**", "**$20,953,976.000**"))
            self.assertTrue(check(self.root, self.csv, self.summary, self.sensitivity), name)
            self.assertEqual(check(self.root, self.csv, self.summary, self.sensitivity, write=True), [])
            self.assertEqual(path.read_text(), original)

    def test_missing_or_duplicate_blocks_refused(self):
        block = claim_block(self.csv)
        for text in ("", block + "\n" + block):
            with self.assertRaises(ValueError):
                update_block(text, block)

    def test_duplicate_or_missing_headline_row_refused(self):
        row = next(line for line in self.csv.splitlines() if line.startswith("independent,manual,"))
        for text in (self.csv + row + "\n", self.csv.replace(row + "\n", "")):
            with self.assertRaises(ValueError):
                claim_block(text)


if __name__ == "__main__":
    unittest.main()
