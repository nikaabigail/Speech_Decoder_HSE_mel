"""Generated files and subprocess mocks; no scientific run is started."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import ensemble as en
import multiseed as ms


class PlanTests(unittest.TestCase):
    def test_unique_portable_plan_and_optional_mel(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = ms.plan(Path(temporary) / "runs", Path(temporary) / "patients.json",
                           "fixture", "ecog8", seeds=[4, 42], include_mel=True)
            self.assertEqual(len(rows), 14)
            self.assertEqual(len({str(r["run_dir"]) for r in rows}), 14)
            for row in rows:
                self.assertEqual(row["command"][0], sys.executable)
                self.assertIn("--run-dir", row["command"])
                self.assertIn("--stage", row["command"])
                self.assertEqual(row["command"][-1] if row["identity"]["target"] == "mel" else row["command"][-3], "both")
            self.assertEqual(rows[6]["identity"]["target"], "mel")
            self.assertIsNone(rows[6]["identity"]["layer"])

    def test_duplicate_or_negative_seeds_rejected(self):
        for seeds in ([4, 4], [-1], []):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                ms.plan("runs", "patients.json", "fixture", "ecog8", seeds)

    def test_dry_run_never_spawns_or_creates_run_dirs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patients = root / "patients.json"; patients.write_text('[]')
            rows = ms.plan(root / "runs", patients, "fixture", "ecog8", [4])
            with patch.object(ms.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                result = ms.execute(rows, patients, dry_run=True)
            run.assert_not_called()
            self.assertTrue(all(v["action"] == "train" for v in result))
            self.assertFalse((root / "runs").exists())

    def test_all_existing_paths_are_checked_before_any_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patients = root / "patients.json"; patients.write_text('[]')
            rows = ms.plan(root / "runs", patients, "fixture", "ecog8", [4])
            bad = rows[-1]["run_dir"]; bad.mkdir(parents=True)
            (bad / "partial.txt").write_text("synthetic interrupted run")
            with patch.object(ms.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                ms.execute(rows, patients)
            run.assert_not_called()
            self.assertTrue((bad / "partial.txt").exists())

    def test_verified_completion_reuse_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patients = root / "patients.json"; patients.write_text('[]')
            row = ms.plan(root / "runs", patients, "fixture", "ecog8", [4])[0]
            run_dir = row["run_dir"]; run_dir.mkdir(parents=True)
            complete = dict(schema="public_training_completion_v1", stage="both", **row["identity"],
                            patients_sha256=en.sha256(patients), artifact_sha256={})
            for key in en.ARTIFACT_KEYS:
                relative = key + (".json" if key.endswith("report") else ".pth")
                path = run_dir / relative
                if key.endswith("report"):
                    path.write_text(json.dumps({"config": {"seed": 4, "patient": "fixture",
                        "regression_model": row["identity"]["model_name"]}}))
                else:
                    path.write_bytes(b"generated fixture")
                complete[key] = relative; complete["artifact_sha256"][relative] = en.sha256(path)
            (run_dir / "complete.json").write_text(json.dumps(complete))
            with patch.object(ms.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                result = ms.execute([row], patients)
            run.assert_not_called(); self.assertEqual(result[0]["action"], "reuse_verified")
            (run_dir / complete["regression_checkpoint"]).write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                ms.check_reusable(run_dir, row["identity"], patients)


if __name__ == "__main__":
    unittest.main()
