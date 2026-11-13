"""Generated/package-only checks: no patient input, weights or model execution."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import train as training


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".cff", ".csv", ".sh", ".ps1"}


def public_files():
    return [p for p in ROOT.rglob("*") if p.is_file() and
            not any(part in {".git", "__pycache__", ".pytest_cache"}
                    for part in p.relative_to(ROOT).parts)]


class PublicationContractTests(unittest.TestCase):
    def regression_fixture(self, root):
        run = (root / "run").resolve()
        (run / "model_dumps").mkdir(parents=True)
        (run / "results").mkdir()
        identity = dict(patient="synthetic", layout="ecog8", target="whisper", layer=4,
                        seed=4, model_name=training.model_name("ecog8", "whisper", 4))
        weights = run / "model_dumps/regression.pth"
        weights.write_bytes(b"generated fixture; never loaded as a scientific model")
        report = run / "results/regression.json"
        report.write_text(json.dumps(dict(config=dict(patient=identity["patient"], seed=4,
                          mode="regression", model=identity["model_name"]))), encoding="utf-8")
        complete = dict(schema="public_training_completion_v1", stage="regression",
                        **identity, patients_sha256="synthetic-config-hash",
                        regression_checkpoint=weights.relative_to(run).as_posix(),
                        regression_report=report.relative_to(run).as_posix(),
                        artifact_sha256={p.relative_to(run).as_posix(): training.sha256(p)
                                         for p in (weights, report)})
        (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
        return run, identity

    def test_completed_regression_can_be_reserved_without_changing_artifacts(self):
        with tempfile.TemporaryDirectory() as work:
            run, identity = self.regression_fixture(Path(work))
            before = {p.relative_to(run).as_posix(): p.read_bytes()
                      for p in run.rglob("*") if p.is_file()}
            with training.reserve_run(run, "hidden", identity, "synthetic-config-hash") as previous:
                self.assertEqual(previous["stage"], "regression")
                self.assertTrue((run / ".run.lock").is_file())
            after = {p.relative_to(run).as_posix(): p.read_bytes()
                     for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_partial_hidden_attempts_are_refused_and_preserved(self):
        partials = ("model_dumps/classification_hidden_partial.pth",
                    "results/classification_hidden_partial.json", "failed.json",
                    "regression_complete.json")
        for partial in partials:
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as work:
                run, identity = self.regression_fixture(Path(work))
                (run / partial).write_bytes(b"generated interrupted-stage fixture")
                before = {p.relative_to(run).as_posix(): p.read_bytes()
                          for p in run.rglob("*") if p.is_file()}
                with self.assertRaises(ValueError):
                    with training.reserve_run(run, "hidden", identity, "synthetic-config-hash"):
                        self.fail("A partial hidden run must not reach scientific execution")
                after = {p.relative_to(run).as_posix(): p.read_bytes()
                         for p in run.rglob("*") if p.is_file()}
                self.assertEqual(before, after)

    def test_existing_exclusive_lock_is_not_removed_by_second_writer(self):
        with tempfile.TemporaryDirectory() as work:
            run, identity = self.regression_fixture(Path(work))
            lock = run / ".run.lock"
            lock.write_text("synthetic-owner", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with training.reserve_run(run, "hidden", identity, "synthetic-config-hash"):
                    self.fail("Concurrent owner must be refused")
            self.assertEqual(lock.read_text(encoding="utf-8"), "synthetic-owner")
            lock.unlink()
            with training.reserve_run(run, "hidden", identity, "synthetic-config-hash"):
                with self.assertRaises(FileExistsError):
                    with training.reserve_run(run, "hidden", identity, "synthetic-config-hash"):
                        self.fail("Nested second writer must be refused")
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

    def test_all_python_sources_parse(self):
        sources = [p for p in public_files() if p.suffix == ".py"]
        self.assertGreater(len(sources), 5)
        for source in sources:
            with self.subTest(source=source.relative_to(ROOT).as_posix()):
                ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))

    def test_help_needs_only_standard_library_and_writes_nothing(self):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        for entrypoint in ("train.py", "multiseed.py", "ensemble.py"):
            with self.subTest(entrypoint=entrypoint), tempfile.TemporaryDirectory() as work:
                result = subprocess.run(
                    [sys.executable, "-B", "-S", str(ROOT / entrypoint), "--help"],
                    cwd=work, env=environment, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                self.assertIn("usage:", result.stdout.lower())
                self.assertEqual(list(Path(work).iterdir()), [],
                                 "Even --help must not create run/data artifacts")

    def test_no_patient_arrays_weights_or_credentials_in_repository(self):
        prohibited = {".pt", ".pth", ".npy", ".npz", ".h5", ".hdf5", ".mat",
                      ".wav", ".mp3", ".safetensors", ".key", ".pem", ".pkl", ".joblib"}
        for path in public_files():
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertNotIn(path.suffix.lower(), prohibited)
                self.assertNotIn(path.name, {"patients.json", "id_ed25519", "id_rsa", ".env"})
                if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith("."):
                    continue
                content = path.read_text(encoding="utf-8-sig")
                self.assertNotRegex(content, r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
                self.assertNotRegex(content, r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")

    def test_no_workstation_paths_in_public_files(self):
        # Build the expressions in parts so this test does not flag its own text.
        forbidden = ["C:" + r"[\\/]Users[\\/]", "D:" + r"[\\/](?:Research|Osadchi|SiriusBackups)[\\/]",
                     r"/datasets/" + r"research/baryshev/", r"/home/" + r"baryshev/"]
        for path in public_files():
            if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith("."):
                continue
            content = path.read_text(encoding="utf-8-sig")
            for pattern in forbidden:
                with self.subTest(path=path.relative_to(ROOT).as_posix(), pattern=pattern):
                    self.assertIsNone(re.search(pattern, content, re.I))

    def test_example_configuration_is_metadata_only(self):
        config = json.loads((ROOT / "configs/patients.example.json").read_text(encoding="utf-8"))
        self.assertIsInstance(config, list)
        self.assertEqual(len(config), 2)
        for patient in config:
            with self.subTest(patient=patient.get("name")):
                self.assertIn(patient["name"], ("ecog_example", "seeg_example"))
                self.assertGreaterEqual(len(patient["files_list"]), 3)
                for filename in patient["files_list"]:
                    self.assertFalse(Path(filename).is_absolute())
                    self.assertNotRegex(filename, r"^[A-Za-z]:")
                    self.assertNotIn("..", Path(filename).parts)
                    self.assertFalse((ROOT / filename).exists())

    def test_historical_metric_limits_and_upstream_credit_are_present(self):
        results = (ROOT / "HISTORICAL_RESULTS.md").read_text(encoding="utf-8").lower()
        provenance = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8").lower()
        notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8").lower()
        self.assertIn("sample", results)
        self.assertIn("445", results)
        self.assertIn("seed", results)
        self.assertIn("histor", provenance)
        self.assertIn("pet67/ossadtchi-ml-test-bench-speech", notice)
        self.assertRegex(notice, r"not specified|not provided|no license|license.*null|unspecified")

    def test_unrelated_legacy_dependency_freeze_was_removed(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for unrelated in ("tensorflow", "django", "box2d", "systemd-python", "python-apt"):
            self.assertNotIn(unrelated, requirements)
        self.assertIn("torch", requirements)
        self.assertIn("transformers", requirements)
        self.assertLess(len(requirements.splitlines()), 45)


if __name__ == "__main__":
    unittest.main()
