"""Synthetic math and artifact-provenance checks; no trained models or data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import ensemble as en


class ProbabilityTests(unittest.TestCase):
    def test_stable_softmax_and_equal_probability_average(self):
        logits = np.zeros((2, 27))
        logits[0, 4] = 1000
        logits[1, 7] = 1000
        values = en.softmax(logits)
        np.testing.assert_allclose(values.sum(1), 1)
        rows = [dict(layer=layer, probs=values, classes=np.array([4, 7]),
                     event_ids=["fixture:0", "fixture:1"]) for layer in en.LAYERS]
        result = en.fixed_ensemble(rows)
        self.assertEqual((result["correct"], result["count"], result["accuracy"]), (2, 2, 1.0))
        np.testing.assert_allclose(result["probabilities"], values)

    def test_average_probabilities_not_majority_vote(self):
        rows = []
        for layer, first, second in [(3, .51, .49), (4, .51, .49), (5, .01, .99)]:
            values = np.zeros((1, 27)); values[0, :2] = [first, second]
            rows.append(dict(layer=layer, probs=values, classes=np.array([1]), event_ids=["one"]))
        self.assertEqual(en.fixed_ensemble(rows)["correct"], 1)

    def test_event_identity_is_checked_even_for_identical_labels(self):
        values = np.full((2, 27), 1 / 27)
        rows = [dict(layer=layer, probs=values, classes=np.array([1, 1]),
                     event_ids=["a", "b"]) for layer in en.LAYERS]
        rows[2]["event_ids"] = ["b", "a"]
        with self.assertRaises(ValueError):
            en.fixed_ensemble(rows)

    def test_missing_layer_wrong_labels_or_invalid_probability_rejected(self):
        values = np.full((1, 27), 1 / 27)
        rows = [dict(layer=layer, probs=values.copy(), classes=np.array([1]),
                     event_ids=["one"]) for layer in en.LAYERS]
        with self.assertRaises(ValueError):
            en.fixed_ensemble(rows[:2])
        wrong = copy.deepcopy(rows); wrong[2]["layer"] = 6
        with self.assertRaises(ValueError):
            en.fixed_ensemble(wrong)
        wrong = copy.deepcopy(rows); wrong[1]["classes"] = np.array([2])
        with self.assertRaises(ValueError):
            en.fixed_ensemble(wrong)
        wrong = copy.deepcopy(rows); wrong[0]["probs"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            en.fixed_ensemble(wrong)
        wrong = copy.deepcopy(rows)
        for row in wrong: row["classes"] = np.array([-1])
        with self.assertRaises(ValueError):
            en.fixed_ensemble(wrong)

    def test_historical_sampling_rng_and_duplicate_silence_indices(self):
        labels = np.array([0, 1, 0, 2, 1, 2, 1, 2, 1, 2])
        rng = np.random.RandomState(42)
        _ = en.balanced_indices(np.array([0, 1, 1, 2]), rng)
        actual = en.balanced_indices(labels, rng)
        baseline = np.random.RandomState(42)
        first = np.array([0, 1, 1, 2])
        baseline.choice(np.where(first == 0)[0], int(np.sum(first != 0) / len(set(first))))
        words = np.where(labels != 0)[0]
        expected = np.sort(np.concatenate([baseline.choice(np.where(labels == 0)[0],
                       int(len(words) / len(set(labels)))), words]))
        np.testing.assert_array_equal(actual, expected)

    def test_class_order_includes_internal_silence_only(self):
        rows = [[0, 5, "a"], [8, 12, "b"], [12, 15, "a"]]
        self.assertEqual(en.classes_from_words(rows, {"silent": 0, "a": 1, "b": 2}), [1, 0, 2, 1])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patients = self.root / "patients.json"
        self.patients.write_text(json.dumps([dict(name="fixture",
            files_list=["data/a.h5", "data/b.h5", "data/c.h5"], sampling_rate=4096,
            ecog_channels=list(range(16)), sound_channel=16,
            test_start_file_regression_index=2, test_start_file_classification_index=2)]), encoding="utf-8")
        self.run = self.root / "run"; self.run.mkdir()
        self.expected = dict(patient="fixture", layout="ecog8", target="whisper", layer=3,
                             seed=4, model_name=en.model_name("ecog8", 3))
        self.complete = dict(schema="public_training_completion_v1", stage="both", **self.expected,
                             patients_sha256=en.sha256(self.patients), artifact_sha256={})
        for key in en.ARTIFACT_KEYS:
            relative = key + (".json" if key.endswith("report") else ".pth")
            path = self.run / relative
            if key.endswith("report"):
                path.write_text(json.dumps({"config": {"patient": "fixture", "seed": 4,
                    "regression_model": self.expected["model_name"]}}), encoding="utf-8")
            else:
                path.write_bytes(b"synthetic checkpoint; never loaded")
            self.complete[key] = relative
            self.complete["artifact_sha256"][relative] = en.sha256(path)
        self.save_complete()

    def tearDown(self):
        self.temporary.cleanup()

    def save_complete(self):
        (self.run / "complete.json").write_text(json.dumps(self.complete), encoding="utf-8")

    def test_verified_pair_and_relative_data_paths(self):
        _, artifacts = en.validate_complete(self.run, self.expected, self.patients)
        self.assertEqual(len(artifacts), 4)
        patient = en.load_patient(self.patients, "fixture")
        self.assertEqual(patient["files_list"], [str((self.root / f"data/{name}.h5").resolve())
                                                for name in ("a", "b", "c")])

    def test_changed_artifact_config_or_identity_rejected(self):
        checkpoint = self.run / self.complete["regression_checkpoint"]
        checkpoint.write_bytes(b"different")
        with self.assertRaises(ValueError):
            en.validate_complete(self.run, self.expected, self.patients)
        checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
        changed = dict(self.expected, seed=42)
        with self.assertRaises(ValueError):
            en.validate_complete(self.run, changed, self.patients)
        self.patients.write_text('[]', encoding="utf-8")
        with self.assertRaises(ValueError):
            en.validate_complete(self.run, self.expected, self.patients)

    def test_path_escape_and_report_mismatch_rejected(self):
        outside = self.root / "outside.pth"; outside.write_bytes(b"outside")
        with self.assertRaises(ValueError):
            en.validated_artifact(self.run, "../outside.pth", en.sha256(outside))
        with self.assertRaises(ValueError):
            en.validated_artifact(self.run, str(outside), en.sha256(outside))
        path = self.run / self.complete["classification_report"]
        path.write_text(json.dumps({"config": {"patient": "fixture", "seed": 9}}), encoding="utf-8")
        self.complete["artifact_sha256"][path.name] = en.sha256(path); self.save_complete()
        with self.assertRaises(ValueError):
            en.validate_complete(self.run, self.expected, self.patients)


if __name__ == "__main__":
    unittest.main()
