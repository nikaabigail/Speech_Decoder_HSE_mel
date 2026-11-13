"""Generated CPU checks: no recordings, trained weights, downloads or optimization."""
import ast
import copy
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

os.environ["ECOG_DEVICE"] = "cpu"
import numpy as np
import torch

torch.set_num_threads(1)

from library import bench_models_regression as bmr
from library import runner_classification as rc
from library import runner_common as common
from library import runtime, whisper_target as wt
from library.models_regression import SimpleNet
from library.models_classification import Mel2WordHidden


class CoreTests(unittest.TestCase):
    def test_original_record_splits_are_preserved(self):
        self.assertEqual(runtime.make_split(12, 8), (list(range(7)), [7], [8, 9, 10, 11]))
        self.assertEqual(runtime.make_split(12, 10), (list(range(9)), [9], [10, 11]))
        self.assertEqual(runtime.DEVICE.type, "cpu")

    def test_window_includes_both_endpoints_and_has_no_future(self):
        x = np.arange(1030 * 2, dtype=np.float32).reshape(1030, 2)
        y = np.arange(1030, dtype=np.float32)[:, None]
        windows, targets = next(common.data_generator(x, y, 3, 1000, 0, shuffle=False, infinite=False))
        self.assertEqual(windows.shape, (3, 2, 1001))
        np.testing.assert_array_equal(windows[0], x[:1001].T)
        np.testing.assert_array_equal(windows[2], x[2:1003].T)
        np.testing.assert_array_equal(targets[:, 0], [1000, 1001, 1002])

    def test_encoder_preserves_hidden3030_and_regression_outputs(self):
        torch.manual_seed(4)
        for channels, output in ((8, 50), (6, 40)):
            model = SimpleNet(channels, output, 1000, 0).eval()
            signal = torch.randn(2, channels, 1001)
            with torch.no_grad():
                prediction = model(signal)
                hidden = model(signal, return_hidden=True)
            self.assertEqual(tuple(prediction.shape), (2, output))
            self.assertEqual(tuple(hidden.shape), (2, 3030))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertEqual(model.features_batchnorm.num_features, 3030)

    def test_word_head_geometry_is_27_classes_without_second_stride(self):
        model = Mel2WordHidden(3030, 27).eval()
        self.assertEqual(model.MEL_PRE_DONSAMPLING, 1)
        self.assertEqual(model.mels2features[0].kernel_size, (3030, 10))
        self.assertEqual(model.max_pool.kernel_size, 10)
        with torch.no_grad():
            prediction = model(torch.zeros(2, 3030, 52))
        self.assertEqual(tuple(prediction.shape), (2, 27))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_release_exposes_only_mel40_and_base_six_layers(self):
        for layout in ("8_16", "6_12"):
            prefix = f"SimpleNetBase_WithLSTM__CNANNELS_{layout}__LAG_1000_0__"
            mel = getattr(bmr, prefix + "40MELS")
            self.assertEqual(mel.OUTPUT_SIZE, 40)
            for layer in range(1, 7):
                target = getattr(bmr, prefix + f"WHISPER_BASE_L{layer}")
                self.assertEqual(target.WHISPER_MODEL, "openai/whisper-base")
                self.assertEqual((target.WHISPER_LAYER, target.OUTPUT_SIZE), (layer, 50))
        self.assertFalse(any("LEAD" in n or "TINY" in n or "HUBERT" in n or "LPCS" in n for n in vars(bmr)))

    def test_word_frame_boundaries_and_intervening_silence(self):
        x = np.arange(12 * 2).reshape(12, 2)
        frames, labels = rc.prepare_frames(x, [(2, 4, "женя"), (7, 9, "широко")], 1)
        self.assertEqual(labels, [common.WORDS_REMAP["женя"], 0, common.WORDS_REMAP["широко"]])
        np.testing.assert_array_equal(frames[0], x[2:4])
        np.testing.assert_array_equal(frames[1], x[4:7])
        np.testing.assert_array_equal(frames[2], x[7:9])

    def test_crop_and_right_zero_pad_preserve_input(self):
        original = [np.arange(12).reshape(6, 2), np.ones((2, 2))]
        before = [v.copy() for v in original]
        out = rc.prepare_x_batch_for_net(original, 4)
        self.assertEqual(out.shape, (2, 2, 4))
        np.testing.assert_array_equal(out[0], before[0][:4].T)
        np.testing.assert_array_equal(out[1, :, 2:], 0)
        for actual, expected in zip(original, before):
            np.testing.assert_array_equal(actual, expected)

    def test_historical_width_uses_all_duration_metadata(self):
        early = [np.zeros((20, 2)), np.zeros((25, 2))]
        heldout = [np.zeros((100, 2))]
        self.assertGreater(rc.calc_max_words_length([early, heldout]), rc.calc_max_words_length([early]))

    def test_silence_sampling_is_repeatable_and_retains_every_word(self):
        labels = np.array([1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0])
        samples = np.arange(len(labels))
        np.random.seed(42)
        first = rc.fix_class_imbalance(samples, labels)
        np.random.seed(42)
        second = rc.fix_class_imbalance(samples, labels)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(set(first[0][first[1] != 0]), set(samples[labels != 0]))

    def test_full_evaluation_visits_every_trial_once(self):
        class DummyHead(torch.nn.Module):
            def forward(self, inputs):
                result = torch.zeros(len(inputs), 27)
                result.scatter_(1, inputs[:, 0, :1].long(), 1)
                return result
        x = np.empty(7, dtype=object)
        labels = np.arange(7)
        for i in labels:
            x[i] = np.full((12, 1), i)
        true, predicted = rc.get_full_predictions(types.SimpleNamespace(model=DummyHead()), x, labels, 12, 3)
        np.testing.assert_array_equal(true, labels)
        np.testing.assert_array_equal(predicted, labels)

    def test_reducer_fit_is_unchanged_by_heldout_transform(self):
        rng = np.random.RandomState(4)
        train = rng.normal(size=(80, 8)).astype(np.float32)
        heldout = (1000 + rng.normal(size=(12, 8))).astype(np.float32)
        reducer = wt.fit_reducer(train, True, 4, seed=4)
        before = copy.deepcopy(reducer.named_steps["standardscaler"].mean_)
        wt.apply_reducer(reducer, heldout)
        np.testing.assert_array_equal(before, reducer.named_steps["standardscaler"].mean_)
        np.testing.assert_allclose(before, train.astype(np.float64).mean(axis=0))
        self.assertTrue(reducer.named_steps["pca"].whiten)

    def test_whisper_loader_freezes_teacher_and_pins_revision(self):
        calls = []
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.layers = [None] * 6
                self.config = types.SimpleNamespace(d_model=512)
        class Model:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append((name, kwargs))
                return types.SimpleNamespace(encoder=Encoder())
        class Extractor:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append((name, kwargs))
                return object()
        with patch.dict(sys.modules, {"transformers": types.SimpleNamespace(WhisperModel=Model, WhisperFeatureExtractor=Extractor)}), patch.dict(wt._MODELS, {}, clear=True):
            encoder, _ = wt._get_model(wt.WHISPER_MODEL_ID)
            self.assertFalse(encoder.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(v[1]["revision"] == wt.WHISPER_REVISION for v in calls))
        with self.assertRaises(ValueError):
            wt._get_model("openai/whisper-tiny")

    def test_hidden_feature_alignment_retains_exact_requested_length(self):
        values = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
        aligned = wt.align_to_ecog(values, 101)
        self.assertEqual(aligned.shape, (101, 4))
        self.assertEqual(aligned.dtype, np.float32)

    def test_io_contract_uses_readonly_hdf5_and_tensor_only_checkpoints(self):
        root = Path(__file__).resolve().parents[1] / "library"
        for name in ("runner_regression.py", "runner_classification.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "h5py" and node.func.attr == "File":
                        self.assertGreaterEqual(len(node.args), 2)
                        self.assertEqual(ast.literal_eval(node.args[1]), "r")
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                        self.assertIn("weights_only", {k.arg for k in node.keywords})
                        self.assertTrue(next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "weights_only"))


if __name__ == "__main__":
    unittest.main()
