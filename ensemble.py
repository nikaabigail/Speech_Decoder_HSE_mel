"""Evaluate the declared, equal-probability Whisper L3/L4/L5 ensemble.

Model/data imports are lazy. There is no subset search, seed mixing, training,
checkpoint discovery by modification time, or fallback to a missing layer.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

LAYERS = (3, 4, 5)
ARTIFACT_KEYS = ("regression_checkpoint", "classification_checkpoint",
                 "regression_report", "classification_report")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_name(layout, layer):
    require(layout in ("ecog8", "seeg6") and type(layer) is int and 1 <= layer <= 6,
            "Expected the ecog8/seeg6 layout and a base layer from1 through6")
    channels = "8_16" if layout == "ecog8" else "6_12"
    return f"SimpleNetBase_WithLSTM__CNANNELS_{channels}__LAG_1000_0__WHISPER_BASE_L{layer}"


def load_patient(path, name, layout="ecog8"):
    # One shared stdlib-only validator for training, planning and evaluation.
    from train import load_patient as validate_patient
    return validate_patient(path, name, layout)


def validated_artifact(run_dir, relative, expected_sha256):
    root = Path(run_dir).resolve(strict=True)
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "Artifact paths must be relative to their isolated run")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Artifact escapes its run directory")
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64 and
            sha256(path) == expected_sha256, "Artifact checksum mismatch")
    return path


def validate_complete(run_dir, expected, patients_path):
    """Verify a completed pair without loading arrays, models or raw data."""
    run_dir = Path(run_dir).resolve(strict=True)
    complete = read_json(run_dir / "complete.json")
    require(complete.get("schema") == "public_training_completion_v1",
            "Unsupported completion schema")
    require(all(complete.get(key) == value for key, value in expected.items()),
            "Completed run identity differs from the requested experiment")
    require(complete.get("stage") in ("both", "hidden"), "A completed word-head is required")
    require(complete.get("patients_sha256") == sha256(patients_path),
            "Dataset configuration differs from the completed run")
    pins = complete.get("artifact_sha256", {})
    artifacts = {}
    for key in ARTIFACT_KEYS:
        relative = complete.get(key)
        require(isinstance(relative, str) and relative in pins, "Missing artifact hash: " + key)
        artifacts[key] = validated_artifact(run_dir, relative, pins[relative])
    for key in ("regression_report", "classification_report"):
        config = read_json(artifacts[key]).get("config", {})
        require(config.get("seed") == expected["seed"] and
                config.get("patient") == expected["patient"], "Report seed/patient mismatch")
        if key == "classification_report":
            require(config.get("regression_model") == expected["model_name"],
                    "Word-head report refers to another regression model")
    return complete, artifacts


def softmax(logits):
    import numpy as np
    values = np.asarray(logits, dtype=np.float64)
    require(values.ndim == 2 and len(values) > 0 and np.isfinite(values).all(),
            "Finite, nonempty [events,classes] logits required")
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def fixed_ensemble(rows):
    """Average exactly three aligned probability matrices, with equal weights."""
    import numpy as np
    require(len(rows) == 3 and tuple(v["layer"] for v in rows) == LAYERS,
            "Only the predeclared, ordered L3/L4/L5 ensemble is supported")
    reference = np.asarray(rows[0]["classes"])
    events = list(rows[0]["event_ids"])
    require(reference.ndim == 1 and len(reference) > 0 and len(events) == len(reference),
            "A nonempty event panel is required")
    require(np.issubdtype(reference.dtype, np.integer) and
            bool(np.all((reference >= 0) & (reference < 27))), "Invalid class labels")
    probabilities = []
    for row in rows:
        values = np.asarray(row["probs"], dtype=np.float64)
        require(np.array_equal(row["classes"], reference) and list(row["event_ids"]) == events,
                "Event identity/order or labels differ between layers")
        require(values.shape == (len(reference), 27) and np.isfinite(values).all() and
                np.all(values >= 0) and np.allclose(values.sum(axis=1), 1.0, atol=1e-6),
                "Expected normalized27-class probabilities on the same panel")
        probabilities.append(values)
    averaged = np.mean(probabilities, axis=0)
    correct = int(np.sum(averaged.argmax(axis=1) == reference))
    return dict(correct=correct, count=len(reference), accuracy=correct / len(reference),
                probabilities=averaged, classes=reference, event_ids=events)


def balanced_indices(classes, rng):
    """Historical silence sampling, including replacement and sorted duplicates."""
    import numpy as np
    labels = np.asarray(classes, dtype=np.int64)
    require(labels.ndim == 1 and len(labels) > 0, "Nonempty class labels required")
    words = np.where(labels != 0)[0]
    silence = np.where(labels == 0)[0]
    selected_silence = rng.choice(silence, int(len(words) / len(set(labels))))
    return np.sort(np.concatenate([selected_silence, words])).astype(np.int64)


def classes_from_words(words_info, word_map):
    classes = []
    previous_end = 0
    for start, end, word in words_info:
        if previous_end != 0 and previous_end != start:
            classes.append(word_map["silent"])
        classes.append(word_map[word])
        previous_end = end
    return classes


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def per_layer_test_probs(patient, layer, seed, artifacts):
    """Historical frozen-head replay on all selected test events, once each.

    Test-file z-scoring and silence sampling are retained historical choices;
    this is offline synchronous evaluation with known word boundaries.
    """
    import numpy as np
    import torch
    import h5py
    import sklearn.preprocessing
    from library import bench_models_regression as bmr
    from library import bench_models_classification as bmc
    from library.runtime import DEVICE, make_split, set_seed
    from library.runner_common import WORDS_REMAP
    from library.runner_classification import (
        predict_regression_hidden, prepare_frames, load_words_info,
        get_words_filepath, prepare_x_batch_for_net, HIDDEN_STRIDE,
    )

    config = read_json(artifacts["classification_report"])["config"]
    width = config.get("max_words_length")
    require(type(width) is int and width > 0, "Saved classifier width is required; no test fitting")
    mname = config.get("regression_model")
    require(isinstance(mname, str) and mname.endswith(f"WHISPER_BASE_L{layer}"),
            "Classifier report targets another layer")
    set_seed(seed)
    bench = getattr(bmr, mname)(patient)
    classifier = bmc.Mel2WordHidden(bench.model.final_out_features, patient, mname)
    models = (bench.model, classifier.model)
    try:
        for model, key in zip(models, ("regression_checkpoint", "classification_checkpoint")):
            model.load_state_dict(torch.load(artifacts[key], map_location=DEVICE, weights_only=True), strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        _, _, test_indices = make_split(len(patient["files_list"]),
                                        patient["test_start_file_classification_index"])
        require(config.get("split", {}).get("test_files") == test_indices,
                "Saved classifier split differs from the configured test")
        test_set = set(test_indices)
        rng = np.random.RandomState(seed)
        all_classes, all_logits, all_events = [], [], []
        for file_index, filepath in enumerate(patient["files_list"]):
            words = load_words_info(get_words_filepath(filepath))
            classes = np.asarray(classes_from_words(words, WORDS_REMAP), dtype=np.int64)
            selected = balanced_indices(classes, rng)
            # Consume exactly the same per-file RNG draws before test files.
            if file_index not in test_set:
                continue
            with h5py.File(filepath, "r") as handle:
                data = handle["RawData"]["Samples"][()]
            ecog = data[:, patient["ecog_channels"]].astype("double")
            ecog = bench.preprocess_ecog(ecog, patient["sampling_rate"]).astype("float32")
            hidden = predict_regression_hidden(bench, ecog, HIDDEN_STRIDE)
            hidden = sklearn.preprocessing.scale(hidden, copy=False)
            frames, frame_classes = prepare_frames(hidden, words, bench.downsampling_coef * HIDDEN_STRIDE)
            require(np.array_equal(frame_classes, classes), "Frame/annotation event order differs")
            ragged = np.empty(len(frames), dtype=object)
            for i, frame in enumerate(frames):
                ragged[i] = frame
            xb = prepare_x_batch_for_net(ragged[selected], width)
            with torch.inference_mode():
                logits = classifier.model(torch.as_tensor(xb, dtype=torch.float32, device=DEVICE))
            all_logits.extend(logits.cpu().numpy())
            all_classes.extend(classes[selected])
            all_events.extend(f"{file_index}:{int(index)}" for index in selected)
        probabilities = softmax(all_logits)
        labels = np.asarray(all_classes, dtype=np.int64)
        return dict(layer=layer, seed=seed, classes=labels, probs=probabilities,
                    event_ids=all_events, accuracy=float(np.mean(probabilities.argmax(1) == labels)))
    finally:
        for wrapper in (bench, classifier):
            writer = getattr(getattr(wrapper, "logger", None), "tensorboard_writer", None)
            if writer is not None:
                writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--layout", required=True, choices=("ecog8", "seeg6"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Verify completed artifacts without data/model inference")
    args = parser.parse_args(argv)
    root = args.runs.expanduser().resolve(strict=True)
    patients_path = args.patients.expanduser().resolve(strict=True)
    patient = load_patient(patients_path, args.patient, args.layout)
    inputs = []
    for layer in LAYERS:
        expected = dict(patient=args.patient, layout=args.layout, target="whisper", layer=layer,
                        seed=args.seed, model_name=model_name(args.layout, layer))
        run = root / f"seed_{args.seed}" / f"L{layer}"
        complete, artifacts = validate_complete(run, expected, patients_path)
        inputs.append((layer, complete, artifacts))
    if args.validate_only:
        print(json.dumps(dict(passed=True, layers=list(LAYERS), seed=args.seed, inference_performed=False)))
        return 0
    destination = root / "ensembles" / f"seed_{args.seed}.json"
    require(not destination.exists(), "Ensemble output already exists; preserve it")
    os.environ["BENCH_SEED"] = str(args.seed)
    with working_directory(root / "ensembles" / f"work_seed_{args.seed}"):
        rows = [per_layer_test_probs(patient, layer, args.seed, artifacts)
                for layer, _, artifacts in inputs]
    result = fixed_ensemble(rows)
    output = dict(schema="fixed_l345_evaluation_v1", created_at=datetime.now(timezone.utc).isoformat(),
                  patient=args.patient, layout=args.layout, seed=args.seed, layers=list(LAYERS),
                  weights=[1/3] * 3, selection="declared fixed L3/L4/L5; no test subset selection",
                  metric="deterministic full pass over historically balanced test events",
                  patients_sha256=sha256(patients_path), correct=result["correct"],
                  count=result["count"], accuracy=result["accuracy"],
                  single_layers=[dict(layer=row["layer"], accuracy=row["accuracy"]) for row in rows],
                  sources=[dict(layer=layer, artifact_sha256=complete["artifact_sha256"])
                           for layer, complete, _ in inputs])
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
