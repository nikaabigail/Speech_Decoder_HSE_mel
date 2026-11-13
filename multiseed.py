"""Run isolated Whisper-base layer experiments without deleting prior artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ensemble import ARTIFACT_KEYS, model_name, read_json, require, sha256, validated_artifact

SEEDS = (42, 1, 2, 3, 4)
LAYERS = (1, 2, 3, 4, 5, 6)
ROOT = Path(__file__).resolve().parent


def expected_model(layout, target, layer):
    if target == "whisper":
        return model_name(layout, layer)
    require(target == "mel", "Only the MEL control or Whisper-base is supported")
    channels = "8_16" if layout == "ecog8" else "6_12"
    return f"SimpleNetBase_WithLSTM__CNANNELS_{channels}__LAG_1000_0__40MELS"


def plan(runs, patients, patient, layout, seeds=SEEDS, include_mel=False):
    require(layout in ("ecog8", "seeg6"), "Unknown channel layout")
    require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int and s >= 0 for s in seeds),
            "Nonnegative unique integer seeds required")
    rows = []
    for seed in seeds:
        targets = [("whisper", layer) for layer in LAYERS]
        if include_mel:
            targets.append(("mel", None))
        for target, layer in targets:
            run_dir = Path(runs).expanduser().resolve() / f"seed_{seed}" / (f"L{layer}" if target == "whisper" else "mel")
            command = [sys.executable, str(ROOT / "train.py"), "--patients", str(Path(patients).expanduser().resolve()),
                       "--patient", patient, "--layout", layout, "--target", target,
                       "--seed", str(seed), "--run-dir", str(run_dir), "--stage", "both"]
            if target == "whisper":
                command.extend(["--layer", str(layer)])
            identity = dict(patient=patient, layout=layout, target=target, layer=layer, seed=seed,
                            model_name=expected_model(layout, target, layer))
            rows.append(dict(run_dir=run_dir, command=command, identity=identity))
    return rows


def check_reusable(run_dir, identity, patients):
    """Return False for a new run, True for verified complete artifacts; fail on partial runs."""
    path = Path(run_dir)
    if not path.exists():
        return False
    require(path.is_dir(), "Run path exists and is not a directory")
    if not any(path.iterdir()):
        return False
    complete_path = path / "complete.json"
    require(complete_path.is_file(), "Incomplete existing run; inspect it instead of overwriting")
    complete = read_json(complete_path)
    require(complete.get("schema") == "public_training_completion_v1" and
            complete.get("stage") in ("both", "hidden") and
            all(complete.get(key) == value for key, value in identity.items()),
            "Existing run identity/stage differs")
    require(complete.get("patients_sha256") == sha256(patients), "Patient configuration changed")
    pins = complete.get("artifact_sha256", {})
    for key in ARTIFACT_KEYS:
        relative = complete.get(key)
        require(isinstance(relative, str) and relative in pins, "Incomplete artifact pins")
        validated_artifact(path, relative, pins[relative])
        if key.endswith("report"):
            config = read_json(path / relative).get("config", {})
            require(config.get("seed") == identity["seed"] and
                    config.get("patient") == identity["patient"], "Report identity mismatch")
            if key == "classification_report":
                require(config.get("regression_model") == identity["model_name"],
                        "Word-head report belongs to another regression model")
    return True


def execute(rows, patients, dry_run=False):
    report = []
    # Check every planned run before starting any training subprocess.
    for row in rows:
        reuse = check_reusable(row["run_dir"], row["identity"], patients)
        action = "reuse_verified" if reuse else "train"
        report.append(dict(identity=row["identity"], action=action, command=row["command"]))
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    for row, status in zip(rows, report):
        if not dry_run and status["action"] == "train":
            subprocess.run(row["command"], check=True)
            require(check_reusable(row["run_dir"], row["identity"], patients),
                    "Training exited without verified completion")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--layout", required=True, choices=("ecog8", "seeg6"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--include-mel", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; never start a subprocess")
    args = parser.parse_args(argv)
    patients = args.patients.expanduser().resolve(strict=True)
    from ensemble import load_patient
    load_patient(patients, args.patient, args.layout)  # No scientific files opened.
    rows = plan(args.runs, patients, args.patient, args.layout, args.seeds, args.include_mel)
    execute(rows, patients, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
