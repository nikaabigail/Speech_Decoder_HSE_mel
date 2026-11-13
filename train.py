"""Portable single-run entry point for the historical MEL/Whisper-base study.

Scientific runners are imported only after configuration, seed and output
isolation have been validated. Importing this module or --help opens no data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def load_patient(path, name, layout):
    """Resolve user-owned data paths relative to the JSON, without opening HDF5."""
    path = Path(path).expanduser().resolve(strict=True)
    rows = read_json(path)
    require(isinstance(rows, list), 'Patient configuration must be a JSON list')
    selected = [row for row in rows if row.get('name') == name]
    require(len(selected) == 1, 'Select exactly one uniquely named patient')
    require(re.fullmatch(r'[A-Za-z0-9_-]+', name) and '___' not in name,
            'Patient name must be a safe filename component without triple underscores')
    patient = dict(selected[0])
    files = patient.get('files_list')
    require(isinstance(files, list) and len(files) >= 3 and all(isinstance(f, str) for f in files),
            'At least three ordered recording paths are required')
    resolved = []
    for value in files:
        require(value.strip() and 'DATA_ROOT' not in value and '${' not in value,
                'Replace example data paths in your private configuration')
        source = Path(value).expanduser()
        resolved.append(str((source if source.is_absolute() else path.parent / source).resolve()))
    require(len(set(resolved)) == len(resolved), 'Recording paths must be distinct')
    patient['files_list'] = resolved
    frequency = patient.get('sampling_rate')
    require(type(frequency) is int and frequency > 0, 'A positive integer sampling rate is required')
    channels = patient.get('ecog_channels')
    sound = patient.get('sound_channel')
    require(isinstance(channels, list) and len(channels) >= (16 if layout == 'ecog8' else 12) and
            all(type(c) is int and c >= 0 for c in channels) and len(set(channels)) == len(channels),
            'Provide the ordered neural-channel list expected by the selected layout')
    require(type(sound) is int and sound >= 0 and sound not in channels,
            'The audio channel must be separate from the neural channels')
    reg_start = patient.get('test_start_file_regression_index')
    head_start = patient.get('test_start_file_classification_index')
    require(type(reg_start) is int and type(head_start) is int and
            2 <= reg_start <= head_start < len(files),
            'Each stage needs distinct training, validation and test recording sets')
    return patient


def model_name(layout, target, layer):
    require(layout in ('ecog8', 'seeg6') and target in ('whisper', 'mel'), 'Unsupported model family')
    require(type(layer) is int and 1 <= layer <= 6, 'Whisper-base layers are1..6')
    channels = '8_16' if layout == 'ecog8' else '6_12'
    suffix = f'WHISPER_BASE_L{layer}' if target == 'whisper' else '40MELS'
    return f'SimpleNetBase_WithLSTM__CNANNELS_{channels}__LAG_1000_0__{suffix}'


def validate_report(path, identity, mode):
    config = read_json(path)['config']
    require(config['patient'] == identity['patient'] and config['seed'] == identity['seed'] and
            config['mode'] == mode, 'Saved report identity does not match this run')
    require(config.get('model', config.get('regression_model')) == identity['model_name'],
            'Saved report belongs to a different regression model')


def one_artifact(run_dir, folder, pattern):
    matches = sorted((run_dir / folder).glob(pattern))
    require(len(matches) == 1, f'Expected one unambiguous artifact: {folder}/{pattern}')
    return matches[0]


def validate_prior_regression(run_dir, identity, patients_hash):
    previous = read_json(run_dir / 'complete.json')
    require(previous.get('schema') == 'public_training_completion_v1' and
            previous.get('stage') == 'regression' and previous.get('patients_sha256') == patients_hash,
            'Hidden-only requires a completed matching regression run')
    require(all(previous.get(k) == v for k, v in identity.items()), 'Regression identity changed')
    for key in ('regression_checkpoint', 'regression_report'):
        relative = previous[key]
        artifact = (run_dir / relative).resolve()
        require(not Path(relative).is_absolute() and artifact.is_relative_to(run_dir) and
                sha256(artifact) == previous['artifact_sha256'][relative],
                'Previous regression artifact changed or escaped the run directory')
    validate_report(run_dir / previous['regression_report'], identity, 'regression')
    return previous


@contextmanager
def reserve_run(run_dir, stage, identity, patients_hash):
    """Serialize writers and refuse a partial head attempt before opening data."""
    if stage != 'hidden':
        require(not run_dir.exists() or not any(run_dir.iterdir()),
                'Run directory is nonempty; existing checkpoints are never deleted or overwritten')
    else:
        require(run_dir.is_dir(), 'Hidden-only requires an existing completed regression directory')
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / '.run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        previous = None
        if stage == 'hidden':
            previous = validate_prior_regression(run_dir, identity, patients_hash)
            require(not (run_dir / 'failed.json').exists() and
                    not (run_dir / 'regression_complete.json').exists() and
                    not any((run_dir / 'model_dumps').glob('classification_hidden*')) and
                    not any((run_dir / 'results').glob('classification_hidden*')),
                    'Partial or completed hidden attempt exists; implicit overwrite/resume is refused')
        else:
            require(all(p.name == '.run.lock' for p in run_dir.iterdir()),
                    'Run directory changed while acquiring the exclusive lock')
        yield previous
    finally:
        lock.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients', type=Path, required=True, help='Private JSON, based on configs/patients.example.json')
    parser.add_argument('--patient', required=True)
    parser.add_argument('--layout', choices=['ecog8', 'seeg6'], default='ecog8')
    parser.add_argument('--target', choices=['whisper', 'mel'], default='whisper')
    parser.add_argument('--layer', type=int, choices=range(1, 7), default=4)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['regression', 'hidden', 'both'], default='both')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Validate config and print the plan; no model, data or training I/O')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.seed >= 0, 'Seed must be nonnegative')
    patients_path = args.patients.expanduser().resolve(strict=True)
    patient = load_patient(patients_path, args.patient, args.layout)
    name = model_name(args.layout, args.target, args.layer)
    run_dir = args.run_dir.expanduser().resolve()
    require(run_dir != ROOT and not ROOT.is_relative_to(run_dir),
            'Use a dedicated run directory, not the repository or its ancestor')
    identity = dict(patient=args.patient, layout=args.layout, target=args.target,
                    layer=args.layer if args.target == 'whisper' else None, seed=args.seed,
                    model_name=name)
    patients_hash = sha256(patients_path)
    plan = dict(**identity, stage=args.stage, run_dir=str(run_dir),
                recording_count=len(patient['files_list']), device=args.device,
                patients_sha256=patients_hash)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    with reserve_run(run_dir, args.stage, identity, patients_hash) as previous:
        return execute_run(args, patient, run_dir, identity, patients_path, patients_hash,
                           plan, previous)


def execute_run(args, patient, run_dir, identity, patients_path, patients_hash, plan, previous):
    name = identity['model_name']
    for filepath in patient['files_list']:
        require(Path(filepath).is_file(), 'Recording file missing: ' + filepath)
        if args.stage != 'regression':
            annotation = str(Path(filepath).with_suffix('')) + '_words.txt'
            require(Path(annotation).is_file(), 'Word annotation missing: ' + annotation)
    os.environ['BENCH_SEED'] = str(args.seed)
    os.environ['ECOG_DEVICE'] = args.device
    run_dir.mkdir(parents=True, exist_ok=True)
    for folder in ('results', 'model_dumps'):
        (run_dir / folder).mkdir(exist_ok=True)
    old_cwd = Path.cwd()
    started = datetime.now(timezone.utc).isoformat()
    try:
        os.chdir(run_dir)
        write_json(run_dir / 'run_config.json', dict(**plan, started_at=started))
        from library.runner_regression import run_regression
        from library.runner_classification import run_hidden
        if args.stage in ('regression', 'both'):
            run_regression(name, patient, runs_count=1, is_debug=False)
        if args.stage in ('hidden', 'both'):
            run_hidden(name, patient, is_debug=False)
        prefix = f"{args.patient}___{name}___"
        artifacts = dict(
            regression_checkpoint=one_artifact(run_dir, 'model_dumps', f'regression___{prefix}*.pth'),
            regression_report=one_artifact(run_dir, 'results', f'regression___{prefix}*.json'))
        validate_report(artifacts['regression_report'], identity, 'regression')
        if args.stage in ('hidden', 'both'):
            artifacts.update(
                classification_checkpoint=one_artifact(run_dir, 'model_dumps', f'classification_hidden___{prefix}*.pth'),
                classification_report=one_artifact(run_dir, 'results', f'classification_hidden___{prefix}*.json'))
            validate_report(artifacts['classification_report'], identity, 'classification_hidden')
        require(sha256(patients_path) == patients_hash, 'Patient configuration changed during training')
        completion = dict(schema='public_training_completion_v1', **identity, stage=args.stage,
                          patients_sha256=patients_hash, started_at=started,
                          completed_at=datetime.now(timezone.utc).isoformat())
        completion.update({key: value.relative_to(run_dir).as_posix() for key, value in artifacts.items()})
        completion['artifact_sha256'] = {value.relative_to(run_dir).as_posix(): sha256(value)
                                         for value in artifacts.values()}
        if previous is not None:
            write_json(run_dir / 'regression_complete.json', previous)
        write_json(run_dir / 'complete.json', completion)
        packages = {}
        for package in ('torch', 'numpy', 'scipy', 'scikit-learn', 'h5py', 'librosa', 'transformers', 'tensorboard'):
            packages[package] = importlib.metadata.version(package)
        write_json(run_dir / 'environment.json', dict(python=sys.version, packages=packages))
        print(json.dumps(completion, indent=2))
        return 0
    except Exception as exc:
        write_json(run_dir / 'failed.json', dict(**identity, error=str(exc),
                   at=datetime.now(timezone.utc).isoformat()))
        traceback.print_exc()
        raise
    finally:
        os.chdir(old_cwd)


if __name__ == '__main__':
    raise SystemExit(main())
