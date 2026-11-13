# MEL baseline before replacing the acoustic target with Whisper

After correcting the recording split and individual implementation issues,
the best saved ordinary MEL runs reached **48.82% for P1 (sEEG, Procenko)** and
**67.39% for P2 (ECoG, Ivanova)**.

| Participant | Saved test accuracy | Seed | Run date | Word-head width |
|---|---:|---:|---|---:|
| P1, six sEEG channels | **48.8243513%** | 42 | 2026-06-13 | 81 frames |
| P2, eight ECoG channels | **67.3912176%** | 3 | 2026-06-20 | 52 frames |

These are best saved individual plain-MEL runs, not five-seed means or maxima
over all experimental methods. They use `classification_hidden`, with no
augmentation or multitask objective. The P2 result was preserved as the MEL
control in the June MEL/Whisper comparison; its date is not the date of the
earliest June13 replication. Run dates refer to training outputs, not patient
recording dates. The P2 five-seed MEL mean, 61.21%, is reported separately in
[HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md).

## The implementation

```text
audio -> 40 log-mel bins ---------------------- MSE target
                                                ^
neural window -> compact CNN/BiLSTM -> hidden3030 -> linear40
                                       |
                                frozen encoder
                                       v
                              27-class word head
```

The acoustic target is MEL; Whisper is not used in this route. The classifier
reads the hidden representation rather than the reconstructed MEL bins.
The saved MEL runs use the following corrected recording-level split,
with validation separated from test:

| Stage | P1 training / validation / test | P2 training / validation / test |
|---|---|---|
| MEL regression | 0..2 / 3 / 4..5 | 0..6 / 7 / 8..11 |
| Hidden word head | 0..3 / 4 / 5 | 0..8 / 9 / 10..11 |

The retained implementation already contains this computation family:

- `library/bench_models_regression.py`: the two `40MELS` configurations;
- `library/common_preprocessing.py`: ECoG and MEL preprocessing;
- `library/models_regression.py`: compact neural encoder;
- `library/runner_regression.py`: stage-one MEL regression;
- `library/runner_classification.py`: `run_hidden`, segment preparation and head;
- `library/runtime.py`: recording-level `make_split`;
- `train.py --target mel`: the portable entry point.

No second, diverging copy of those modules is added. Static comparison of the
retained definitions against the recovered MEL source confirmed the network,
MEL preprocessing, layouts and segment/split helpers. The P2 saved report is
also present byte-for-byte in the recovered source snapshot
`04b2874b39aace9ebadb668feaf2ae4a173db802`, dated June21,2026. This is a local
provenance identifier, not a claim that it is a commit in this public repository.

## Commands for the corresponding architecture

After supplying your authorized data in the private configuration:

```bash
python train.py --patients configs/patients.local.json --patient seeg_example --layout seeg6 --target mel --seed 42 --run-dir runs/mel_baseline_P1
python train.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --target mel --seed 3 --run-dir runs/mel_baseline_P2
```

These commands select the recovered MEL architecture and split convention.
They are not a promise of byte-identical historical scores: original dependency
environments are not fully recovered, and the unified runtime includes later
deterministic CUDA settings absent from the earliest P1 source. Historical
width selection from all-partition metadata and per-record normalization remain
as documented in [PROVENANCE.md](PROVENANCE.md).

## Metric and source evidence

Both original reports store sampled `test_accuracy`, obtained using historical
random test minibatches with replacement. Neither contains a separately saved
`test_accuracy_full`. Therefore 48.82% and 67.39% should not be described as new
deterministic full-pass results, combined with the 445-event ensemble as one
matched ablation, or subtracted from published paper results as a controlled gain.

`benchmarks/mel_baseline_runs.json` preserves exact scores, seed, stage splits,
width, selected iteration and SHA256 of both regression and head reports.
Large training logs, patient data, private paths and weights are not included.
This documentation update did not run new training or patient inference.
