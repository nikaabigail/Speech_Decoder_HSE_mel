# Provenance and packaging changes

## Scientific origin

The compact CNN/BiLSTM and two-stage word-decoding recipe derive from
[pet67/ossadtchi-ml-test-bench-speech](https://github.com/pet67/ossadtchi-ml-test-bench-speech)
and Petrosyan et al. (2022). This extension replaces the MEL regression target
with frozen Whisper-base encoder states, uses train-fitted scaler/PCA50, and
compares layers and equal-probability ensembles. Architecture geometry and the
retained loss, optimizer, sampling and checkpoint-selection paths come from
the recovered historical working source.

The source directory had no Git history or execution-time source seal for the
June experiments. Consequently this is a cleaned reconstruction of the working
implementation, not a certified byte-identical June source/environment release.
Saved metrics retain their dates and estimator definitions. Dependency versions
are the versions used to check the publication copy, not an original lockfile.

The earlier plain-MEL runs were subsequently traced to their saved regression
and hidden-head reports: P1/seed42, June13,2026 (48.8243513%) and P2/seed3,
June20,2026 (67.3912176%). The P2 report is additionally preserved in recovered
local Git snapshot `04b2874b39aace9ebadb668feaf2ae4a173db802` from June21,2026.
The matching MEL computation path is already retained by `train.py --target mel`;
this provenance addition does not introduce another model or change training.
The earliest P1 runtime did not set all later deterministic CUDA flags, so the
source match is not a promise of a bit-identical P1 training rerun.

## Historical protocol retained explicitly

- The neural regression window has 1001 samples on the downsampled time grid,
  with lag-backward 1000 and no lag-forward. At 1024 Hz, endpoint separation is
  976.6 ms. Word-head examples cover annotated word intervals; the effective
  neural receptive field includes the preceding regression history.
- Neural preprocessing and hidden-feature z-scoring operate per recording.
  This uses unlabelled statistics from the whole evaluation recording. It is
  an offline/transductive normalization convention, not a strictly causal
  stream or exclusively train-statistics transform.
- Audio uses historical max-absolute normalization without automatic DC
  centering, 30-second Whisper context, and the original alignment rule.
- The scaler/PCA target reducer is fitted only on regression-training frames.
- Regression-training records are concatenated before sampling windows; a
  window near an internal join can span two training recordings. That
  historical convention was preserved rather than changed during packaging.
- Regression checkpoints follow the historical rolling validation correlation
  rule; word heads follow rolling validation accuracy. Neither was replaced
  with full-validation selection during packaging.
- Word-head width is the historical 95th percentile calculated over prepared
  segment lengths from all recording partitions. This includes held-out
  duration metadata. It must be disclosed; this release does not describe
  that historical metadata convention as leakage-free strict evaluation.
- Silence is subsampled using the historical seeded balancing procedure.
  Sampled `test_accuracy` and deterministic `test_accuracy_full` remain
  separately named. They must not be conflated with each other or ASR WER.
- No new train-only-width, DC-corrected, POWER, Transformer, asynchronous or
  external-dataset variant is introduced in this publication copy.

## Before → after

| Area | Recovered working source | Publication copy |
|---|---|---|
| Experiment families | Numerous unrelated targets, controls and pilots | MEL40 and Whisper-base L1..L6, regression and hidden-head stages |
| Entrypoint/config | Workstation paths and working-directory assumptions | Explicit private JSON; paths relative to that JSON; dedicated run directories |
| Multi-seed runs | Could delete old checkpoints before another seed | Separate seed/target directories; exact identity/hash checks before reuse |
| Ensemble report | Exploratory subset scans were available | Fixed L3/L4/L5, equal probabilities, same event IDs, no test-based search |
| HDF5/checkpoint I/O | Writable HDF5 handles; generic weight loading | Read-only HDF5; tensor-only checkpoint loading |
| Artifacts | Large local logs, caches, weights and raw data | Source, tests, metadata-only examples and compact historical aggregate values |
| Dependencies | Full workstation package freeze | Direct dependencies needed by the retained recipe |

Removed experimental branches are not described as failed or successful new
experiments; they are outside this repository's scope. Private source copies,
exact before/after diffs, original hashes and packaging QA are retained outside
the publishable tree. The publication copy does not modify the working research
directories, active cluster jobs or saved scientific inputs.

## Verification boundary

CPU tests use synthetic arrays, temporary fixtures and mocked orchestration.
They check interfaces, configuration, state geometry, metric arithmetic, artifact
identity and packaging. Help and dry-run paths open no recordings and download
no models. No research training or real-patient inference was run to create
this release, and its historical accuracies were not re-established by tests.
Real reproduction requires the authorized recordings, annotations, an agreed
environment and explicit scientific execution.
