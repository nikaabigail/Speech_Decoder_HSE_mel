# Historical results and evaluation boundaries

## Plain MEL replication: best saved individual runs

After the recording-split corrections and implementation fixes, ordinary MEL
hidden-head decoding reached **48.82% for P1** (seed42, June13,2026) and
**67.39% for P2** (seed3, June20,2026). These are individual-run maxima of the
plain-MEL recipe; they are not five-seed means or maxima over augmentation and
multitask experiments. The P2 number was retained as a MEL control in the
MEL/Whisper comparison. Both reports store the historical sampled
`test_accuracy`, without a saved `test_accuracy_full`.

See [MEL_BASELINE.md](MEL_BASELINE.md) and `benchmarks/mel_baseline_runs.json`
for implementation pointers, exact values, splits and source-report hashes.

## Whisper layers and the matched MEL multi-seed control

The following numbers were recovered from saved June 20–26, 2026 result JSONs
for participant P2, eight-channel ECoG, Whisper-base, seeds 42, 1, 2, 3, 4.
No new scientific experiment was run to prepare this repository.

| Target | Mean accuracy | Sample SD |
|---|---:|---:|
| MEL40 | 0.6121477046 | 0.0607257614 |
| Whisper L1 | 0.5655848303 | 0.0938216662 |
| Whisper L2 | 0.7053532934 | 0.0369249107 |
| Whisper L3 | 0.7228103792 | 0.0302203334 |
| Whisper L4 | 0.7379441118 | 0.0299757838 |
| Whisper L5 | 0.7282475050 | 0.0233071580 |
| Whisper L6 | 0.7256207585 | 0.0382497568 |

These single-target means use the historical `test_accuracy`: random test
minibatches sampled with replacement. The stored `test_accuracy_full` is a
different, deterministic complete traversal. Do not silently replace one with
the other when reproducing the early layer plot.

## Fixed L3/L4/L5 triples

Each seed has three separate neural encoders and word heads, one per layer.
Probabilities are averaged with equal weights within that seed.

| Seed | Correct / total | Full-set accuracy |
|---|---:|---:|
| 1 | 347/445 | 0.7797752809 |
| 2 | 362/445 | 0.8134831461 |
| 3 | 334/445 | 0.7505617978 |
| 42 | 332/445 | 0.7460674157 |
| 4 | 362/445 | 0.8134831461 |
| Mean | — | **0.7806741573** |
| Sample SD | — | 0.0326268610 |

The 78.07% point is the mean of five triples, not the accuracy of a combined
15-model ensemble. The 81.35% result is an individual triple's full-set result.
The five seeds reuse the same recordings; they are not five independent
participants or independent test datasets.

The single-layer curve and ensemble point use different historical evaluation
estimators. The old plot's improvement and p-value annotation should therefore
not be presented as a matched comparison against the plotted L4 mean. The new
`ensemble.py` reports all three single-model results and the fixed triple on the
same event set. It does not select a best subset or weights on the test.

Petrosyan et al.'s published P2 result around 70% comes from their own protocol;
it is a contextual reference, not the matched control for a causal gain claim.
The MEL40 control above is the within-project historical comparator. Full
provenance and protocol qualifications are in [PROVENANCE.md](PROVENANCE.md).

Compact machine-readable values are in `benchmarks/historical_results.json`.
