# ECoG word decoding with Whisper-base targets

Compact research code for a two-stage, closed-vocabulary overt-speech decoder:

```text
patient audio -> frozen Whisper-base L1..L6 -> train-fitted PCA50 target
                                                ^ MSE
neural window -> CNN/BiLSTM -> hidden3030 -> linear50
                                 |
                          frozen encoder
                                 v
                      trained 27-class word head

fixed L3/L4/L5 probabilities -> equal mean -> predicted word
```

This publication copy contains the historical Whisper-base layer study and its
MEL40 control. It derives from the compact network in Petrosyan et al. (2022).
Whisper supplies the regression target; audio is used during training. The word
classifier reads the neural encoder's internal3030-dimensional representation.

The six layers are separate training targets. An L3/L4/L5 ensemble contains
three independently trained neural paths of the same seed. Five-seed results
describe five such triples; they are not a single 15-model ensemble.

## Scope and recorded results

The ordinary MEL replication after split corrections and implementation fixes
reached best saved single-run accuracies of **48.82% for P1 (sEEG)** and
**67.39% for P2 (ECoG)**. The MEL code is included through `train.py --target mel`.
See [MEL_BASELINE.md](MEL_BASELINE.md) for the recovered source, exact scores,
seeds and evaluation boundaries. These maxima are separate from the later
five-seed MEL mean below.

The early layer means were 56.56%, 70.54%, 72.28%, 73.79%, 72.82%, 72.56% for L1..L6.
The fixed L3/L4/L5 mean over five seeds was 78.07%; its best repeats reached
362/445 = 81.35%. See [HISTORICAL_RESULTS.md](HISTORICAL_RESULTS.md) for the full
numbers and the important distinction between sampled single-model accuracy
and full 445-event ensemble accuracy. These are preserved historical results,
not measurements produced by the packaging checks.

This repository is a cleaned working-source reconstruction, not a verified
byte-identical snapshot of the June 2026 training environment. The recording-level
split is preserved. Historical per-record normalization and pooled window-width
metadata are documented in [PROVENANCE.md](PROVENANCE.md); this recipe is an
offline historical protocol, not a new blind benchmark.

## Install

Python 3.10 is the locally verified interpreter. Create an environment and install
the small direct-dependency list:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt
```

Install a PyTorch CPU or CUDA build suitable for your machine if required.
The requirements record current local verification versions, not a recovered
June lockfile. Whisper-base weights are fetched from Hugging Face on the first
actual Whisper training run and are not included in this repository. The model
revision is pinned in `library/whisper_target.py`.

## Configure your authorized data

Copy `configs/patients.example.json` to `configs/patients.local.json` and edit the
recording paths. Relative paths resolve against the configuration file's folder.
The example names and recording paths are placeholders; no patient data or
original recording filenames are distributed. Follow the upstream repository
for dataset context and obtain the appropriate data access.

Each recording is HDF5 with a numeric `RawData/Samples` dataset shaped
`[samples,channels]`. Its sibling `record01_words.txt` contains tab-separated
`start_sample`, `end_sample`, `word`, with integer sample indices at the original
recording rate. The word list is the historical 26-word Russian vocabulary in
`library/runner_common.py`, plus the generated silence class.

The fixed layouts select positions 8:16 (`ecog8`) or 6:12 (`seeg6`) from the
configured neural-channel array. Therefore the example lists the full neural
channel population, with a separate audio channel. Supply the same ordered
layout; this entry point is dataset-specific, not an arbitrary-channel adapter.

With 12 ECoG recordings the historical regression split is train 0..6,
validation 7, test 8..11; the head split is train 0..8, validation 9, test 10..11.
Indices are zero-based. With 6 sEEG recordings they are respectively
train 0..2 / validation 3 / test 4..5 and train 0..3 / validation 4 / test 5.

## Run one layer

All output paths are isolated. Existing nonempty runs are refused; checkpoints
are never automatically deleted. The plan can be checked without opening raw
recordings, loading models or downloading weights:

```bash
python train.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --target whisper --layer 4 --seed 4 --run-dir runs/seed_4/L4 --dry-run
```

Execute the same plan, or train the MEL control:

```bash
python train.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --target whisper --layer 4 --seed 4 --run-dir runs/seed_4/L4
python train.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --target mel --seed 4 --run-dir runs/seed_4/mel
```

The default `--stage both` trains regression and then the frozen-encoder word
head. To split those stages, first use `--stage regression`, then use
`--stage hidden` with the identical configuration and run directory. The second
call verifies the prior completion identity and artifact hashes. `--device cpu`
or `--device cuda` can override automatic selection.

## Layer sweep and fixed ensemble

```bash
python multiseed.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --runs runs --include-mel --dry-run
python multiseed.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --runs runs --include-mel
python ensemble.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --runs runs --seed 4 --validate-only
python ensemble.py --patients configs/patients.local.json --patient ecog_example --layout ecog8 --runs runs --seed 4
```

The sweep defaults to layers 1..6 and seeds 42, 1, 2, 3, 4. Run `ensemble.py` for each
seed to obtain five separate fixed triples. It evaluates individual models and
their equal-probability ensemble on the same full event set; no subset or weight
selection is performed on the test. The standalone command performs ensemble
evaluation; training runners also retain their historical post-selection test
reports. Scientific training is not invoked by the synthetic tests.

Each completed training run writes `complete.json` with exact artifact SHA256,
configuration identity and relative paths. Actual package versions are recorded
in `environment.json`. Models, caches, TensorBoard logs and raw data are ignored
by git. The publication archive contains only source, examples, tests and
compact aggregate historical results.

## CPU checks

```bash
python -m unittest discover -s tests -v
```

Checks use generated arrays or mocks; they do not load patient recordings,
download Whisper, run research training or establish scientific reproducibility.
See [PROVENANCE.md](PROVENANCE.md) for the changes made while packaging.

## Credit

- Petrosyan et al. (2022), *Speech decoding from a small set of spatially
  segregated minimally invasive intracranial EEG electrodes with a compact and
  interpretable neural network*; [upstream code](https://github.com/pet67/ossadtchi-ml-test-bench-speech).
- OpenAI Whisper: [model and code](https://github.com/openai/whisper).
- This project adds frozen Whisper-base layer targets, train-fitted PCA,
  multi-seed comparison and fixed layer ensembling to the compact neural recipe.

See [NOTICE.md](NOTICE.md) for attribution and upstream license status.
