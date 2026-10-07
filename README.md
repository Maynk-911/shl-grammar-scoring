# SHL Hiring Assessment 2026 — Grammar Scoring Engine

A reproducible, CPU-based regression pipeline predicting 0–5 grammar scores from WAV audio.
The executed **[SHL_Grammar_Scoring.ipynb](SHL_Grammar_Scoring.ipynb)** is the primary deliverable;
it contains the complete implementation, report, training/validation metrics and plots.

## Measured results

Selected model: **ExtraTreesRegressor**, 300 trees, minimum leaf size 3, 80% feature subsampling.
Five predeclared models were compared on three fixed development folds. Highest pooled CV Pearson
selected the winner; RMSE was the tie-break. A separate 20% holdout was excluded from selection.

| Evaluation | Samples | RMSE | Pearson |
|---|---:|---:|---:|
| Mean baseline, final holdout | 154 | 1.2488 | Undefined (constant) |
| Ridge, development out-of-fold | 615 | 0.8393 | 0.7342 |
| ExtraTrees, development out-of-fold | 615 | 0.7647 | 0.7886 |
| **ExtraTrees, untouched holdout** | **154** | **0.6891** | **0.8385** |
| Nonzero-score holdout subset | 146 | 0.7077 | 0.7381 |
| Final model, all training data (in-sample) | 769 | **0.1442** | 0.9953 |

**Training RMSE is explicitly displayed in notebook section 14.** Its large advantage over held-out
performance reflects model fitting and must not be presented as generalization accuracy.
Full-range correlation benefits from easily distinguished zero-score examples. No Kaggle leaderboard
score is claimed. See `results/experiment_results.csv` for all five experiments, including RandomForest
and HistGradientBoosting, and `results/holdout_bootstrap_intervals.csv` for uncertainty estimates.

## Dataset and inspection

- `Dataset_Final/train.csv`: 769 rows, columns `filename,label`; label range 0–5, including 37 zeros.
- `Dataset_Final/test.csv`: 216 rows, same columns; `label=-1` is an unused placeholder.
- All 985 listed WAVs are readable, mono, 16 kHz. No missing CSV values, duplicate CSV records,
  or exact duplicate audio were found. Training duration: 20.06–61.04 seconds; test: 6.48–61.03 seconds.
- **212 filenames occur in both splits.** Each split uses its own directory; features are keyed by split
  and filename, and per-file caches by audio content/format and extractor fingerprint.
- The supplied `sample_submission.csv` has **204 stale rows**, with only 25 test-ID matches.
  The user explicitly confirmed that **all 216 test.csv filenames in exact order** are authoritative.
  `submission.csv` preserves the required `filename,label` columns. This correction is documented in the notebook.

## Approach and explanation

PCM → floating-point mono/16 kHz → **177 descriptors** → median imputation → constant-feature removal
→ scaling → regression → clip to [0,5]. Learned preprocessing is fitted separately in each training fold.
No IDs, test labels or test-driven feature selection are used. Seed: 42. Development: 615; holdout: 154.

Features summarize duration, RMS, zero crossings, spectral centroid/bandwidth/rolloff/flatness/contrast,
20 MFCCs and differences, energy-based pauses/activity, and autocorrelation pitch/voicing proxies.
Original amplitude, silence and full clips are preserved. Correlated features are retained because no
validation experiment justifies deleting them. Constant features are removed inside the pipeline.

The strongest holdout permutation effects include `mfcc_01_std`, `mfcc_delta_01_std`, and `flatness_p10`.
These are acoustic correlates of ratings, not direct evidence of correct syntax. Accent, recording
conditions and task content may confound the signal. Speaker identities are unavailable, so an
unseen-speaker guarantee is not possible. Importance is an explanation after selection, not another tuning stage.

Local `faster-whisper-small` CPU/int8 transcription was benchmarked: 13.65 seconds for a 60.51-second
clip, approximately 3.7 hours for the full dataset based on that single pilot. Full transcription was
deferred to respect the compute budget. No paid API, GPU, transcription model or network is needed to run
the final solution. No ensemble or broad hyperparameter search was added.

## Run locally

Use Python 3.10 or a compatible newer environment. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/execute_notebook.py
.\.venv\Scripts\python.exe scripts/verify_artifacts.py
```

Put the dataset in `Dataset_Final/` with the supplied CSVs and `train/`, `test/` folders.
Alternatively set `$env:SHL_DATA_DIR = 'path/to/Dataset_Final'` before notebook execution.
The artifact verification script expects the delivered `Dataset_Final/` layout.

The notebook reuses cached audio features, compares the five candidates, evaluates the frozen winner,
refits on all labels and generates `submission.csv`, saved models, figures and measured reports.
Feature extraction took about 4.6 minutes locally; reruns reuse the 985 per-file cache entries.
Cached feature failures are reported explicitly; missing/corrupt audio never causes silent row deletion.

Optional command-line run without notebook plots:

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py
```

Predict one WAV after training:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py Dataset_Final/test/audio_128.wav
```

`scripts/build_notebook.py` regenerates the standalone notebook from maintained source modules and
its explanatory cells; doing so clears execution outputs. Run `execute_notebook.py` afterward.

## Run on Kaggle

1. Import `SHL_Grammar_Scoring.ipynb` into a Python notebook and attach the competition dataset.
2. Run all cells. The notebook discovers a unique CSV directory under `/kaggle/input`; if more than one
   dataset matches, set `SHL_DATA_DIR` explicitly in the environment cell.
3. Download/submit `/kaggle/working/submission.csv` after its validation cell passes.

The notebook embeds all functions, so uploading `src/` is unnecessary. It uses NumPy, pandas, SciPy,
scikit-learn, matplotlib, joblib and threadpoolctl; no internet installation or large model download
is included. It was executed end to end locally; execution on Kaggle itself is a separate check.

## Files

```text
SHL_Grammar_Scoring.ipynb       Executed, standalone notebook and brief report
submission.csv                216 predictions in exact test order
requirements.txt              Local dependencies
src/                          Dataset, audio and modeling functions
scripts/                      Execution, verification, prediction and ASR benchmark
features/audio_features.csv   Cached feature table (split-qualified)
features/audio/               Resumable per-file caches; ignored by Git
models/                       Local final/holdout models; ignored by Git
results/                      Audits, experiments, metrics, plots and verification
```

## Delivery and checks

`results/verification.json` records checks of schema/order/bounds, holdout isolation, audio integrity,
independent metric recomputation, executed notebook outputs, figures and audio/submission edge cases.
Raw competition audio, local environments and model binaries are excluded from Git by `.gitignore`.
The project is ready to publish; public GitHub publication and Kaggle upload are tracked separately
from local validation in `results/delivery_status.json`.
