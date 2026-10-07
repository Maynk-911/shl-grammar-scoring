# Final GitHub and Kaggle upload guide

The trained model, methodology, executed notebook and predictions are frozen. Verification reproduced
every submission prediction from the saved model and checked the reported training/holdout metrics.
No retraining was performed. Training RMSE: **0.1442**; holdout RMSE: **0.6891**; holdout Pearson: **0.8385**.

## 1. Publish the prepared GitHub folder

Use **`publication/github/shl-grammar-scoring/`**, or extract **`SHL_GitHub_Upload.zip`**.
Upload the contents of `shl-grammar-scoring`, preserving the directories below.
Do not upload the ZIP as the repository's only file.

```text
.gitignore
README.md
UPLOAD_GUIDE.md
UPLOAD_MANIFEST.json
SHL_Grammar_Scoring.ipynb
requirements.txt
submission.csv
src/                         All 4 Python files
scripts/                     All 7 prepared Python scripts
features/audio_features.csv
results/                     All prepared JSON/CSV reports and figures/
```

`UPLOAD_MANIFEST.json` lists every other upload file with its size and SHA-256.
The upload folder contains no raw audio, Dataset_Final, virtual environment, credentials, transcript
cache or model binaries. The existing `SHL_Grammar_Scoring_Project.zip` is the local runnable bundle
with the trained model; use the new GitHub ZIP for publication.

1. Sign in and open [New repository](https://github.com/new).
2. Set the name to **`shl-grammar-scoring`** and visibility to **Public**.
3. Suggested description: `CPU-based grammar scoring from speech audio for SHL Hiring Assessment 2026.`
4. Leave automatic README, .gitignore, license and code generation options off; create the repository.
   These preparation choices follow [GitHub's repository creation guidance](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).
5. Click **uploading an existing file** on the empty repository, or **Add file > Upload files**.
6. In File Explorer, open the prepared `shl-grammar-scoring` folder. Select its contents, including
   `.gitignore`, and drag them into the upload area. Preserve `src/`, `scripts/`, `features/`, and `results/`.
7. Use commit message **`Add verified SHL grammar scoring solution`** and commit the upload.
   If GitHub instead proposes a branch/PR, create and merge it into the default branch.
   See [GitHub's upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).
8. Confirm the README appears at the repository root and the notebook displays saved outputs,
   **Training RMSE**, validation results and plots. Check the Public visibility and copy the repository URL.

No notebook execution or model regeneration is needed for this upload. A clone can rebuild the model
using the README instructions once the competition dataset is supplied separately.

## 2. Upload the final CSV to Kaggle

Exact file: **`publication/kaggle/submission.csv`**. It is byte-for-byte identical to the verified
project-root `submission.csv` and the GitHub copy.

- 216 prediction rows, plus one header line.
- Columns exactly `filename,label`; no extra index column.
- Filenames and order exactly match `Dataset_Final/test.csv`.
- All predictions are finite and in [0,5].
- The 204-row sample IDs were stale; the user explicitly authorized the 216 test IDs.

1. Sign in to the enrolled account and open [SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026).
2. Click **Submit Predictions** and select the exact CSV above.
3. Suggested description: **`ExtraTrees acoustic features; seed 42; CV Pearson 0.7886; holdout RMSE 0.6891, Pearson 0.8385.`**
4. Submit once. Wait for processing, then record the submission status and returned leaderboard score.
   CSV upload and processing are described in [Kaggle's competition documentation](https://www.kaggle.com/docs/competitions).

Upload only the CSV in this step, not the ZIP, checksum file, or notebook. The executed notebook is
provided in the public repository; supply its link/file through any separate SHL assessment form requested.
If Kaggle reports an ID/row-count error, retain the error message and resolve the competition format;
do not reorder IDs or retrain to address a format issue.

No GitHub publication or Kaggle submission has been performed by this preparation step.
