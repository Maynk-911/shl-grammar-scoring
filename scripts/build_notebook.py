"""Generate a standalone, readable notebook from the maintained source modules."""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# SHL Hiring Assessment 2026 — Grammar Scoring Engine

## 1–4. Problem, dataset and evaluation
Predict a continuous **0–5 grammar score** from a spoken WAV recording. The supplied data has
769 labeled training clips and 216 unlabeled test clips. Zeros are real training labels; they are
retained. Test `label=-1` values are placeholders and never used for learning.

**RMSE** measures score error in the original units; lower is better. **Pearson correlation**
measures linear agreement in ranking; higher is better. Constant predictions have undefined Pearson.
Both are reported on clipped predictions. Model selection uses development CV Pearson first and RMSE
as a tie-break, following the stated metric priority; the exact leaderboard aggregation was not independently verified.

**Submission correction authorized by the user:** `sample_submission.csv` contains 204 stale IDs.
The output uses its `filename,label` column schema and all 216 filenames in exact `test.csv` order.

**Architecture:** CSV/audio audit → mono 16 kHz waveform → 177 cached acoustic descriptors →
fold-local median imputation / constant removal / scaling → compact regressors →
development CV selection → untouched holdout evaluation → full-data refit → clipped submission.

The notebook is standalone: all required functions appear below. No API, model download or internet is required.''')
md('''## 5. Environment & imports
Use the installed Kaggle scientific Python packages. Locally, install `requirements.txt`.
`SHL_DATA_DIR` can point to a folder containing the three CSVs and the `train/`, `test/` WAV directories.
Only Kaggle input discovery is automatic; outputs are written to the current writable directory.
Seed 42 and fixed folds are used throughout.''')
code('''import os, sys, json, hashlib, time, platform
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import joblib

RESULTS = Path('results'); RESULTS.mkdir(exist_ok=True)
FEATURES = Path('features'); FEATURES.mkdir(exist_ok=True)
MODELS = Path('models'); MODELS.mkdir(exist_ok=True)
FIGURES = RESULTS / 'figures'; FIGURES.mkdir(exist_ok=True)
versions = {p: metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}
(RESULTS/'environment.json').write_text(json.dumps({'python': sys.version, 'packages': versions}, indent=2))
print('Python:', platform.python_version(), '| Packages:', versions)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
''')
md('''## 6. Data loading and integrity checks
Filenames are only unique **within** a split: 212 names occur in both train and test.
All joins and cache lookups preserve the split. The audit checks headers, durations, missing records,
sample rate, channels, truncated PCM, and exact audio duplicates. Errors produce a diagnostic report
and stop the run rather than silently dropping samples or fabricating predictions.
Speaker IDs are unavailable, so speaker overlap cannot be ruled out.''')
code((ROOT/'src/data.py').read_text(encoding='utf-8'))
code('''DATA_ROOT = find_data_root()
train, test, sample, audit, data_report = inspect_dataset(DATA_ROOT, RESULTS)
print('Shapes:', {name: frame.shape for name, frame in [('train', train), ('test', test), ('sample', sample)]})
display(train.head(3))
print('Missing values:', data_report['missing_values'])
print('Duplicate rows:', data_report['duplicate_rows'], '| Exact duplicate audio:', data_report['duplicate_audio'])
print('Audio sample rates:', data_report['sample_rates'], '| Channels:', data_report['channels'])
display(audit.groupby('split').duration.agg(['min','median','max']).round(2))
print('Sample/test ID overlap:', data_report['sample_test_overlap'], '| Submission policy:', data_report['submission_policy'])
''')
md('''## 7. Exploratory data analysis
Label and feature relationships use labeled training data only. Test metadata is audited for integrity,
but test distributions do not drive feature or model selection. The 37 zero scores remain in the task.
Short files are retained rather than trimmed or excluded.''')
code('''train_meta = train.merge(audit.query("split == 'train'"), on='filename', validate='one_to_one')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
train.label.value_counts().sort_index().plot.bar(ax=axes[0], color='#34699a')
axes[0].set(xlabel='Grammar score', ylabel='Count', title='Training score distribution')
axes[1].hist(train_meta.duration, bins=22, color='#479b87')
axes[1].set(xlabel='Duration (seconds)', ylabel='Count', title='Training clip durations')
axes[2].scatter(train_meta.duration, train_meta.label, s=12, alpha=.35)
axes[2].set(xlabel='Duration (seconds)', ylabel='Grammar score', title='Score vs duration')
save_plot('01_dataset_overview.png')
''')
md('''## 8–9. Audio preprocessing and feature engineering
Convert PCM to floating-point mono and resample to 16 kHz if necessary. Preserve amplitude, silence and
full duration; no aggressive denoising or peak normalization. Frames are 25 ms with a 10 ms hop.

The 177 descriptors cover duration, RMS energy, zero crossings, spectral shape and contrast,
20 MFCCs with temporal differences, pause/activity proxies, and autocorrelation pitch/voicing proxies.
They summarize acoustic fluency and articulation, **not grammatical syntax directly**. Chroma is omitted
because musical pitch-class summaries add little rationale here. Activity onset rate is not word speaking rate.

Cache keys include audio content, sample format and extractor source hash. Each completed file is saved,
so interrupted runs resume without recomputing completed features. Undefined pitch summaries are imputed
inside training folds. The five-clip pilot is run before full extraction.

**Transcript feasibility:** a locally cached `faster-whisper-small` CPU/int8 pilot took 13.65 seconds for
a 60.51-second training clip (one-file estimate: ~3.7 hours for 985 clips). Full transcription is deferred
to respect the resource budget; no transcript model or paid API is needed for this solution.
The benchmark script and measured report are included in the project.''')
feature_source = (ROOT/'src/feature_extraction.py').read_text(encoding='utf-8')
code(feature_source)
fingerprint = hashlib.sha256(feature_source.encode()).hexdigest()
code(f'''EXTRACTOR_FINGERPRINT = '{fingerprint}'
# Source fingerprint is generated by scripts/build_notebook.py; update it if editing the extractor.
with threadpool_limits(limits=2):
    pilot = cached_features(DATA_ROOT, audit.iloc[:5], FEATURES, EXTRACTOR_FINGERPRINT)
    features = cached_features(DATA_ROOT, audit, FEATURES, EXTRACTOR_FINGERPRINT)
train_features = features.loc[features.split.eq('train')].set_index('filename').loc[train.filename]
test_features = features.loc[features.split.eq('test')].set_index('filename').loc[test.filename]
X = train_features.drop(columns='split').reset_index(drop=True)
X_test = test_features.drop(columns='split').reset_index(drop=True)
y = train.label.to_numpy(float)
assert X.columns.equals(X_test.columns)
assert len(X) == len(train) and len(X_test) == len(test)
print('Feature matrices:', X.shape, X_test.shape)
''')
md('''## 10. Baseline and validation strategy
Reserve 20% of labeled examples (154 clips) for one final holdout evaluation. Target quantile bins
balance the split while pooling rare exact score levels. On the remaining 615 examples, compare five
predeclared models using three fixed folds. All imputers, constant removal and scalers are fitted inside
each fold. No filename, ID or test placeholder label is a feature.

The dummy mean and Ridge are simple references. ExtraTrees, RandomForest and histogram gradient boosting
are the stronger candidates. Tree depth/leaf constraints limit variance; boosting uses internal early stopping.
No broad parameter search or leaderboard tuning is performed. No ensemble is added without separate evidence.''')
code((ROOT/'src/model.py').read_text(encoding='utf-8'))
code('''development, holdout, folds = make_validation_split(y)
assignments = np.full(len(y), 'holdout', dtype=object)
assignments[development] = 'development'
fold_ids = np.full(len(y), -1)
for fold, (_, valid) in enumerate(folds):
    fold_ids[valid] = fold
pd.DataFrame({'filename': train.filename, 'partition': assignments, 'cv_fold': fold_ids}).to_csv(RESULTS/'validation_split.csv', index=False)
print(f'Development: {len(development)}, final holdout: {len(holdout)}, CV folds: {len(folds)}')
dev_X = X.iloc[development]
constant = dev_X.columns[dev_X.nunique(dropna=True) <= 1].tolist()
correlations = dev_X.corr().abs()
redundant_pairs = int(np.sum(np.triu(correlations.to_numpy() > .98, k=1)))
print('Constant features on development:', constant)
print('Feature pairs with |correlation| > .98:', redundant_pairs)
print('Missing feature values:', int(dev_X.isna().sum().sum()))
# Correlated descriptors are retained: no validation evidence supports an arbitrary deletion.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, feature in zip(axes, ['silence_ratio', 'mfcc_00_mean']):
    ax.scatter(dev_X[feature], y[development], alpha=.35, s=12)
    ax.set(xlabel=feature, ylabel='Grammar score', title='Development data only')
save_plot('02_audio_features.png')
''')
md('''## 11–12. Model experiments and comparison
The experiment table reports pooled out-of-fold development metrics and fold-mean training metrics.
Training fit quality is never used to choose the model. Development CV is used for selection, so its best
score can be optimistic; the separate holdout estimates generalization after the choice is frozen.''')
code('''experiments, best_name, oof = compare_models(X, y, folds, RESULTS)
display(experiments.drop(columns=['Notes']).round(4))
pd.DataFrame({'filename': train.filename, **oof}).to_csv(RESULTS/'development_oof_predictions.csv', index=False)
print('Selected using development CV:', best_name)
''')
md('''## 13. Freeze the selected model
The model family and settings are now fixed. Fit it on development data and evaluate the reserved
holdout once. No changes are made based on that result.''')
code('''selected, holdout_pred, holdout_metrics, development_metrics = fit_and_evaluate(X, y, best_name, development, holdout)
pd.DataFrame({'filename': train.filename.iloc[holdout], 'actual': y[holdout], 'prediction': holdout_pred}).to_csv(RESULTS/'holdout_predictions.csv', index=False)
joblib.dump({'model': selected, 'development': development, 'holdout': holdout}, MODELS/'holdout_model.joblib')
final_model = clone(make_candidates()[best_name])
with threadpool_limits(limits=4):
    final_model.fit(X, y)
    train_predictions = np.clip(final_model.predict(X), *BOUNDS)
training_metrics = regression_metrics(y, train_predictions)
joblib.dump({'model': final_model, 'features': X.columns.tolist(), 'bounds': BOUNDS,
             'extractor_fingerprint': EXTRACTOR_FINGERPRINT}, MODELS/'grammar_model.joblib')
''')
md('''## 14. Training performance — required Training RMSE
These are predictions from the final model refitted on **all 769 training clips**. They are in-sample
metrics, not estimates of performance on unseen speakers. The holdout score below was measured using
the development-only fitted model before this refit.''')
code('''print(f"Training RMSE: {training_metrics['RMSE']:.6f}")
print(f"Training Pearson Correlation: {training_metrics['Pearson']:.6f}")
''')
md('''## 15. Validation performance
Both a mean baseline and the frozen selected model are evaluated on the same reserved holdout.
The 95% bootstrap intervals describe sampling uncertainty on this split; they do not account for
unobserved speaker clustering or distribution shift.''')
code('''baseline_holdout = regression_metrics(y[holdout], np.full(len(holdout), y[development].mean()))
print(f"Validation RMSE: {holdout_metrics['RMSE']:.6f}")
print(f"Validation Pearson Correlation: {holdout_metrics['Pearson']:.6f}")
print(f"Holdout mean-baseline RMSE: {baseline_holdout['RMSE']:.6f}; Pearson undefined for constant predictions")
subgroup_rows = []
for name, mask in [('zero scores', y[holdout] == 0), ('nonzero scores', y[holdout] > 0)]:
    if mask.any():
        subgroup_rows.append({'Subset': name, 'Count': int(mask.sum()),
                              **regression_metrics(y[holdout][mask], holdout_pred[mask])})
subgroup_metrics = pd.DataFrame(subgroup_rows)
display(subgroup_metrics.round(4))
subgroup_metrics.to_csv(RESULTS/'holdout_subgroups.csv', index=False)
rng = np.random.default_rng(SEED)
bootstrap = []
for _ in range(500):
    idx = rng.integers(0, len(holdout), len(holdout))
    bootstrap.append(regression_metrics(y[holdout][idx], holdout_pred[idx]))
intervals = pd.DataFrame(bootstrap).quantile([.025, .975])
display(intervals.rename(index={.025: '95% lower', .975: '95% upper'}).round(4))
intervals.to_csv(RESULTS/'holdout_bootstrap_intervals.csv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
axes[0].scatter(y[holdout], holdout_pred, s=18, alpha=.55)
axes[0].plot([0,5], [0,5], '--', color='gray')
axes[0].set(xlabel='Actual score', ylabel='Predicted score', title='Untouched holdout')
axes[1].scatter(holdout_pred, y[holdout]-holdout_pred, s=18, alpha=.55)
axes[1].axhline(0, linestyle='--', color='gray')
axes[1].set(xlabel='Predicted score', ylabel='Actual − predicted', title='Holdout residuals')
save_plot('03_holdout_performance.png')
''')
md('''## 16. Interpretability
Rank candidate features using the development-fitted model (impurity importance for forests,
absolute standardized coefficient for Ridge; development correlation for other estimators).
Then permute the top 15 features three times on the holdout, measuring the increase in RMSE.
This is a post-selection explanation only. It does not trigger feature selection or model tuning.
Correlated features can share importance; association does not prove that a feature measures grammar.''')
code('''supported = X.columns[selected.named_steps['variancethreshold'].get_support()]
regressor = selected.steps[-1][1]
if hasattr(regressor, 'feature_importances_'):
    ranking = pd.Series(regressor.feature_importances_, index=supported)
elif hasattr(regressor, 'coef_'):
    ranking = pd.Series(np.abs(regressor.coef_), index=supported)
else:
    ranking = X.iloc[development].corrwith(pd.Series(y[development], index=development)).abs()
top_features = ranking.nlargest(15).index.tolist()
importance_rows = []
permutation_rng = np.random.default_rng(SEED)
with threadpool_limits(limits=4):
    for feature in top_features:
        increases = []
        for repeat in range(3):
            shuffled = X.iloc[holdout].copy()
            shuffled[feature] = permutation_rng.permutation(shuffled[feature].to_numpy())
            score = regression_metrics(y[holdout], selected.predict(shuffled))
            increases.append(score['RMSE']-holdout_metrics['RMSE'])
        importance_rows.append({'feature': feature, 'RMSE increase': np.mean(increases), 'repeat std': np.std(increases)})
importance = pd.DataFrame(importance_rows).sort_values('RMSE increase', ascending=False)
importance.to_csv(RESULTS/'feature_importance.csv', index=False)
fig, ax = plt.subplots(figsize=(8,5))
plot_data = importance.sort_values('RMSE increase')
ax.barh(plot_data.feature, plot_data['RMSE increase'], xerr=plot_data['repeat std'], color='#34699a')
ax.axvline(0, color='gray', linewidth=1)
ax.set(xlabel='Holdout RMSE increase after permutation', title='Feature importance (top 15 candidates)')
save_plot('04_feature_importance.png')
display(importance.head(5).round(4))
''')
md('''## 17–18. Final test prediction and submission generation
Predict from the final full-training model, clip to **[0, 5]**, then verify columns, row count,
finite values, unique IDs and exact test order after reading the written CSV back from disk.
Train/test filenames are never joined across splits. The stale 204-row template is used only for its schema.''')
code('''with threadpool_limits(limits=4):
    test_predictions = np.clip(final_model.predict(X_test), *BOUNDS)
submission = build_submission(test, sample, test_predictions, Path('submission.csv'))
assert len(submission) == 216
print('Validated submission:', submission.shape, '| columns:', submission.columns.tolist())
print('Prediction range:', float(submission.label.min()), 'to', float(submission.label.max()))
display(submission.head())
metrics_report = {'selected_model': best_name, 'feature_count': X.shape[1],
                  'selection': 'Highest development CV Pearson, RMSE tie-break; holdout excluded',
                  'training': training_metrics, 'holdout': holdout_metrics,
                  'baseline_holdout': {**baseline_holdout, 'Pearson': None}, 'development_fit': development_metrics,
                  'development_rows': len(development), 'holdout_rows': len(holdout),
                  'test_predictions': len(submission), 'score_bounds': list(BOUNDS)}
(RESULTS/'metrics.json').write_text(json.dumps(metrics_report, indent=2, allow_nan=False))
pd.DataFrame([{'artifact': 'submission.csv', 'model': best_name, 'rows': len(submission),
               'holdout_rmse': holdout_metrics['RMSE'], 'holdout_pearson': holdout_metrics['Pearson'],
               'status': 'Generated and locally validated; not uploaded to Kaggle'}]).to_csv(RESULTS/'submission_log.csv', index=False)
''')
md('''## 19. Brief report and conclusion
The summary below uses measured results from this execution. No leaderboard performance is claimed.''')
code('''chosen = experiments.loc[experiments.Model.eq(best_name)].iloc[0]
important_names = ', '.join(importance.head(3).feature)
display(Markdown(f"""
**Approach:** 177 interpretable acoustic descriptors, cached once, with fold-local preprocessing.
**Selected model:** {best_name}; highest development CV Pearson among five predeclared candidates.
Development CV RMSE **{chosen['CV RMSE']:.4f}**, Pearson **{chosen['CV Pearson']:.4f}**.

**Required Training RMSE: {training_metrics['RMSE']:.4f}**; Training Pearson **{training_metrics['Pearson']:.4f}**.
Untouched holdout RMSE **{holdout_metrics['RMSE']:.4f}**, Pearson **{holdout_metrics['Pearson']:.4f}**.
Mean-baseline holdout RMSE **{baseline_holdout['RMSE']:.4f}**.
For the nonzero-score holdout subset, RMSE is **{subgroup_metrics.loc[subgroup_metrics.Subset.eq('nonzero scores'), 'RMSE'].iloc[0]:.4f}**
and Pearson **{subgroup_metrics.loc[subgroup_metrics.Subset.eq('nonzero scores'), 'Pearson'].iloc[0]:.4f}**.
The full-score correlation benefits from the readily distinguishable zero-score examples.

**Explanation:** the strongest measured permutation effects include {important_names}. Acoustic summaries
capture fluency, pauses, articulation and recording properties that may correlate with ratings. They do not
directly detect grammatical errors, and recording conditions/accent may confound the relationship.

**Limitations:** small data, a single outer holdout, no speaker IDs, no full transcript analysis and no
independent leaderboard evaluation. Training metrics are optimistic. Permutation effects are noisy and
shared among correlated features. Zero labels were preserved as supplied; their semantics were not independently verified.

**Output:** `submission.csv` contains all 216 test IDs in exact order with the supplied column schema.
The user confirmed the template IDs were stale. Source, measured experiments, features and plots are saved.
Public GitHub publication and Kaggle upload remain external delivery steps.
"""))
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10.11'},
})
nbf.validate(notebook)
nbf.write(notebook, ROOT/'SHL_Grammar_Scoring.ipynb')
print(f'Wrote {len(cells)} notebook cells.')
