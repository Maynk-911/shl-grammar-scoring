"""Small predeclared model comparison with an untouched final holdout."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
BOUNDS = (0., 5.)


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y), np.clip(np.asarray(prediction), *BOUNDS)
    # sqrt(mean(...)) is compatible with both old and new sklearn RMSE APIs.
    rmse = float(np.sqrt(np.mean((y-prediction)**2)))
    correlation = float(pearsonr(y, prediction).statistic) if np.std(prediction) > 1e-10 and np.std(y) > 1e-10 else np.nan
    return {'RMSE': rmse, 'Pearson': correlation}


def make_candidates():
    def pipeline(model):
        # Every learned transform is fitted inside each training fold.
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             VarianceThreshold(), StandardScaler(), model)
    return {
        'Dummy mean': pipeline(DummyRegressor()),
        'Ridge': pipeline(Ridge(alpha=100.)),
        'ExtraTrees': pipeline(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                              max_features=.8, random_state=SEED, n_jobs=4)),
        'RandomForest': pipeline(RandomForestRegressor(n_estimators=250, min_samples_leaf=3,
                                max_features=.8, random_state=SEED, n_jobs=4)),
        'HistGradientBoosting': pipeline(HistGradientBoostingRegressor(max_iter=220,
                                         max_leaf_nodes=10, min_samples_leaf=20,
                                         learning_rate=.05, l2_regularization=5.,
                                         early_stopping=True, random_state=SEED)),
    }


def make_validation_split(y):
    # Quantile bins combine rare exact score levels; all samples retain their actual label.
    bins = pd.qcut(pd.Series(y), q=5, labels=False, duplicates='drop').to_numpy()
    development, holdout = train_test_split(np.arange(len(y)), test_size=.2,
                                            stratify=bins, random_state=SEED)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = [(development[a], development[b]) for a,b in splitter.split(development, bins[development])]
    return development, holdout, folds


def compare_models(X, y, folds, output, feature_set='audio'):
    output = Path(output)
    rows, oof_predictions = [], {}
    for name, candidate in make_candidates().items():
        start = time.perf_counter()
        pred = np.full(len(y), np.nan)
        train_scores, fold_scores = [], []
        for fold, (fit, valid) in enumerate(folds):
            fitted = clone(candidate)
            with threadpool_limits(limits=4):
                fitted.fit(X.iloc[fit], y[fit])
                pred[valid] = np.clip(fitted.predict(X.iloc[valid]), *BOUNDS)
                train_scores.append(regression_metrics(y[fit], fitted.predict(X.iloc[fit])))
            fold_scores.append(regression_metrics(y[valid], pred[valid]))
        mask = np.isfinite(pred)
        score = regression_metrics(y[mask], pred[mask])
        row = {'Experiment': name, 'Feature Set': feature_set, 'Model': name,
               'CV RMSE': score['RMSE'], 'CV Pearson': score['Pearson'],
               'CV fold RMSE std': float(np.std([v['RMSE'] for v in fold_scores])),
               'Training RMSE': float(np.mean([s['RMSE'] for s in train_scores])),
               'Training Pearson': float(np.nanmean([s['Pearson'] for s in train_scores])) if name != 'Dummy mean' else np.nan,
               'Seconds': time.perf_counter()-start,
               'Notes': '3-fold OOF on development only; training metrics are fold means'}
        rows.append(row)
        oof_predictions[name] = pred
        pd.DataFrame(rows).to_csv(output / 'experiment_results.csv', index=False)
        print(f'{feature_set} | {name}: CV RMSE={score["RMSE"]:.4f}, Pearson={score["Pearson"]:.4f}', flush=True)
    table = pd.DataFrame(rows)
    # Pearson is primary as requested; use RMSE as the secondary sort criterion.
    eligible = table.loc[table.Model != 'Dummy mean'].copy()
    eligible['CV Pearson'] = eligible['CV Pearson'].fillna(-np.inf)
    best_name = eligible.sort_values(['CV Pearson', 'CV RMSE'], ascending=[False, True]).iloc[0].Model
    return table, best_name, oof_predictions


def fit_and_evaluate(X, y, best_name, development, holdout):
    selected = clone(make_candidates()[best_name])
    with threadpool_limits(limits=4):
        selected.fit(X.iloc[development], y[development])
        holdout_pred = np.clip(selected.predict(X.iloc[holdout]), *BOUNDS)
        development_pred = np.clip(selected.predict(X.iloc[development]), *BOUNDS)
    return selected, holdout_pred, regression_metrics(y[holdout], holdout_pred), regression_metrics(y[development], development_pred)


def build_submission(test, sample, predictions, destination):
    predictions = np.clip(np.asarray(predictions, dtype=float), *BOUNDS)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError('Invalid prediction count or nonfinite predictions')
    # Explicit user authorization: sample IDs are stale; test.csv defines rows/order.
    if list(sample.columns) != ['filename', 'label']:
        raise ValueError('Submission schema changed')
    submission = pd.DataFrame({'filename': test.filename.to_numpy(), 'label': predictions})
    submission = submission[sample.columns]
    submission.to_csv(destination, index=False)
    check = pd.read_csv(destination)
    assert list(check.columns) == list(sample.columns)
    assert check.filename.tolist() == test.filename.tolist()
    assert len(check) == len(test) and check.filename.is_unique
    assert np.isfinite(check.label).all() and check.label.between(*BOUNDS).all()
    return submission
