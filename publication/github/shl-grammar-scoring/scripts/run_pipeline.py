"""Run the pipeline locally; the final notebook contains the same implementation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from src.data import find_data_root, inspect_dataset
from src.feature_extraction import cached_features
from src.model import make_validation_split, compare_models, fit_and_evaluate, make_candidates, regression_metrics, build_submission, BOUNDS


def main():
    root = find_data_root()
    results = Path('results')
    train, test, sample, audit, report = inspect_dataset(root, results)
    source = Path('src/feature_extraction.py').read_text(encoding='utf-8')
    with threadpool_limits(limits=2):
        features = cached_features(root, audit, Path('features'), hashlib.sha256(source.encode()).hexdigest())
    train_features = features.loc[features.split.eq('train')].set_index('filename').loc[train.filename]
    test_features = features.loc[features.split.eq('test')].set_index('filename').loc[test.filename]
    X = train_features.drop(columns='split').reset_index(drop=True)
    X_test = test_features.drop(columns='split').reset_index(drop=True)
    y = train.label.to_numpy(float)
    development, holdout, folds = make_validation_split(y)
    assignments = np.full(len(y), 'holdout', dtype=object)
    assignments[development] = 'development'
    fold_ids = np.full(len(y), -1)
    for fold, (_, valid) in enumerate(folds):
        fold_ids[valid] = fold
    pd.DataFrame({'filename': train.filename, 'partition': assignments, 'cv_fold': fold_ids}).to_csv(results/'validation_split.csv', index=False)
    experiment, best_name, oof = compare_models(X, y, folds, results)
    selected, validation_pred, validation_metrics, development_metrics = fit_and_evaluate(X, y, best_name, development, holdout)
    pd.DataFrame({'filename': train.filename.iloc[holdout], 'actual': y[holdout], 'prediction': validation_pred}).to_csv(results/'holdout_predictions.csv', index=False)
    pd.DataFrame({'filename': train.filename, **oof}).to_csv(results/'development_oof_predictions.csv', index=False)
    final = clone(make_candidates()[best_name])
    with threadpool_limits(limits=4):
        final.fit(X, y)
        train_predictions = np.clip(final.predict(X), *BOUNDS)
        test_predictions = np.clip(final.predict(X_test), *BOUNDS)
    training_metrics = regression_metrics(y, train_predictions)
    submission = build_submission(test, sample, test_predictions, Path('submission.csv'))
    models = Path('models'); models.mkdir(exist_ok=True)
    joblib.dump({'model': final, 'features': X.columns.tolist(), 'bounds': BOUNDS,
                 'extractor_fingerprint': hashlib.sha256(source.encode()).hexdigest()}, models/'grammar_model.joblib')
    joblib.dump({'model': selected, 'development': development, 'holdout': holdout}, models/'holdout_model.joblib')
    model_report = {'selected_model': best_name, 'feature_count': X.shape[1],
                    'selection': 'Highest pooled development CV Pearson; RMSE tie-break. Holdout excluded.',
                    'training': training_metrics, 'development_fit': development_metrics,
                    'holdout': validation_metrics, 'development_rows': len(development), 'holdout_rows': len(holdout),
                    'test_predictions': len(submission), 'score_bounds': list(BOUNDS)}
    (results/'metrics.json').write_text(json.dumps(model_report, indent=2), encoding='utf-8')
    print(json.dumps(model_report, indent=2), flush=True)


if __name__ == '__main__':
    main()
