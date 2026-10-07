"""Read-only model/CSV consistency checks. Never fits or changes an estimator."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import nbformat
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.model import regression_metrics, make_validation_split

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = [ROOT/n for n in ['SHL_Grammar_Scoring.ipynb', 'submission.csv',
                 'models/grammar_model.joblib', 'models/holdout_model.joblib',
                 'features/audio_features.csv', 'results/metrics.json',
                 'results/experiment_results.csv']]
    protected += sorted((ROOT/'src').glob('*.py'))
    before = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in protected}
    train = pd.read_csv(ROOT/'Dataset_Final/train.csv')
    test = pd.read_csv(ROOT/'Dataset_Final/test.csv')
    submission = pd.read_csv(ROOT/'submission.csv')
    metrics = json.loads((ROOT/'results/metrics.json').read_text())
    for split, frame in [('train', train), ('test', test)]:
        assert {p.name for p in (ROOT/'Dataset_Final'/split).glob('*.wav')} == set(frame.filename)
    assert submission.filename.tolist() == test.filename.tolist()
    assert list(submission.columns) == ['filename', 'label'] and len(submission) == 216
    assert submission.filename.is_unique and np.isfinite(submission.label).all()
    assert submission.label.between(0, 5).all()
    features = pd.read_csv(ROOT/'features/audio_features.csv')
    assert not features.duplicated(['split', 'filename']).any()
    final = joblib.load(ROOT/'models/grammar_model.joblib')
    evaluation = joblib.load(ROOT/'models/holdout_model.joblib')
    X = features.query("split == 'train'").set_index('filename').loc[train.filename, final['features']]
    X_test = features.query("split == 'test'").set_index('filename').loc[test.filename, final['features']]
    assert X.shape == (769, 177) and X_test.shape == (216, 177)
    y = train.label.to_numpy(float)
    development, holdout, _ = make_validation_split(y)
    assert np.array_equal(development, evaluation['development'])
    assert np.array_equal(holdout, evaluation['holdout'])
    with threadpool_limits(limits=4):
        train_pred = np.clip(final['model'].predict(X), *final['bounds'])
        test_pred = np.clip(final['model'].predict(X_test), *final['bounds'])
        holdout_pred = np.clip(evaluation['model'].predict(X.iloc[holdout]), *final['bounds'])
    assert np.allclose(test_pred, submission.label, rtol=0, atol=1e-12)
    for split, actual, predicted in [('training', y, train_pred), ('holdout', y[holdout], holdout_pred)]:
        scores = regression_metrics(actual, predicted)
        for metric in ['RMSE', 'Pearson']:
            assert np.isclose(scores[metric], metrics[split][metric], rtol=0, atol=1e-12)
    saved_holdout = pd.read_csv(ROOT/'results/holdout_predictions.csv')
    assert saved_holdout.filename.tolist() == train.filename.iloc[holdout].tolist()
    assert np.array_equal(saved_holdout.actual, y[holdout])
    assert np.allclose(saved_holdout.prediction, holdout_pred, rtol=0, atol=1e-12)
    notebook = nbformat.read(ROOT/'SHL_Grammar_Scoring.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not [o for c in code for o in c.outputs if o.output_type == 'error']
    text = '\n'.join(o.get('text', '') for c in code for o in c.outputs)
    for label, value in [('Training RMSE', metrics['training']['RMSE']),
                         ('Training Pearson Correlation', metrics['training']['Pearson']),
                         ('Validation RMSE', metrics['holdout']['RMSE']),
                         ('Validation Pearson Correlation', metrics['holdout']['Pearson'])]:
        assert f'{label}: {value:.6f}' in text
    for module in ['data.py', 'feature_extraction.py', 'model.py']:
        assert any(c.source.strip() == (ROOT/'src'/module).read_text(encoding='utf-8').strip() for c in code)
    fingerprint = hashlib.sha256((ROOT/'src/feature_extraction.py').read_text(encoding='utf-8').encode()).hexdigest()
    assert final['extractor_fingerprint'] == fingerprint
    images = list((ROOT/'results/figures').glob('*.png'))
    assert len(images) == 4 and all(p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for p in images)
    experiments = pd.read_csv(ROOT/'results/experiment_results.csv')
    best = experiments.query("Model != 'Dummy mean'").sort_values(['CV Pearson','CV RMSE'], ascending=[False,True]).iloc[0]
    assert best.Model == metrics['selected_model'] == 'ExtraTrees'
    assert final['model'].steps[-1][1].n_estimators == 300
    for value in [metrics['training']['RMSE'], metrics['holdout']['RMSE'], metrics['holdout']['Pearson']]:
        assert f'{value:.4f}' in (ROOT/'README.md').read_text(encoding='utf-8')
    with zipfile.ZipFile(ROOT/'SHL_Grammar_Scoring_Project.zip') as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert archive.read(name) == (ROOT/name).read_bytes(), f'Previous archive differs: {name}'
    after = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in protected}
    assert before == after
    report = {'status': 'passed', 'retraining_performed': False,
              'methodology_changed': False, 'protected_artifacts_unchanged': True,
              'checks': ['train/test audio filename inventories', '216-row CSV schema, bounds and order',
                         'saved final model reproduces every submission prediction',
                         'saved models reproduce reported training and holdout metrics',
                         'saved split and holdout labels agree with original training CSV',
                         'executed notebook contains the recorded metric values and matching source',
                         'README, experiment selection and model parameters agree',
                         'four PNG figures present; previous project archive matches source files'],
              'metrics': metrics, 'submission_sha256': before['submission.csv'],
              'protected_sha256': before}
    (ROOT/'results/final_artifact_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','protected_sha256']}, indent=2))


if __name__ == '__main__':
    main()
