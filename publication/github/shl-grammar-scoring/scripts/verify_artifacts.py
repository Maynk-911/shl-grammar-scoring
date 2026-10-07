"""Check submission alignment, validation isolation, notebook outputs and cache integrity."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import tempfile
import numpy as np
import pandas as pd
import nbformat
from scipy.io import wavfile
from src.feature_extraction import extract_audio_features, load_audio
from src.model import build_submission, regression_metrics

root = Path(__file__).resolve().parents[1]
train = pd.read_csv(root/'Dataset_Final/train.csv')
test = pd.read_csv(root/'Dataset_Final/test.csv')
sample = pd.read_csv(root/'Dataset_Final/sample_submission.csv')
submission = pd.read_csv(root/'submission.csv')
assert list(submission.columns) == list(sample.columns) == ['filename', 'label']
assert submission.filename.tolist() == test.filename.tolist()
assert len(submission) == 216 and submission.filename.is_unique
assert np.isfinite(submission.label).all() and submission.label.between(0, 5).all()
features = pd.read_csv(root/'features/audio_features.csv')
assert len(features) == 985 and not features.duplicated(['split', 'filename']).any()
assert features.shape[1] == 179
assert len(features.query("split == 'train'")) == 769
assert len(features.query("split == 'test'")) == 216
audit = pd.read_csv(root/'results/audio_audit.csv')
assert audit.error.isna().all() and audit.sha256.is_unique
split = pd.read_csv(root/'results/validation_split.csv')
assert split.filename.tolist() == train.filename.tolist()
assert split.partition.eq('holdout').sum() == 154
assert split.loc[split.partition.eq('holdout'), 'cv_fold'].eq(-1).all()
oof = pd.read_csv(root/'results/development_oof_predictions.csv')
assert oof.loc[split.partition.eq('holdout')].drop(columns='filename').isna().all().all()
assert oof.loc[split.partition.eq('development')].drop(columns='filename').notna().all().all()
holdout = pd.read_csv(root/'results/holdout_predictions.csv')
assert set(holdout.filename) == set(split.loc[split.partition.eq('holdout'), 'filename'])
metrics = json.loads((root/'results/metrics.json').read_text())
measured = regression_metrics(holdout.actual, holdout.prediction)
assert np.isclose(measured['RMSE'], metrics['holdout']['RMSE'])
assert np.isclose(measured['Pearson'], metrics['holdout']['Pearson'])
notebook = nbformat.read(root/'SHL_Grammar_Scoring.ipynb', as_version=4)
nbformat.validate(notebook)
code_cells = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is not None for c in code_cells)
assert not [o for c in code_cells for o in c.outputs if o.output_type == 'error']
rendered_text = '\n'.join(o.get('text', '') for c in code_cells for o in c.outputs)
assert 'Training RMSE:' in rendered_text and 'Validation RMSE:' in rendered_text
assert sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs) >= 4
source = (root/'src/feature_extraction.py').read_text(encoding='utf-8')
fingerprint = hashlib.sha256(source.encode()).hexdigest()
assert any(source.strip() == c.source for c in code_cells)
assert any(fingerprint in c.source for c in code_cells)

# Meaningful edge cases: silent waveform, stereo resampling, wrong row count,
# clipping/order preservation with a stale template, and undefined correlation.
with tempfile.TemporaryDirectory(dir=root) as temporary:
    temp = Path(temporary)
    wavfile.write(temp/'silence.wav', 16000, np.zeros(16000, dtype=np.int16))
    silent = extract_audio_features(temp/'silence.wav')
    assert silent['silence_ratio'] == 1. and np.isnan(silent['pitch_hz_mean'])
    signal = (10000*np.sin(2*np.pi*200*np.arange(8000)/8000)).astype(np.int16)
    wavfile.write(temp/'stereo.wav', 8000, np.column_stack([signal, signal]))
    mono, rate = load_audio(temp/'stereo.wav')
    assert rate == 16000 and mono.ndim == 1 and len(mono) == 16000
    fake_test = pd.DataFrame({'filename': ['b.wav','a.wav'], 'label': [-1,-1]})
    fake_sample = pd.DataFrame({'filename': ['stale.wav'], 'label': [-1]})
    result = build_submission(fake_test, fake_sample, [-3, 9], temp/'check.csv')
    assert result.filename.tolist() == ['b.wav','a.wav'] and result.label.tolist() == [0,5]
    try:
        build_submission(fake_test, fake_sample, [1], temp/'invalid.csv')
        raise AssertionError('Wrong-length predictions accepted')
    except ValueError:
        pass
    assert np.isnan(regression_metrics([0,1,2],[1,1,1])['Pearson'])
report = {'status': 'passed', 'notebook_code_cells_executed': len(code_cells),
          'submission_rows': len(submission), 'feature_rows': len(features),
          'features': features.shape[1]-2,
          'checks': ['exact CSV schema and test order', 'finite bounded predictions',
                     'split-qualified feature rows', 'no exact audio duplicates',
                     'holdout excluded from CV', 'holdout metrics independently recomputed',
                     'notebook schema, execution and embedded figures',
                     'notebook/source extractor consistency', 'audio and submission edge cases']}
(root/'results/verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
