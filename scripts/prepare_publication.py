"""Prepare immutable GitHub copies and Kaggle CSV, without training or editing predictions."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'publication/github/shl-grammar-scoring'
KAGGLE = ROOT/'publication/kaggle'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


audit = json.loads((ROOT/'results/final_artifact_audit.json').read_text())
assert audit['status'] == 'passed' and audit['protected_artifacts_unchanged']
for name, expected in audit['protected_sha256'].items():
    assert digest(ROOT/name) == expected, f'Frozen artifact changed: {name}'

# Explicit selection avoids publishing local runtime files or training data by accident.
files = [ROOT/n for n in ['.gitignore', 'README.md', 'UPLOAD_GUIDE.md',
                          'SHL_Grammar_Scoring.ipynb', 'requirements.txt', 'submission.csv',
                          'features/audio_features.csv']]
files += sorted((ROOT/'src').glob('*.py'))
script_names = ['benchmark_asr.py', 'build_notebook.py', 'execute_notebook.py',
                'package_project.py', 'predict.py', 'run_pipeline.py', 'verify_artifacts.py']
files += [ROOT/'scripts'/name for name in script_names]
files += sorted(p for p in (ROOT/'results').rglob('*') if p.is_file() and p.suffix in {'.json','.csv','.png'})
PUBLIC.mkdir(parents=True, exist_ok=True)
records = []
for source in files:
    relative = source.relative_to(ROOT)
    destination = PUBLIC/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert digest(source) == digest(destination)
    assert source.stat().st_size < 25*1024*1024
    records.append({'path': relative.as_posix(), 'bytes': source.stat().st_size, 'sha256': digest(source)})
manifest = {'status': 'verified', 'model_or_methodology_changes': False,
            'note': 'Hashes cover all upload files except this manifest itself.',
            'files': records}
(PUBLIC/'UPLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
expected = {r['path'] for r in records} | {'UPLOAD_MANIFEST.json'}
actual = {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob('*') if p.is_file()}
assert actual == expected, f'Unexpected upload files: {actual-expected}'
assert len(actual) <= 100
archive_path = ROOT/'SHL_GitHub_Upload.zip'
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(actual):
        archive.write(PUBLIC/name, 'shl-grammar-scoring/'+name)
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    for name in sorted(actual):
        assert archive.read('shl-grammar-scoring/'+name) == (PUBLIC/name).read_bytes()

# Prepare Kaggle after the GitHub export. Keep the original CSV unchanged.
KAGGLE.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT/'submission.csv', KAGGLE/'submission.csv')
test = pd.read_csv(ROOT/'Dataset_Final/test.csv')
submission = pd.read_csv(KAGGLE/'submission.csv')
assert submission.filename.tolist() == test.filename.tolist()
assert list(submission.columns) == ['filename','label'] and len(submission) == 216
assert submission.label.between(0,5).all() and not submission.isna().any().any()
checksum = digest(KAGGLE/'submission.csv')
assert checksum == audit['submission_sha256'] == digest(PUBLIC/'submission.csv')
(KAGGLE/'submission.sha256').write_text(f'{checksum}  submission.csv\n', encoding='ascii')
(KAGGLE/'UPLOAD_ONLY_SUBMISSION_CSV.txt').write_text(
    'Upload only submission.csv to SHL Hiring Assessment 2026.\n'
    '216 rows; filename,label; exact test.csv order; unchanged verified predictions.\n'
    'Description: ExtraTrees acoustic features; seed 42; CV Pearson 0.7886; '
    'holdout RMSE 0.6891, Pearson 0.8385.\n', encoding='utf-8')
for name, expected_hash in audit['protected_sha256'].items():
    assert digest(ROOT/name) == expected_hash, f'Frozen artifact changed: {name}'
summary = {'status': 'ready for user upload', 'retrained': False, 'protected_artifacts_unchanged': True,
           'github_folder': PUBLIC.relative_to(ROOT).as_posix(),
           'github_zip': archive_path.name, 'github_files': len(actual),
           'github_zip_bytes': archive_path.stat().st_size,
           'kaggle_csv': (KAGGLE/'submission.csv').relative_to(ROOT).as_posix(),
           'submission_rows': len(submission), 'submission_sha256': checksum,
           'prediction_min': float(submission.label.min()), 'prediction_max': float(submission.label.max()),
           'publications_performed': False}
(ROOT/'publication/preparation_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
