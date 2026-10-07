"""Bundle delivery files, excluding raw competition data and the local environment."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
paths = [root/n for n in ['README.md', 'SHL_Grammar_Scoring.ipynb', 'submission.csv', 'requirements.txt', '.gitignore']]
for directory in ['src', 'scripts', 'results']:
    paths += [p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
paths += [root/'features/audio_features.csv']
# The trained model is included in the downloadable package for immediate WAV inference.
# Git excludes it; it can be recreated by running the notebook.
paths += [root/'models/grammar_model.joblib']
destination = root/'SHL_Grammar_Scoring_Project.zip'
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.relative_to(root))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert not any(n.startswith(('Dataset_Final/', '.venv/')) for n in archive.namelist())
print(f'Packaged {len(paths)} files: {destination.name} ({destination.stat().st_size/1024/1024:.2f} MB)')
