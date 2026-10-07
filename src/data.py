"""Strict dataset inspection; IDs are always qualified by split."""
from pathlib import Path
import hashlib
import json
import os
import wave
import numpy as np
import pandas as pd


def find_data_root():
    explicit = os.environ.get('SHL_DATA_DIR')
    if explicit:
        root = Path(explicit)
        if not (root / 'train.csv').is_file():
            raise FileNotFoundError(f'SHL_DATA_DIR has no train.csv: {root}')
        return root
    candidates = [Path('Dataset_Final'), Path('.')]
    if Path('/kaggle/input').exists():
        candidates += [p.parent for p in Path('/kaggle/input').rglob('train.csv')]
    valid = [p for p in candidates if all((p / n).is_file() for n in
             ['train.csv', 'test.csv', 'sample_submission.csv'])]
    valid = list(dict.fromkeys(p.resolve() for p in valid))
    if len(valid) != 1:
        raise FileNotFoundError(f'Expected one dataset, found {valid}. Set SHL_DATA_DIR.')
    return valid[0]


def inspect_dataset(root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, test, sample = [pd.read_csv(root / n) for n in
                          ['train.csv', 'test.csv', 'sample_submission.csv']]
    for name, frame in [('train', train), ('test', test), ('sample', sample)]:
        if list(frame.columns) != ['filename', 'label']:
            raise ValueError(f'Unexpected {name} columns: {frame.columns.tolist()}')
        if frame.filename.isna().any() or frame.filename.duplicated().any():
            raise ValueError(f'Missing or duplicate {name} IDs')
        if any(Path(n).name != n for n in frame.filename):
            raise ValueError('Expected plain filenames, not paths')
    if not np.isfinite(train.label).all() or not train.label.between(0, 5).all():
        raise ValueError('Training labels must be finite and in [0, 5]')
    rows = []
    for split, frame in [('train', train), ('test', test)]:
        actual = {p.name for p in (root / split).glob('*.wav')}
        expected = set(frame.filename)
        if actual != expected:
            raise ValueError(f'{split}: missing={expected-actual}, unlisted={actual-expected}')
        for filename in frame.filename:
            path = root / split / filename
            row = {'split': split, 'filename': filename, 'error': ''}
            try:
                with wave.open(str(path)) as wav:
                    channels, width, sr, frames, _, _ = wav.getparams()
                    pcm = wav.readframes(frames)
                    if len(pcm) != frames * channels * width or frames == 0:
                        raise ValueError('Truncated or empty PCM audio')
                    # Include sample rate/channel layout in the duplicate signature.
                    digest = hashlib.sha256(f'{sr}:{channels}:{width}:'.encode() + pcm).hexdigest()
                    row.update(duration=frames/sr, sample_rate=sr, channels=channels,
                               sample_width=width, sha256=digest)
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(output / 'audio_audit.csv', index=False)
    if audit.error.ne('').any():
        raise ValueError('Unreadable audio; see results/audio_audit.csv. No rows silently dropped.')
    if audit.sha256.duplicated().any():
        raise ValueError('Duplicate audio detected. Resolve duplicates/use grouped validation first.')
    report = {
        'train_rows': len(train), 'test_rows': len(test), 'sample_rows': len(sample),
        'columns': list(train.columns), 'target_range': [float(train.label.min()), float(train.label.max())],
        'label_counts': train.label.value_counts().sort_index().to_dict(),
        'missing_values': {n: d.isna().sum().to_dict() for n, d in [('train', train), ('test', test), ('sample', sample)]},
        'duplicate_rows': {n: int(d.duplicated().sum()) for n, d in [('train', train), ('test', test)]},
        'same_filename_across_splits': len(set(train.filename) & set(test.filename)),
        'duplicate_audio': int(audit.sha256.duplicated().sum()),
        'sample_matches_test': sample.filename.tolist() == test.filename.tolist(),
        'sample_test_overlap': len(set(sample.filename) & set(test.filename)),
        'submission_policy': 'User confirmed stale sample IDs; use test.csv IDs/order and sample columns.',
        'sample_rates': audit.sample_rate.value_counts().to_dict(),
        'channels': audit.channels.value_counts().to_dict(),
        'duration_by_split': audit.groupby('split').duration.agg(['min', 'median', 'max']).to_dict(),
    }
    (output / 'data_inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return train, test, sample, audit, report
