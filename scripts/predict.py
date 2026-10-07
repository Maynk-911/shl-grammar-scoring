"""Predict one WAV with the saved model. Only load trusted local model artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.feature_extraction import extract_audio_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('audio', type=Path)
parser.add_argument('--model', type=Path, default=Path('models/grammar_model.joblib'))
args = parser.parse_args()
artifact = joblib.load(args.model)
source = Path(__file__).resolve().parents[1]/'src/feature_extraction.py'
actual_hash = hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest()
if artifact.get('extractor_fingerprint') != actual_hash:
    raise ValueError('Extractor changed since model training; regenerate the model before prediction.')
with threadpool_limits(limits=2):
    row = pd.DataFrame([extract_audio_features(args.audio)])
    if set(row.columns) != set(artifact['features']):
        raise ValueError('Feature schema does not match trained model')
    prediction = float(np.clip(artifact['model'].predict(row[artifact['features']])[0], *artifact['bounds']))
print(json.dumps({'filename': args.audio.name, 'grammar_score': prediction}, indent=2))
