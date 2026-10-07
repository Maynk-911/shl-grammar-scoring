"""Compact CPU audio descriptors, cached by audio content and extractor source."""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.fft import rfft, irfft, dct
from functools import lru_cache

FEATURE_VERSION = 'audio-v1-16k-25ms-10ms-40mel-20mfcc'


def load_audio(path, target_sr=16000):
    sr, audio = wavfile.read(path)
    if audio.dtype.kind in 'iu':
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32)
        audio = (audio - 128) / 128 if info.min == 0 else audio / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f'Empty or nonfinite waveform: {path}')
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
    # Keep silence, amplitude and full duration; no denoising or peak normalization.
    return audio, target_sr


@lru_cache(maxsize=4)
def mel_filters(sr=16000, n_fft=512, n_mels=40):
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    lo, hi = 2595*np.log10(1+50/700), 2595*np.log10(1+(sr/2)/700)
    edges = 700*(10**(np.linspace(lo, hi, n_mels+2)/2595)-1)
    filters = []
    for left, center, right in zip(edges[:-2], edges[1:-1], edges[2:]):
        triangle = np.maximum(0, np.minimum((freqs-left)/(center-left), (right-freqs)/(right-center)))
        filters.append(triangle / max(triangle.sum(), 1e-12))
    return np.asarray(filters, dtype=np.float32)


def summarize(out, name, values, percentiles=True):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    stats = {'mean': np.mean, 'std': np.std}
    for suffix, func in stats.items():
        out[f'{name}_{suffix}'] = float(func(values)) if values.size else np.nan
    if percentiles:
        for q in [10, 90]:
            out[f'{name}_p{q}'] = float(np.percentile(values, q)) if values.size else np.nan


def extract_audio_features(path):
    y, sr = load_audio(path)
    duration = len(y)/sr
    frame_len, hop, n_fft = 400, 160, 512
    padded = np.pad(y, (0, max(0, frame_len-len(y))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[::hop]
    rms = np.sqrt(np.mean(frames**2, axis=1)+1e-12)
    threshold = max(1e-4, float(np.percentile(rms, 95))*0.1)
    active = rms > threshold
    spectrum = np.abs(rfft(frames*np.hanning(frame_len).astype(np.float32), n=n_fft, axis=1))
    power = spectrum**2
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    norm = spectrum / np.maximum(spectrum.sum(axis=1, keepdims=True), 1e-12)
    centroid = norm @ freqs
    bandwidth = np.sqrt(np.sum(norm*(freqs[None, :]-centroid[:, None])**2, axis=1))
    rolloff = freqs[np.argmax(np.cumsum(norm, axis=1) >= .85, axis=1)]
    flatness = np.exp(np.mean(np.log(power+1e-12), axis=1))/(power.mean(axis=1)+1e-12)
    flux = np.sqrt(np.mean(np.diff(norm, axis=0)**2, axis=1))
    mfcc = dct(np.log(power @ mel_filters(sr, n_fft).T+1e-10), type=2, norm='ortho', axis=1)[:, :20]
    out = {'duration': duration, 'silence_ratio': float(1-active.mean()),
           'active_seconds': float(active.mean()*duration),
           'clipping_ratio': float(np.mean(np.abs(y) >= .999)),
           'dc_offset': float(y.mean()), 'peak_amplitude': float(np.max(np.abs(y)))}
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    for name, values in [('rms', rms), ('zcr', zcr), ('centroid', centroid),
                         ('bandwidth', bandwidth), ('rolloff', rolloff), ('flatness', flatness), ('flux', flux)]:
        summarize(out, name, values)
    for i in range(20):
        summarize(out, f'mfcc_{i:02d}', mfcc[:, i])
        summarize(out, f'mfcc_delta_{i:02d}', np.diff(mfcc[:, i]), False)
    for i, (lo, hi) in enumerate(zip([50, 200, 400, 800, 1600, 3200], [200, 400, 800, 1600, 3200, 8001])):
        band = spectrum[:, (freqs >= lo) & (freqs < hi)]
        contrast = 20*np.log10((np.percentile(band, 90, axis=1)+1e-8)/(np.percentile(band, 10, axis=1)+1e-8))
        summarize(out, f'contrast_{i}', contrast, False)
    changes = np.flatnonzero(np.diff(np.r_[False, ~active, False]))
    pauses = (changes[1::2]-changes[::2])*hop/sr
    pauses = pauses[pauses >= .2]
    out.update(pause_count_per_second=float(len(pauses)/duration),
               pause_duration_mean=float(pauses.mean()) if len(pauses) else 0.,
               pause_duration_max=float(pauses.max()) if len(pauses) else 0.,
               activity_onsets_per_second=float(np.sum(np.diff(active.astype(int)) == 1)/duration))
    # Autocorrelation pitch is a voicing/prosody proxy, not a phonetic annotation.
    long_y = np.pad(y, (0, max(0, 640-len(y))))
    pitch_frames = np.lib.stride_tricks.sliding_window_view(long_y, 640)[::320].copy()
    pitch_energy = np.sqrt(np.mean(pitch_frames**2, axis=1))
    pitch_frames -= pitch_frames.mean(axis=1, keepdims=True)
    pitch_frames *= np.hanning(640)
    acf = irfft(np.abs(rfft(pitch_frames, n=2048, axis=1))**2, n=2048, axis=1)[:, :321]
    lags = np.argmax(acf[:, 40:321], axis=1)+40
    periodicity = acf[np.arange(len(acf)), lags]/np.maximum(acf[:, 0], 1e-12)
    voiced = (periodicity > .45) & (pitch_energy > threshold)
    out['voiced_frame_ratio'] = float(voiced.mean())
    summarize(out, 'pitch_hz', sr/lags[voiced])
    summarize(out, 'periodicity', periodicity, False)
    return out


def cached_features(root, audit, cache_dir, extractor_fingerprint):
    cache_dir = Path(cache_dir)
    (cache_dir / 'audio').mkdir(parents=True, exist_ok=True)
    rows, errors, hits = [], [], 0
    start = time.perf_counter()
    for number, row in enumerate(audit.itertuples(index=False), 1):
        key = hashlib.sha256(f'{FEATURE_VERSION}:{extractor_fingerprint}:{row.sha256}'.encode()).hexdigest()
        cache = cache_dir / 'audio' / f'{key}.json'
        try:
            if cache.exists():
                feature = json.loads(cache.read_text())
                hits += 1
            else:
                feature = extract_audio_features(root / row.split / row.filename)
                temporary = cache.with_suffix('.tmp')
                temporary.write_text(json.dumps(feature), encoding='utf-8')
                temporary.replace(cache)
            rows.append({'split': row.split, 'filename': row.filename, **feature})
        except Exception as exc:
            errors.append({'split': row.split, 'filename': row.filename, 'error': str(exc)})
        if number % 100 == 0 or number == len(audit):
            print(f'Audio features: {number}/{len(audit)}, cache hits={hits}, elapsed={time.perf_counter()-start:.1f}s', flush=True)
    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / 'extraction_errors.csv', index=False)
        raise RuntimeError('Feature extraction failed. See extraction_errors.csv; rows were not dropped.')
    result = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    result.to_csv(cache_dir / 'audio_features.csv', index=False)
    return result
