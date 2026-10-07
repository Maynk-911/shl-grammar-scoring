"""Cached local ASR and shared Whisper-encoder embeddings; never writes baseline files."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel, BatchedInferencePipeline
from src.feature_extraction import load_audio
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CONFIG = {'version': 1, 'model': 'faster-whisper-small', 'device': 'cpu', 'compute_type': 'int8',
          'beam_size': 1, 'batch_size': 2, 'cpu_threads': 6, 'language': 'en',
          'vad_filter': True, 'word_timestamps': False, 'max_new_tokens': 224,
          'encoder_pooling': 'duration-trimmed-frame-mean', 'audio_policy': 'full clip'}


class CapturingWhisper(WhisperModel):
    def encode(self, features):
        encoded = super().encode(features)
        values = np.asarray(encoded)
        for row, duration in zip(values, self.current_durations):
            valid = row[:max(1, min(len(row), int(np.ceil(duration*50))))]
            if self.embedding_sum is None:
                self.embedding_sum = np.zeros(row.shape[-1], dtype=np.float64)
            self.embedding_sum += valid.astype(np.float64).sum(axis=0)
            self.embedding_frames += len(valid)
        return encoded


class CapturingPipeline(BatchedInferencePipeline):
    def forward(self, features, tokenizer, chunks_metadata, options):
        self.model.current_durations = [m['duration'] for m in chunks_metadata]
        return super().forward(features, tokenizer, chunks_metadata, options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    audit = pd.read_csv(ROOT/'results/audio_audit.csv')
    records = audit.loc[audit.split.eq(args.split)]
    if args.limit:
        records = records.iloc[:args.limit]
    cache = HERE/'cache/speech'; cache.mkdir(parents=True, exist_ok=True)
    snapshots = Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots'
    model_dirs = [p for p in snapshots.glob('*') if (p/'model.bin').is_file()]
    if len(model_dirs) != 1:
        raise RuntimeError('Expected one complete local model; no downloads during extraction')
    config = {**CONFIG, 'model_revision': model_dirs[0].name,
              'model_bytes': (model_dirs[0]/'model.bin').stat().st_size}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    (HERE/'speech_config.json').write_text(json.dumps(config, indent=2))
    model = None
    processed, hits, failures = [], 0, []
    started = time.perf_counter()
    for number, row in enumerate(records.itertuples(index=False), 1):
        key = hashlib.sha256(f'{row.sha256}:{signature}'.encode()).hexdigest()
        output = cache/f'{key}.json'
        embedding_file = cache/f'{key}.npy'
        if output.exists() and embedding_file.exists():
            result = json.loads(output.read_text(encoding='utf-8'))
            hits += 1
        else:
            if model is None:
                model = CapturingWhisper(str(model_dirs[0]), device='cpu', compute_type='int8',
                                         cpu_threads=CONFIG['cpu_threads'], num_workers=1, local_files_only=True)
                pipeline = CapturingPipeline(model)
            try:
                tick = time.perf_counter()
                model.embedding_sum = None
                model.embedding_frames = 0
                audio, sr = load_audio(ROOT/'Dataset_Final'/row.split/row.filename)
                with threadpool_limits(limits=CONFIG['cpu_threads']):
                    segments, info = pipeline.transcribe(audio, language='en', beam_size=1, batch_size=CONFIG['batch_size'],
                                                        vad_filter=True, word_timestamps=False,
                                                        max_new_tokens=224, condition_on_previous_text=False)
                    segments = list(segments)
                duration = len(audio)/sr
                embedding = (model.embedding_sum/max(1, model.embedding_frames)).astype(np.float32) if model.embedding_sum is not None else np.zeros(768, dtype=np.float32)
                if not np.isfinite(embedding).all():
                    raise ValueError('Nonfinite encoder representation')
                result = {'split': row.split, 'filename': row.filename, 'audio_sha256': row.sha256,
                          'config_signature': signature, 'duration': duration,
                          'duration_after_vad': info.duration_after_vad,
                          'text': ' '.join(s.text.strip() for s in segments),
                          'segments': [{'start': s.start, 'end': s.end, 'text': s.text,
                                        'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob,
                                        'compression_ratio': s.compression_ratio} for s in segments],
                          'embedding_frames': model.embedding_frames,
                          'elapsed_seconds': time.perf_counter()-tick}
                np.save(embedding_file, embedding, allow_pickle=False)
                temporary = output.with_suffix('.tmp')
                temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
                temporary.replace(output)
            except Exception as exc:
                failures.append({'split': row.split, 'filename': row.filename, 'error': str(exc)})
                print(f'ERROR {row.filename}: {exc}', flush=True)
                (HERE/f'{args.split}_speech_errors.json').write_text(json.dumps(failures, indent=2))
                raise
        processed.append({'split': row.split, 'filename': row.filename, 'key': key,
                          'seconds': result['elapsed_seconds'], 'words': len(result['text'].split()),
                          'encoder_frames': result['embedding_frames']})
        if number % 10 == 0 or number == len(records):
            progress = {'completed': number, 'total': len(records), 'cache_hits': hits,
                        'elapsed_seconds': round(time.perf_counter()-started, 1),
                        'mean_uncached_seconds': float(np.mean([p['seconds'] for p in processed]))}
            (HERE/f'{args.split}_speech_progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
            pd.DataFrame(processed).to_csv(HERE/f'{args.split}_speech_index.csv', index=False)
    print('Speech extraction complete.', flush=True)


if __name__ == '__main__':
    main()
