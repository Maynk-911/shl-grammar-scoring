"""Benchmark one local ASR clip without network/model downloads."""
from pathlib import Path
import json
import time
from faster_whisper import WhisperModel

root = Path(__file__).resolve().parents[1]
snapshots = Path.home() / '.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots'
models = [p for p in snapshots.glob('*') if (p / 'model.bin').is_file()]
if not models:
    raise SystemExit('No cached small ASR model available.')
start = time.perf_counter()
model = WhisperModel(str(models[0]), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
loaded = time.perf_counter()
segments, info = model.transcribe(str(root / 'Dataset_Final/train/audio_192.wav'),
                                  language='en', beam_size=1, vad_filter=True,
                                  condition_on_previous_text=False)
text = ' '.join(segment.text.strip() for segment in segments)
elapsed = time.perf_counter()-loaded
result = {'model': 'faster-whisper-small', 'device': 'cpu', 'compute_type': 'int8',
          'load_seconds': loaded-start, 'transcription_seconds': elapsed,
          'audio_seconds': info.duration, 'estimated_985_clip_hours': elapsed*985/3600,
          'filename': 'audio_192.wav', 'split': 'train'}
transcript_dir = root / 'features/transcripts'
transcript_dir.mkdir(parents=True, exist_ok=True)
(transcript_dir/'asr_pilot.json').write_text(json.dumps({**result, 'text': text}, indent=2), encoding='utf-8')
(root / 'results/asr_benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
