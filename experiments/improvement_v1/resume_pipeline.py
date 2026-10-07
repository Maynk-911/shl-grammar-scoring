"""Sequential continuation: join live extraction, reuse caches, evaluate, then gate test work."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import os
import subprocess
import time
import psutil
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
status_path=HERE/'continuation_status.json'
log_path=HERE/'continuation.log'


def status(stage,**details):
    record={'stage':stage,**details}
    temporary=status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2))
    temporary.replace(status_path)
    print(json.dumps(record),flush=True)


def active_extractors():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if 'python' in (p.info['name'] or '').lower() and any(
                'improvement_v1/extract_speech.py' in part.replace('\\','/')
                for part in (p.info['cmdline'] or [])):
                found.append(p.info['pid'])
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    return found


def run(script,*arguments):
    status(script,arguments=list(arguments))
    with log_path.open('a',encoding='utf-8') as log:
        log.write(f'\nRUN {script} {arguments}\n');log.flush()
        completed=subprocess.run([sys.executable,str(HERE/script),*arguments],cwd=ROOT,
                                  stdout=log,stderr=subprocess.STDOUT,env=env)
    if completed.returncode:
        raise RuntimeError(f'{script} failed ({completed.returncode}); see continuation.log')


try:
    # A live pre-existing process owns extraction; never start a competing extractor.
    while True:
        active=active_extractors()
        if not active:break
        progress=json.loads((HERE/'train_speech_progress.json').read_text())
        status('waiting_for_existing_training_extraction',completed=progress['completed'],total=769)
        time.sleep(30)
    index_path=HERE/'train_speech_index.csv'
    if not index_path.exists() or len(pd.read_csv(index_path))!=769:
        run('extract_speech.py')
    if not (HERE/'train_combined_features.csv').exists():
        run('build_features.py')
    if not (HERE/'comparison_report.json').exists():
        run('evaluate.py')
    report=json.loads((HERE/'comparison_report.json').read_text())
    if report['meaningful_under_predeclared_rule']:
        test_index=HERE/'test_speech_index.csv'
        if not test_index.exists() or len(pd.read_csv(test_index))!=216:
            run('extract_speech.py','--split','test')
        if not (HERE/'test_combined_features.csv').exists():
            run('build_features.py','--split','test')
    if not (HERE/'final_summary.json').exists():
        run('finalize.py')
    summary=json.loads((HERE/'final_summary.json').read_text())
    status('complete',decision='USE NEW MODEL' if summary['recommend_replacing_submission'] else 'KEEP ORIGINAL',
           final_summary='experiments/improvement_v1/final_summary.json')
except Exception as exc:
    status('error',error=str(exc))
    raise
