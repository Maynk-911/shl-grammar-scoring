"""Ten inexpensive fits on completed caches only. No audio access, extraction or overwrites."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits
from build_features import linguistic_features
from evaluate import pipeline, score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'completed_cache_review_20261008'
OUT.mkdir(exist_ok=False)

config=json.loads((HERE/'speech_config.json').read_text())
signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
audit=pd.read_csv(ROOT/'results/audio_audit.csv')
available=[];extra=[];test_available=0
for row in audit.itertuples(index=False):
    key=hashlib.sha256(f'{row.sha256}:{signature}'.encode()).hexdigest()
    path=HERE/'cache/speech'/f'{key}.json'
    if not path.exists() or not path.with_suffix('.npy').exists():continue
    record=json.loads(path.read_text(encoding='utf-8'))
    embedding=np.load(path.with_suffix('.npy'),allow_pickle=False)
    assert record['audio_sha256']==row.sha256 and record['config_signature']==signature
    assert record['filename']==row.filename and record['split']==row.split
    assert embedding.shape==(768,) and np.isfinite(embedding).all()
    if row.split=='test':
        test_available+=1
        continue
    available.append(row.filename)
    extra.append({**linguistic_features(record),**{f'emb_{i:03d}':float(v) for i,v in enumerate(embedding)}})

old_oof=pd.read_csv(HERE/'baseline_oof.csv').set_index('filename').loc[available]
folds=old_oof.fold.to_numpy(int);y=old_oof.label.to_numpy(float)
audio=pd.read_csv(ROOT/'features/audio_features.csv')
audio=audio.loc[audio.split.eq('train')].set_index('filename').loc[available].drop(columns='split').reset_index(drop=True)
X=pd.concat([audio,pd.DataFrame(extra)],axis=1)
assert len(X)==len(y)==339 and len(np.unique(folds))==5
# Only cheap statistics are derived from already-saved transcripts. No audio is read.
X.to_csv(OUT/'completed_combined_features.csv',index=False)
pd.DataFrame({'filename':available,'label':y,'original_fold':folds}).to_csv(OUT/'sample_manifest.csv',index=False)
predictions={name:np.full(len(y),np.nan) for name in ['matched_audio','augmented_audio_linguistic_embeddings']}
rows=[];ratios=[]
for fold in range(5):
    fit=np.flatnonzero(folds!=fold);valid=np.flatnonzero(folds==fold)
    ratios.append(len(valid)/len(fit))
    for label,candidate in [('matched_audio','audio_et'),('augmented_audio_linguistic_embeddings','all_et')]:
        model=pipeline(X,candidate)
        with threadpool_limits(limits=4):
            model.fit(X.iloc[fit],y[fit])
            predicted=np.clip(model.predict(X.iloc[valid]),0,5)
        predictions[label][valid]=predicted
        rows.append({'model':label,'fold':fold,'fit_rows':len(fit),'validation_rows':len(valid),**score(y[valid],predicted)})
    print(f'Matched fold {fold+1}/5 complete',flush=True)

table=pd.DataFrame(rows)
table.to_csv(OUT/'fold_results.csv',index=False)
pd.DataFrame({'filename':available,'actual':y,'fold':folds,**predictions}).to_csv(OUT/'oof_predictions.csv',index=False)
summary=table.groupby('model')[['RMSE','Pearson']].agg(['mean','std'])
summary.columns=['_'.join(c) for c in summary.columns]
summary.to_csv(OUT/'cv_summary.csv')
control=table.query("model == 'matched_audio'").set_index('fold')
new=table.query("model == 'augmented_audio_linguistic_embeddings'").set_index('fold')
differences=new[['RMSE','Pearson']]-control[['RMSE','Pearson']]
def interval(values):
    width=t.ppf(.975,4)*np.sqrt((1/5+np.mean(ratios))*np.var(values,ddof=1))
    return [float(np.mean(values)-width),float(np.mean(values)+width)]
rmse_ci=interval(differences.RMSE);rho_ci=interval(differences.Pearson)
wins=int(((differences.RMSE<0)&(differences.Pearson>0)).sum())
meaningful=bool(wins>=4 and rmse_ci[1]<0 and rho_ci[0]>0)

baseline_snapshot=json.loads((HERE/'baseline_snapshot.json').read_text())
changed=[name for name,h in baseline_snapshot.items() if not (ROOT/name).is_file()
         or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
assert not changed,changed
full_baseline=pd.read_csv(HERE/'baseline_cv_summary.csv').iloc[0].to_dict()
report={
    'scope':'Completed-cache subset only; same original fold IDs, equal training/validation rows for both models.',
    'training_samples':len(y),'test_samples_with_complete_new_features':test_available,
    'full_769_sample_baseline':full_baseline,
    'matched_subset_audio_baseline':summary.loc['matched_audio'].to_dict(),
    'augmented_subset_extratrees':summary.loc['augmented_audio_linguistic_embeddings'].to_dict(),
    'new_minus_matched_baseline':{c:float(differences[c].mean()) for c in ['RMSE','Pearson']},
    'new_minus_full_baseline_noncomparable':{
        'RMSE':float(summary.loc['augmented_audio_linguistic_embeddings','RMSE_mean']-full_baseline['RMSE_mean']),
        'Pearson':float(summary.loc['augmented_audio_linguistic_embeddings','Pearson_mean']-full_baseline['Pearson_mean'])},
    'both_metrics_improved_folds':wins,
    'corrected_95CI_delta_RMSE':rmse_ci,'corrected_95CI_delta_Pearson':rho_ci,
    'subset_improvement_meaningful':meaningful,
    'limitations':'339 completed recordings are an incomplete subset, not a new full-data five-fold benchmark. Corrected-CV intervals are approximate.',
    'decision':'KEEP BASELINE',
    'decision_reason':('No statistically consistent matched-CV improvement.' if not meaningful else
                       'Promising subset evidence only; no completed test speech features and further extraction is prohibited.'),
    'recommended_submission':str((ROOT/'submission.csv').resolve()),
    'new_submission_created':False,'audio_processed':0,'new_models_downloaded':0,
    'additional_regressors_or_ensembles_tested':False,
    'protected_baseline_files_unchanged':len(baseline_snapshot),
}
(OUT/'decision.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(summary.round(6).to_string(),flush=True)
print(json.dumps(report,indent=2),flush=True)
