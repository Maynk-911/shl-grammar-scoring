"""Write a candidate only after a meaningful nested-CV gain; preserve every baseline artifact."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluate import pipeline

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def verify_baseline():
    manifest=json.loads((HERE/'baseline_snapshot.json').read_text())
    changed=[p for p,h in manifest.items() if not (ROOT/p).is_file()
             or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError(f'Baseline preservation failure: {changed}')
    return len(manifest)


def main():
    count=verify_baseline()
    report=json.loads((HERE/'comparison_report.json').read_text())
    selection=report['final_selection']
    description=' + '.join(f'{w:.2f} {name}' for name,w in zip(selection['members'],selection['weights']))
    output={'old_matched_5fold':report['baseline_5fold'],
            'historical_old_holdout':report['baseline_holdout_context'],
            'new_nested_5fold':report['new_nested_5fold'],
            'best_model':description,
            'improvement_meaningful':report['meaningful_under_predeclared_rule'],
            'paired_folds_improving_both':report['paired_folds_improving_both'],
            'recommend_replacing_submission':report['meaningful_under_predeclared_rule'],
            'baseline_artifacts_unchanged':count}
    if report['meaningful_under_predeclared_rule']:
        train=pd.read_csv(ROOT/'Dataset_Final/train.csv')
        test=pd.read_csv(ROOT/'Dataset_Final/test.csv')
        original=pd.read_csv(ROOT/'features/audio_features.csv')
        def matrix(split,ids):
            combined=pd.read_csv(HERE/f'{split}_combined_features.csv').set_index('filename').loc[ids].reset_index(drop=True)
            audio=original.loc[original.split.eq(split)].set_index('filename').loc[ids].drop(columns='split').reset_index(drop=True)
            return pd.concat([audio,combined[[c for c in combined if c not in audio.columns]]],axis=1)
        X=matrix('train',train.filename)
        X_test=matrix('test',test.filename)
        assert X.columns.equals(X_test.columns)
        y=train.label.to_numpy(float)
        models=[];pred=np.zeros(len(test))
        with threadpool_limits(limits=4):
            for name,weight in zip(selection['members'],selection['weights']):
                estimator=pipeline(X,name)
                estimator.fit(X,y)
                pred+=weight*np.clip(estimator.predict(X_test),0,5)
                models.append(estimator)
        pred=np.clip(pred,0,5)
        assert np.isfinite(pred).all()
        submission=pd.DataFrame({'filename':test.filename,'label':pred})
        submission.to_csv(HERE/'submission.csv',index=False)
        reread=pd.read_csv(HERE/'submission.csv')
        assert len(reread)==216 and reread.filename.tolist()==test.filename.tolist()
        assert list(reread.columns)==['filename','label'] and reread.label.between(0,5).all()
        joblib.dump({'models':models,'weights':selection['weights'],'names':selection['members'],
                     'features':X.columns.tolist(),'bounds':(0,5),
                     'speech_config':json.loads((HERE/'speech_config.json').read_text())},HERE/'candidate_model.joblib')
        output['candidate_submission']='experiments/improvement_v1/submission.csv'
        output['candidate_sha256']=hashlib.sha256((HERE/'submission.csv').read_bytes()).hexdigest()
        original_submission=pd.read_csv(ROOT/'submission.csv')
        assert original_submission.filename.tolist()==reread.filename.tolist()
        changes=reread.label.to_numpy()-original_submission.label.to_numpy()
        comparison={'rows':len(reread),'columns':reread.columns.tolist(),
                    'test_order_matches':True,'all_predictions_finite':bool(np.isfinite(reread.label).all()),
                    'prediction_range':[float(reread.label.min()),float(reread.label.max())],
                    'mean_absolute_prediction_change':float(np.mean(np.abs(changes))),
                    'max_absolute_prediction_change':float(np.max(np.abs(changes))),
                    'mean_signed_prediction_change':float(np.mean(changes)),
                    'changed_rows':int(np.sum(np.abs(changes)>1e-12)),
                    'old_new_prediction_correlation':float(np.corrcoef(original_submission.label,reread.label)[0,1]),
                    'original_submission_sha256':hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest(),
                    'new_submission_sha256':output['candidate_sha256']}
        (HERE/'submission_comparison.json').write_text(json.dumps(comparison,indent=2))
    else:
        output['candidate_submission']=None
        output['reason']='Predeclared paired consistency/significance threshold not met; retain verified baseline.'
    verify_baseline()
    (HERE/'final_summary.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
