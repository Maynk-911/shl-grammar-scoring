"""Paired five-fold evaluation, with feature/model/ensemble selection nested inside folds."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEED=42
CANDIDATES = {
    'audio_et': ('audio','et3'),
    'audio_hgb': ('audio','hgb10'),
    'audio_ling_et': ('audio_ling','et3'),
    'audio_ling_hgb': ('audio_ling','hgb10'),
    'audio_emb_et': ('audio_emb','et3'),
    'audio_emb_hgb': ('audio_emb','hgb10'),
    'all_et': ('all','et3'),
    'all_hgb': ('all','hgb10'),
}


def score(y,p):
    p=np.clip(p,0,5)
    return {'RMSE':float(np.sqrt(np.mean((np.asarray(y)-p)**2))),
            'Pearson':float(pearsonr(y,p).statistic) if np.std(p)>1e-10 else 0.}


def selected_columns(X,name):
    group,_=CANDIDATES[name]
    audio=[c for c in X.columns if not c.startswith(('ling_','vad_','asr_','emb_'))]
    linguistic=[c for c in X.columns if c.startswith(('ling_','vad_','asr_'))]
    embeddings=[c for c in X.columns if c.startswith('emb_')]
    return audio + (linguistic if group in ['audio_ling','all'] else []) + (embeddings if group in ['audio_emb','all'] else [])


def pipeline(X,name):
    cols=selected_columns(X,name)
    embedding=[c for c in cols if c.startswith('emb_')]
    numeric=[c for c in cols if c not in embedding]
    transforms=[('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                                         VarianceThreshold(),StandardScaler()),numeric)]
    if embedding:
        transforms.append(('embeddings',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                            PCA(n_components=32,svd_solver='randomized',random_state=SEED)),embedding))
    _,kind=CANDIDATES[name]
    if kind.startswith('et'):
        model=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=int(kind[-1]),
                                  max_features=.8,random_state=SEED,n_jobs=4)
    elif kind.startswith('hgb'):
        leaves=int(kind[3:])
        model=HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=leaves,min_samples_leaf=20,
                    learning_rate=.05,l2_regularization=5. if leaves==10 else 10.,
                    early_stopping=True,random_state=SEED)
    else:
        raise ValueError(f'Unsupported model: {kind}')
    return make_pipeline(ColumnTransformer(transforms),model)


def cached_predict(X,y,name,fit,valid):
    if name=='audio_et' and (HERE/'baseline_oof.csv').exists():
        baseline=pd.read_csv(HERE/'baseline_oof.csv')
        # Reuse the completed control only for its EXACT outer train/validation partition.
        valid_folds=baseline.iloc[valid].fold.unique()
        if len(valid_folds)==1:
            fold=valid_folds[0]
            if (np.array_equal(np.sort(valid),np.flatnonzero(baseline.fold.to_numpy()==fold))
                and np.array_equal(np.sort(fit),np.flatnonzero(baseline.fold.to_numpy()!=fold))):
                assert np.array_equal(baseline.label.to_numpy(),y)
                return baseline.iloc[valid].audio_et.to_numpy()
    cols=selected_columns(X,name)
    descriptor={'pipeline_version':1,'model':name,'seed':SEED,
                'fit':np.asarray(fit).tolist(),'valid':np.asarray(valid).tolist()}
    digest=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(X[cols],index=True).to_numpy().tobytes())
    digest.update(np.asarray(y,dtype=np.float64).tobytes())
    cache=HERE/'cache/cv'; cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'{digest.hexdigest()}.npy'
    if path.exists(): return np.load(path,allow_pickle=False)
    fitted=pipeline(X,name)
    with threadpool_limits(limits=4):
        fitted.fit(X.iloc[fit],y[fit])
        pred=np.clip(fitted.predict(X.iloc[valid]),0,5)
    np.save(path,pred,allow_pickle=False)
    return pred


def choose(y,predictions):
    scores={name:score(y,p) for name,p in predictions.items()}
    ranked=sorted(scores,key=lambda n:(-scores[n]['Pearson'],scores[n]['RMSE']))
    # All choices, including blend weights, use only inner OOF data.
    first=ranked[0]
    chosen={'members':[first],'weights':[1.], 'selection_metrics':scores[first]}
    # Only the top two candidates; three blend weights, no continuous optimization.
    second=ranked[1]
    trials=[]
    error_correlation=float(np.corrcoef(y-predictions[first],y-predictions[second])[0,1])
    # Only investigate blends if errors are complementary and the runner-up is competitive.
    eligible=(np.isfinite(error_correlation) and error_correlation<.98
              and scores[second]['RMSE']<=scores[first]['RMSE']*1.05)
    for weight in ([.25,.5,.75] if eligible else []):
        blend=weight*predictions[first]+(1-weight)*predictions[second]
        metrics=score(y,blend)
        trials.append({'members':[first,second],'weights':[weight,1-weight],**metrics})
        previous=chosen['selection_metrics']
        if metrics['Pearson']>previous['Pearson']+.002 and metrics['RMSE']<previous['RMSE']-.002:
            chosen={'members':[first,second],'weights':[weight,1-weight],'selection_metrics':metrics}
    chosen['inner_model_scores']=scores
    chosen['ensemble_trials']=trials
    chosen['ensemble_eligible']=bool(eligible)
    chosen['top_two_error_correlation']=error_correlation
    return chosen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    train=pd.read_csv(ROOT/'Dataset_Final/train.csv');y=train.label.to_numpy(float)
    if args.baseline_only:
        f=pd.read_csv(ROOT/'features/audio_features.csv')
        X=f.query("split == 'train'").set_index('filename').loc[train.filename].drop(columns='split').reset_index(drop=True)
    else:
        combined=pd.read_csv(HERE/'train_combined_features.csv').set_index('filename').loc[train.filename].reset_index(drop=True)
        original=pd.read_csv(ROOT/'features/audio_features.csv')
        original=original.query("split == 'train'").set_index('filename').loc[train.filename].drop(columns='split').reset_index(drop=True)
        # Preserve exact cached acoustic values instead of another CSV round trip.
        X=pd.concat([original,combined[[c for c in combined if c not in original.columns]]],axis=1)
    bins=pd.qcut(pd.Series(y),q=5,labels=False,duplicates='drop').to_numpy()
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,bins))
    fold_ids=np.zeros(len(y),dtype=int)
    names=['audio_et'] if args.baseline_only else list(CANDIDATES)
    predictions={name:np.full(len(y),np.nan) for name in names}
    nested=np.full(len(y),np.nan)
    rows=[];selections=[]
    for fold,(fit,valid) in enumerate(folds):
        fold_ids[valid]=fold
        for name in names:
            pred=cached_predict(X,y,name,fit,valid)
            predictions[name][valid]=pred
            rows.append({'model':name,'fold':fold,**score(y[valid],pred)})
        if not args.baseline_only:
            inner={name:np.full(len(fit),np.nan) for name in names}
            inner_folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED+fold+1).split(fit,bins[fit]))
            for inner_number,(a,b) in enumerate(inner_folds):
                for name in names:
                    inner[name][b]=cached_predict(X,y,name,fit[a],fit[b])
                print(f'Outer fold {fold+1}/5: inner fold {inner_number+1}/3 complete',flush=True)
            selection=choose(y[fit],inner)
            nested[valid]=sum(w*predictions[n][valid] for n,w in zip(selection['members'],selection['weights']))
            selections.append({'fold':fold,**selection})
            rows.append({'model':'nested_selection','fold':fold,**score(y[valid],nested[valid])})
            (HERE/'nested_fold_selections.json').write_text(json.dumps(selections,indent=2))
        pd.DataFrame(rows).to_csv(HERE/('baseline_folds.csv' if args.baseline_only else 'cv_fold_results.csv'),index=False)
        pd.DataFrame({'filename':train.filename,'label':y,'fold':fold_ids,**predictions,
                      **({'nested_selection':nested} if not args.baseline_only else {})}).to_csv(
            HERE/('baseline_oof.csv' if args.baseline_only else 'oof_predictions.csv'),index=False)
        print(f'Completed outer fold {fold+1}/5',flush=True)
    table=pd.DataFrame(rows)
    summary=table.groupby('model')[['RMSE','Pearson']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(HERE/('baseline_cv_summary.csv' if args.baseline_only else 'cv_summary.csv'))
    print(summary.round(5).to_string(),flush=True)
    if args.baseline_only:return
    final_selection=choose(y,predictions)
    (HERE/'final_selection.json').write_text(json.dumps(final_selection,indent=2))
    baseline=table[table.model=='audio_et'].set_index('fold')
    improved=table[table.model=='nested_selection'].set_index('fold')
    difference_rmse=improved.RMSE-baseline.RMSE
    difference_rho=improved.Pearson-baseline.Pearson
    def corrected_interval(d):
        # Conservative paired corrected-CV t interval: correlated training sets, df=4.
        half=t.ppf(.975,4)*np.sqrt((1/5+1/4)*np.var(d,ddof=1))
        return [float(np.mean(d)-half),float(np.mean(d)+half)]
    ci_rmse=corrected_interval(difference_rmse)
    ci_rho=corrected_interval(difference_rho)
    wins=int(((difference_rmse<0)&(difference_rho>0)).sum())
    meaningful=bool(wins>=4 and ci_rmse[1]<0 and ci_rho[0]>0)
    report={'baseline_5fold':summary.loc['audio_et'].to_dict(),
            'new_nested_5fold':summary.loc['nested_selection'].to_dict(),
            'pooled_baseline':score(y,predictions['audio_et']), 'pooled_nested':score(y,nested),
            'paired_folds_improving_both':wins,'fold_count':5,
            'corrected_95CI_delta_RMSE':ci_rmse,'corrected_95CI_delta_Pearson':ci_rho,
            'meaningful_under_predeclared_rule':meaningful,'final_selection':final_selection,
            'limitations':'Corrected five-fold t intervals are approximate; no speaker IDs and no independent new dataset.',
            'baseline_holdout_context':{'RMSE':.6891117612699722,'Pearson':.8384886724804615}}
    (HERE/'comparison_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
