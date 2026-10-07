# Isolated improvement experiment

The existing solution is frozen. `baseline_snapshot.json` records SHA-256 hashes of 1,073 existing
artifacts, and `baseline_backup/` contains a second copy of the verified runnable archive. All new
files stay in this experiment directory. No original model, notebook, feature cache or CSV is overwritten.

## Reused and new information

Reuse all 177 existing audio descriptors. They already include amplitude-threshold pauses,
voiced-frame ratio, autocorrelation pitch, activity-onset rate, energy, spectral summaries and MFCCs.
Do not recompute those descriptors. Add ASR-derived lexical, sentence, repetition and speech-rate
estimates, plus neural-VAD silence and ASR confidence measures. Sentence boundaries are inferred by ASR,
not human annotations; speaking-rate estimates depend on recognition accuracy.

Pool the frozen Whisper encoder output from the SAME inference pass used for transcription.
Exclude encoder padding using chunk durations. Cache each transcript and embedding by audio content
and configuration. Do not fine-tune pretrained weights or use external examples or labels.
Embeddings are reduced to 32 components by PCA fitted inside each training fold.

The cached small model was tested on three full recordings: mean 12.26 seconds/recording,
about 2.6 hours for 769 training clips. A smaller pretrained tiny.en model is investigated to reduce
CPU and memory cost. That optional download stalled and was stopped without changing the baseline.
The fallback uses the already-cached small model on complete recordings, with six CPU threads and
batch size two (one-clip pilot: 9.24 seconds vs 12.16 seconds with the first settings). No cropped
transcripts or external data are used. The optional smaller model is available from
https://huggingface.co/Systran/faster-whisper-tiny.en; implementation is the installed faster-whisper.
Test feature extraction is deferred until there is evidence to justify a candidate submission.

## Evaluation fixed before new results

Five target-quantile-stratified outer folds, shuffle=True, seed=42, covering all 769 labeled clips.
Re-evaluate the OLD ExtraTrees configuration on these exact folds for a fair paired comparison.
The previous 154-row holdout score is historical context, not the comparison denominator.

Eight predeclared configurations: old audio-only ExtraTrees; audio+linguistic ExtraTrees/HGB;
combined features with ExtraTrees leaf sizes 3/5, HGB leaf limits 10/15, and one sklearn
GradientBoostingRegressor. XGBoost, LightGBM and CatBoost are not installed. No broad parameter search.

Within EACH outer training fold, three inner folds choose the feature/model configuration and optionally
blend the top two candidates at weights 0.25, 0.50 or 0.75. A blend is selected only if inner OOF
Pearson improves by more than 0.002 AND RMSE drops by more than 0.002. Outer-fold predictions are never
used to select their own model or weights. Report fold mean/sample standard deviation for RMSE and
Pearson, pooled OOF scores, paired fold differences, and the individual model/ensemble choices.

Recommend replacement only when BOTH metrics improve in at least 4/5 outer folds and approximate
corrected-CV 95% t intervals exclude zero in the favorable direction for BOTH paired metric differences.
Correction uses variance multiplier (1/5 + 1/4) and 4 degrees of freedom to acknowledge overlapping
training sets. This is conservative but still approximate; it is not a substitute for independent
speaker-disjoint data. Never claim training-set metrics are validation.

If replacement is justified, fit the selection chosen from all-training OOF results and write a NEW
model and submission only here. The original submission remains intact in all cases.

## Resumption scope (user steering)

The original extraction process was still active; it was not restarted. Valid cached transcript/embedding pairs are reused. No further model download is permitted. Compare exactly two regressors (ExtraTrees and HistGradientBoosting), each on audio, audio+linguistic, audio+encoder, and all features, using the existing five outer folds. Drop the unrun leaf-size and third-regressor experiments. Test blend weights only if the top-two inner-OOF residual correlation is below 0.98 and the runner-up RMSE is within 5% of the leader. Existing outer baseline predictions are reused verbatim. Retain inner-fold selection and the predeclared acceptance rule.
