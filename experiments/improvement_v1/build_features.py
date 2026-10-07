"""Use cached acoustic features plus new transcript, VAD and frozen-encoder descriptors."""
from pathlib import Path
import collections
import json
import hashlib
import re
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def linguistic_features(record):
    text = record['text'].lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    count = len(words)
    safe_count = max(1, count)
    duration = max(record['duration'], 1e-6)
    speech = max(record['duration_after_vad'], 1e-6)
    counts = collections.Counter(words)
    sentences = [re.findall(r"[a-z]+(?:'[a-z]+)?", s) for s in re.split(r'[.!?]+', text)]
    lengths = [len(s) for s in sentences if s]
    bigrams = list(zip(words[:-1], words[1:]))
    segments = record['segments']
    features = {
        'ling_word_count': count,
        'ling_words_per_minute': 60*count/duration,
        'ling_words_per_speech_minute': 60*count/speech if count else 0.,
        'ling_type_token_ratio': len(counts)/safe_count,
        'ling_root_type_token_ratio': len(counts)/np.sqrt(safe_count),
        'ling_hapax_ratio': sum(n == 1 for n in counts.values())/safe_count,
        'ling_bigram_repetition': 1-len(set(bigrams))/len(bigrams) if bigrams else 0.,
        'ling_immediate_word_repetition': sum(a == b for a,b in bigrams)/max(1,len(bigrams)),
        'ling_mean_word_length': np.mean([len(w) for w in words]) if words else 0.,
        'ling_long_word_ratio': sum(len(w) >= 7 for w in words)/safe_count,
        'ling_sentence_count': len(lengths),
        'ling_words_per_sentence_mean': np.mean(lengths) if lengths else 0.,
        'ling_words_per_sentence_std': np.std(lengths) if lengths else 0.,
        'ling_words_per_sentence_max': max(lengths) if lengths else 0.,
        'ling_sentence_rate': 60*len(lengths)/duration,
        'ling_contraction_ratio': sum("'" in w for w in words)/safe_count,
        'ling_ing_word_ratio': sum(w.endswith('ing') for w in words)/safe_count,
        'ling_ed_word_ratio': sum(w.endswith('ed') for w in words)/safe_count,
        'vad_silence_ratio': 1-min(1.,record['duration_after_vad']/duration),
        'asr_empty': float(not words),
    }
    categories = {
        'fillers': 'um uh erm hmm ah',
        'pronouns': 'i me my mine you your yours he him his she her hers it its we us our ours they them their theirs',
        'articles': 'a an the',
        'prepositions': 'in on at by with from to into of for under over through between around during without',
        'conjunctions': 'and but or so yet nor',
        'subordinators': 'because although though while when if unless since whereas until before after',
        'auxiliaries': 'am is are was were be been being have has had do does did can could will would shall should may might must',
        'negation': 'not no never neither nobody nothing',
    }
    for name, lexicon in categories.items():
        features[f'ling_{name}_ratio'] = sum(counts[w] for w in lexicon.split())/safe_count
    for name in ['avg_logprob', 'no_speech_prob', 'compression_ratio']:
        values = [s[name] for s in segments]
        features[f'asr_{name}_mean'] = float(np.mean(values)) if values else np.nan
    # ASR/VAD gaps differ from the baseline's amplitude-threshold pause proxy.
    gaps = [max(0., b['start']-a['end']) for a,b in zip(segments[:-1],segments[1:])]
    features['asr_segment_gap_mean'] = float(np.mean(gaps)) if gaps else 0.
    features['asr_segment_gap_max'] = float(max(gaps)) if gaps else 0.
    features['asr_segments_per_minute'] = 60*len(segments)/duration
    return features


def build(split='train'):
    index = pd.read_csv(HERE/f'{split}_speech_index.csv')
    ids = pd.read_csv(ROOT/f'Dataset_Final/{split}.csv').filename
    assert index.filename.tolist() == ids.tolist(), 'Incomplete/out-of-order speech cache index'
    audio = pd.read_csv(ROOT/'features/audio_features.csv')
    audio = audio.loc[audio.split.eq(split)].set_index('filename').loc[ids].drop(columns='split')
    linguistic, embeddings = [], []
    config=json.loads((HERE/'speech_config.json').read_text())
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    audit=pd.read_csv(ROOT/'results/audio_audit.csv').query('split == @split').set_index('filename')
    for row in index.itertuples(index=False):
        path = HERE/'cache/speech'/f'{row.key}.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['filename'] == row.filename and record['split'] == split
        assert record['config_signature']==signature and record['audio_sha256']==audit.loc[row.filename,'sha256']
        linguistic.append(linguistic_features(record))
        embedding=np.load(path.with_suffix('.npy'), allow_pickle=False)
        assert embedding.shape==(768,) and np.isfinite(embedding).all()
        embeddings.append(embedding)
    linguistic = pd.DataFrame(linguistic, index=ids)
    embedding_array = np.asarray(embeddings)
    embedding_df = pd.DataFrame(embedding_array, index=ids,
                                columns=[f'emb_{i:03d}' for i in range(embedding_array.shape[1])])
    assert not set(audio.columns) & set(linguistic.columns)
    combined = pd.concat([audio, linguistic, embedding_df], axis=1)
    assert combined.index.tolist() == ids.tolist()
    combined.to_csv(HERE/f'{split}_combined_features.csv', index_label='filename')
    linguistic.to_csv(HERE/f'{split}_linguistic_features.csv', index_label='filename')
    np.savez_compressed(HERE/f'{split}_embeddings.npz', filenames=ids.to_numpy(dtype=str), embeddings=embedding_array)
    print(f'{split}: {audio.shape[1]} reused audio + {linguistic.shape[1]} linguistic/VAD + '
          f'{embedding_array.shape[1]} frozen encoder dimensions (fold-local PCA later)', flush=True)
    return combined


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--split',default='train',choices=['train','test'])
    build(parser.parse_args().split)
