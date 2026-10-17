"""Build lexical/native_lfm25.parquet from the LFM discovery judge labels and score it with the frozen
7-model lexical readout (cross-fitted by fold) to measure transfer to LFM text.

Usage (CPU job): python scripts/study/lfm_lexical_table.py
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from moe_exp.study.common import CLASSES, Units, file_sha, publish, verified
from moe_exp.study.driver import ROOT, binding
from moe_exp.study.lexical import LEXICAL, predict


def main():
    b = binding()
    bulk = Path(b['config']['bulk_root'])
    tasks = {t['uid']: t for t in verified(ROOT/'lfm25/ANNOTATION_TASKS.json')['records']}
    sampled = verified(bulk/'lfm25/SAMPLED_SENTENCES.json')['records']
    labels = Units(bulk/'reader/annotations')
    duplicates = Counter(r['annotation_uid'] for r in sampled)
    rows = []
    for r in sampled:
        state = labels.state(r['annotation_uid']) or {}
        inputs = tasks[r['annotation_uid']]['inputs']
        rows.append({'model': 'lfm25', 'context_hash': r['annotation_uid'], 'n_rows_with_context': duplicates[r['annotation_uid']],
                     'label': state.get('label') if state.get('label') in CLASSES else None,
                     'sentence_text': inputs['sentence'], 'prev_sentence': inputs['previous_sentence'],
                     'next_sentence': inputs['next_sentence'], 'family_id': r['family'], 'fold_campaign': r['fold']})
    table = pd.DataFrame(rows)
    path = LEXICAL/'native_lfm25.parquet'
    table.to_parquet(path, index=False)
    scored = table[table.label.notna()].drop_duplicates('context_hash').rename(columns={'fold_campaign': 'fold'})
    scored = predict(scored, LEXICAL/'readout-v1')
    y = scored.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p = scored.lexical_label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    cm = np.bincount(y*7+p, minlength=49).reshape(7, 7)
    sens = np.diag(cm)/np.maximum(cm.sum(1), 1)
    fpr = (cm.sum(0)-np.diag(cm))/np.maximum(cm.sum()-cm.sum(1), 1)
    value = publish(LEXICAL/'readout-v1'/'TRANSFER_LFM25.json', {
        'table': str(path), 'table_sha256': file_sha(path), 'contexts': int(len(scored)),
        'accuracy': float((y == p).mean()),
        'macro_f1': float(np.mean([2*cm[i, i]/max(1, cm[i].sum()+cm[:, i].sum()) for i in range(7)])),
        'confusion': cm.tolist(), 'youden_j': dict(zip(CLASSES, (sens-fpr).tolist())),
        'label_shares': dict(zip(CLASSES, (np.bincount(y, minlength=7)/len(y)).tolist())),
        'note': 'frozen 7-model readout-v1, cross-fitted by family fold; LFM text unseen in training'})
    print(json.dumps({k: v for k, v in value.items() if k != 'confusion'}, indent=1))


if __name__ == '__main__':
    main()
