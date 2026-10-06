"""Plot seven-class native discovery dynamics from the sealed dense LLM audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

BASE = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery/results-84a85c92-b2633756')
SOURCE = BASE / 'CLASS_DYNAMICS.json'
LABEL_SUMMARY = BASE / 'SUMMARY.json'
CLASSES = ('Read', 'Analyze', 'Plan', 'Implement', 'Explore', 'Verify', 'Monitor')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed dense class figure source seal: ' + str(path))
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('dense discovery class figure requires CPU Slurm')
    data, label = sealed(SOURCE), sealed(LABEL_SUMMARY)
    if (data['schema'] != 'dense-discovery-class-summary-v1' or data['families'] != 48 or
        data['label_summary_sha256'] != label['sha256']):
        raise ValueError('dense class audit incomplete or rebound')
    counts = np.array([data['class_counts'].get(name, 0) for name in CLASSES], float)
    raw = np.asarray(data['transition_counts'], int)
    provided = data['transition_probabilities']
    if raw.shape != (7, 7) or len(provided) != 7 or any(len(row) != 7 for row in provided):
        raise ValueError('dense class transition matrix differs')
    matrix = np.zeros((7, 7), float)
    for i in range(7):
        if raw[i].sum():
            matrix[i] = raw[i] / raw[i].sum()
            agrees = np.allclose(np.asarray(provided[i], float), matrix[i], atol=1e-12)
        else:
            agrees = all(value is None for value in provided[i])
        if not agrees:
            raise ValueError('dense transition count/probability disagreement')
    dwell = [data['dwell_sentences_observed_including_censoring'].get(name, []) for name in CLASSES]
    medians = [float(np.median(values)) if values else 0. for values in dwell]
    loops = sorted(data['loop_counts'].items(), key=lambda item: (-item[1], item[0]))[:8]
    if (BASE / 'CLASS_DYNAMICS_FIGURE.json').exists():
        raise FileExistsError('dense discovery class figure receipt already exists')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    im = axes[0, 0].imshow(matrix, vmin=0, vmax=1, cmap='Blues', interpolation='nearest')
    axes[0, 0].set(xticks=range(7), yticks=range(7), xticklabels=CLASSES,
                   yticklabels=CLASSES, xlabel='Next classified sentence',
                   ylabel='Current classified sentence', title='Adjacent-class transition probability')
    axes[0, 0].tick_params(axis='x', labelrotation=45)
    fig.colorbar(im, ax=axes[0, 0], label='Conditional proportion')
    axes[0, 1].bar(CLASSES, counts, color='#376a97')
    axes[0, 1].set(title='Classified sentences', ylabel='Count')
    axes[0, 1].tick_params(axis='x', labelrotation=45)
    axes[1, 0].bar(CLASSES, medians, color='#649783')
    axes[1, 0].set(title='Observed dwell (boundary-censored)', ylabel='Median sentence run')
    axes[1, 0].tick_params(axis='x', labelrotation=45)
    axes[1, 1].barh([key.replace('|', '→') for key, _ in reversed(loops)],
                    [count for _, count in reversed(loops)], color='#a66a57')
    axes[1, 1].set(title='Most frequent three-sentence re-entries', xlabel='Observed count')
    fig.suptitle('Direct-LLM class audit of 48 contiguous native discovery traces', fontsize=13)
    png, pdf = BASE / 'CLASS_DYNAMICS.png', BASE / 'CLASS_DYNAMICS.pdf'
    fig.savefig(png, dpi=210)
    fig.savefig(pdf)
    plt.close(fig)
    body = {'schema': 'dense-discovery-class-figure-v1',
            'job_id': os.environ['SLURM_JOB_ID'], 'class_summary_sha256': data['sha256'],
            'label_summary_sha256': label['sha256'], 'driver_sha256': sha(__file__),
            'png': str(png), 'png_sha256': sha(png), 'pdf': str(pdf), 'pdf_sha256': sha(pdf),
            'caption': ('Adjacent observed sentence classes, count distribution, boundary-censored dwell, '
                        'and three-sentence re-entries in 48 frozen native discovery families. '
                        'Direct same-model LLM class labels are exploratory and cannot alone establish '
                        'substantive verification, semantic control, or an optimal reasoning sequence.')}
    (BASE / 'CLASS_DYNAMICS_FIGURE.json').write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    print(json.dumps({'png': str(png), 'pdf': str(pdf), 'families': 48}), flush=True)


if __name__ == '__main__':
    main()
