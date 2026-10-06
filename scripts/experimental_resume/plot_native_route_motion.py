"""Publication-ready descriptive native route motion and expert exposure figure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1/dense-discovery')
MOTION = ROOT / 'fixed-window-routes-84a85c92-363dbe1b'
EXPOSURE = ROOT / 'route-profiles-84a85c92-40c3d3b0/NATIVE_ROUTE_SUMMARY.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(path.read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('changed figure input seal: ' + str(path))
    return value


def main():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
        raise RuntimeError('paper routing figure requires CPU Slurm')
    motion, exposure = sealed(MOTION / 'SUMMARY.json'), sealed(EXPOSURE)
    if motion['schema'] != 'native-fixed-window-motion-v1' or exposure['schema'] != 'dense-discovery-native-route-summary-v1':
        raise ValueError('not the expected saved native discovery measurements')
    if motion['families'] != 48 or exposure['families'] != 48 or motion['binding_sha256'] != sealed(MOTION / 'BINDING.json')['sha256']:
        raise ValueError('native route summary incomplete or rebound')
    speed = np.asarray(motion['equal_family_layer_profiles']['mean_gate_velocity_by_layer'], float)
    acceleration = np.asarray(motion['equal_family_layer_profiles']['mean_gate_acceleration_by_layer'], float)
    turnover = np.asarray(motion['equal_family_layer_profiles']['mean_top8_turnover_by_layer'], float)
    use = np.asarray(exposure['native_expert_selection_exposure'], float)
    if (speed.shape != acceleration.shape or speed.shape != turnover.shape or speed.shape != (40,) or
        use.shape != (40, 256) or not np.isfinite(use).all() or not np.isfinite(speed).all() or
        not np.isfinite(acceleration).all() or not np.isfinite(turnover).all()):
        raise ValueError('routing figure profiles have unexpected dimensions or missing values')
    if (MOTION / 'ROUTING_MOTION_FIGURE.json').exists():
        raise FileExistsError('figure receipt already exists; do not overwrite a prior output')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layers = np.arange(40)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    for ax, values, title, ylabel in (
        (axes[0, 0], speed, 'Gate-distribution velocity', 'TV per 64-token step'),
        (axes[0, 1], acceleration, 'Gate-distribution acceleration', 'Second-difference L1 / 2'),
        (axes[1, 0], turnover, 'Frequent-expert turnover', 'Top-eight set turnover'),
    ):
        ax.plot(layers, values, color='#285780', linewidth=1.8)
        ax.set(xlim=(0, 39), xlabel='Routed layer', ylabel=ylabel, title=title)
        ax.grid(color='#d7dde2', linewidth=.5)
    image = axes[1, 1].imshow(use, aspect='auto', origin='lower', interpolation='nearest',
                              cmap='viridis', vmin=0, vmax=float(np.percentile(use, 99.5)))
    axes[1, 1].set(title='Native expert selection exposure', xlabel='Expert ID', ylabel='Routed layer')
    fig.colorbar(image, ax=axes[1, 1], label='Selection probability per token')
    fig.suptitle('Discovery native routing: 48 equal-weighted families', fontsize=13)
    png, pdf = MOTION / 'ROUTING_MOTION.png', MOTION / 'ROUTING_MOTION.pdf'
    fig.savefig(png, dpi=210)
    fig.savefig(pdf)
    plt.close(fig)
    body = {'schema': 'native-routing-motion-figure-v1', 'job_id': os.environ['SLURM_JOB_ID'],
            'motion_summary_sha256': motion['sha256'], 'exposure_summary_sha256': exposure['sha256'],
            'driver_sha256': sha(__file__), 'png': str(png), 'png_sha256': sha(png),
            'pdf': str(pdf), 'pdf_sha256': sha(pdf),
            'caption': ('Exact nonoverlapping 64-reasoning-token windows of native Qwen3.6 routes. '
                        'Layer curves average within each of 48 discovery families, then equally across families; '
                        'the heatmap is native top-eight selection exposure. All panels are observational; '
                        'they do not identify a causal sequence or semantic steering effect.')}
    (MOTION / 'ROUTING_MOTION_FIGURE.json').write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    print(json.dumps({'png': str(png), 'pdf': str(pdf), 'families': 48}), flush=True)


if __name__ == '__main__':
    main()
