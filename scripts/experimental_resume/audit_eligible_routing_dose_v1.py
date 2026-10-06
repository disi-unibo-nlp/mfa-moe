"""Summarize assigned first-stage routing dose without looking at semantic ratings."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import run_boundary_micro_screen as base


ROOT = Path('/leonardo_work/IscrC_MIOSR/lmolfett/tools/tmp/'
            'claude-analysis-2026-09-24/steering-v1/runs/routing-control-v1')
DOC = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo/report/experimental-resume-v1')
RUNS = (
    ('positive', ROOT / 'blind-eligible-micro-v3-320ecca116c7ab63/ARM_MAP.json',
     DOC / 'CAUSAL_ELIGIBLE_MICRO_SERIAL_MANIFEST_v3.json'),
    ('deactivation', ROOT / 'blind-eligible-deactivation-v4-85b7f90ae7295d85/ARM_MAP.json',
     DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_SERIAL_MANIFEST_v4.json'),
)
FIELDS = ('active_rows', 'actual_target_hits', 'native_target_hits',
          'membership_changes', 'target_mass', 'weight_l1')


def one_record(record, kind):
    source = record['action_dose'] if kind == 'positive' else record['base_hook_dose']
    if record['role'] == 'native':
        if kind == 'positive' and source:
            raise ValueError('native request unexpectedly has ordered action dose')
        if kind == 'deactivation' and (set(source) != {'0', '1'} or any(
                entry.get('policy') != 'zero' or entry.get('cpu_active_rows') != 0 or
                entry.get('cpu_pulse_rows') != 0 or any(
                    any(float(layer[field]) != 0 for field in FIELDS)
                    for layer in entry.get('counters', {}).values())
                for entry in source.values())):
            raise ValueError('native request unexpectedly has edited base-hook dose')
        return None
    if set(source) != {'0', '1'}:
        raise ValueError('edited request lacks both tensor-parallel rank receipts')
    if kind == 'positive':
        rank0, rank1 = source['0']['dose'], source['1']['dose']
        if rank0 != rank1:
            raise ValueError('tensor-parallel ordered dose differs')
        if len(rank0) != 1:
            raise ValueError('one active ordered policy required')
        layers = next(iter(rank0.values()))
    else:
        rank0, rank1 = source['0']['counters'], source['1']['counters']
        if rank0 != rank1:
            raise ValueError('tensor-parallel base-hook dose differs')
        layers = rank0
    if not layers:
        raise ValueError('edited request has no routed layer dose')
    totals = {field: 0. for field in FIELDS}
    for layer in layers.values():
        if any(field not in layer or type(layer[field]) not in (int, float)
               for field in FIELDS):
            raise ValueError('incomplete routing first-stage counter')
        for field in FIELDS:
            totals[field] += float(layer[field])
    if totals['active_rows'] <= 0:
        raise ValueError('edited request has no active routed rows')
    totals['membership_change_rate'] = totals['membership_changes'] / totals['active_rows']
    totals['target_hit_delta_rate'] = (
        totals['actual_target_hits'] - totals['native_target_hits']) / totals['active_rows']
    totals['target_mass_per_active_row'] = totals['target_mass'] / totals['active_rows']
    totals['weight_l1_per_active_row'] = totals['weight_l1'] / totals['active_rows']
    return totals


def summarize():
    results = []
    for kind, path, manifest_path in RUNS:
        arm_map, manifest = base.sealed(path), base.sealed(manifest_path)
        rows = arm_map['records']
        if (arm_map['manifest_sha256'] != manifest['sha256'] or
                len(rows) != manifest['expected_requests'] or
                len({row['uid'] for row in rows}) != len(rows)):
            raise ValueError('blind-map assignment seal or count differs')
        groups = defaultdict(list)
        for record in rows:
            groups[record['transition'], record['arm']].append(
                (record, one_record(record, kind)))
        table = []
        for (transition, arm), records in sorted(groups.items()):
            if len(records) not in (10, 16):
                raise ValueError('arm/transition cell is incomplete')
            doses = [dose for _, dose in records if dose is not None]
            means = ({key: sum(dose[key] for dose in doses) / len(doses)
                      for key in doses[0]} if doses else {})
            table.append({'transition': transition, 'arm': arm, 'assigned': len(records),
                          'dose_receipts': len(doses), 'mean_dose': means,
                          'mean_emitted_tokens': sum(r['emitted_tokens'] for r, _ in records) / len(records),
                          'reasoning_closed': sum(r['closed_reasoning'] for r, _ in records),
                          'measurement_statuses': sorted({r['measurement_status'] for r, _ in records})})
        results.append({'kind': kind, 'manifest_sha256': manifest['sha256'],
                        'arm_map_sha256': arm_map['sha256'], 'rows': len(rows),
                        'table': table})
    body = {'schema': 'eligible-routing-first-stage-audit-v1', 'runs': results,
            'interpretation': 'Descriptive assigned-request routing dose only; no semantic or utility effect is inferred.'}
    return {**body, 'sha256': base.digest(body)}


def main():
    out = DOC / 'CAUSAL_ELIGIBLE_ROUTING_FIRST_STAGE_AUDIT_v1.json'
    result = summarize()
    if out.exists():
        if base.sealed(out) != result:
            raise ValueError('existing first-stage audit differs')
    else:
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'out': str(out), 'sha256': result['sha256'],
                      'cells': [len(run['table']) for run in result['runs']]}))


if __name__ == '__main__':
    main()
