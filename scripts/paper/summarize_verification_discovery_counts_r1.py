"""Summarize existing discovery/audit closeouts without inference or dispatch."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from moe_exp.verification_study.contracts import file_sha, load, require, save

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'report/verification-study-v3'
MODELS = (
    ('gpt', 'GPT'), ('nemotron', 'NEMOTRON'), ('qwen35', 'QWEN35'),
    ('qwen36', 'QWEN36'), ('gemma', 'GEMMA'), ('glm', 'GLM'),
    ('qwen330b', 'QWEN330B'),
)


def main():
    rows = []
    inputs = {}

    def source(path):
        value = load(path)
        inputs[str(path.resolve())] = value['sha256']
        return value

    campaign = source(REPORT / 'CAMPAIGN.json')
    for model, prefix in MODELS:
        assignments = source(REPORT / model / 'DISCOVERY_ASSIGNMENTS.json')
        require(assignments['assigned'] == 48, 'discovery roster differs')
        row = {'model': model, 'planned_problem_families': 48,
               'planned_native_solutions': 48, 'planned_audit_records': 128,
               'planned_reader_outputs': 256,
               'closed_native_solutions': None, 'audit_records': None,
               'unique_prefix_positions': None, 'audit_families': None,
               'closed_reader_outputs': None, 'both_draws_usable_records': None,
               'eligible_confirmation': None, 'checked_confirmation': None,
               'eligible_apparent_families': None, 'checked_apparent_families': None,
               'eligible_held_out_families': None, 'checked_held_out_families': None,
               'recognition_status': 'NOT_CLOSED'}
        eligibility = REPORT / (prefix + '_ELIGIBILITY_CLOSEOUT-r1.json')
        discovery = REPORT / (prefix + '_DISCOVERY_ADVANCEMENT-r1.json')
        if eligibility.exists():
            result = source(eligibility)
            require(result['campaign_sha256'] == campaign['sha256'], 'foreign audit campaign')
            row.update(closed_native_solutions=result['original_discovery_assignments'],
                       audit_records=result['original_prefixes'],
                       closed_reader_outputs=result['original_draws'],
                       both_draws_usable_records=result['both_draws_parseable_prefixes'],
                       recognition_status=result['qualification_status'])
            frame_path = Path(result['frame_path'])
            expected_frame = result['frame_sha256']
            for decision, name in (('eligible', 'eligible'), ('already_checked', 'checked')):
                values = result['decisions'][decision]
                row[name + '_confirmation'] = values['apparent_family_precision']
                row[name + '_apparent_families'] = values['apparent_families']
                row[name + '_held_out_families'] = values['held_out_support']
        elif discovery.exists():
            result = source(discovery)
            require(result['campaign_sha256'] == campaign['sha256'], 'foreign discovery campaign')
            require(not result['missing_discovery_assignments'], 'incomplete discovery closeout')
            row.update(closed_native_solutions=result['original_assignments'],
                       audit_records=result['audit_prefixes'])
            frame_path = Path(result['audit_frame_path'])
            expected_frame = result['audit_frame_sha256']
        else:
            rows.append(row)
            continue
        frame = source(frame_path)
        require(frame['sha256'] == expected_frame and len(frame['records']) == row['audit_records'],
                'audit frame binding/count differs')
        row['unique_prefix_positions'] = len({(r['source_uid'], r['position']) for r in frame['records']})
        row['audit_families'] = len({r['family'] for r in frame['records']})
        rows.append(row)

    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    table = REPORT / 'DISCOVERY_AUDIT_COUNTS-r1.csv'
    with table.open('x', newline='') as output:
        output.write(stream.getvalue())
    require(table.read_text() == stream.getvalue().replace('\r\n', '\n'), 'CSV readback differs')
    receipt = save(REPORT / 'DISCOVERY_AUDIT_COUNTS-r1.json', {
        'schema': 'verification-discovery-audit-counts-r1',
        'campaign_sha256': campaign['sha256'], 'inputs': inputs,
        'reporting_source_sha256': file_sha(Path(__file__)),
        'table_path': str(table), 'table_file_sha256': file_sha(table), 'rows': rows,
        'independence': '48 shared problem families; not336 independent problems or896 independent prefixes',
        'audit_split': {'apparent_families': 36, 'held_out_families': 12},
        'reader_scope': 'Two fixed draws from one reader; not independent human annotations',
        'confirmation_scope': 'Family-weighted apparent reader confirmation; unknown/disagreeing draws remain unconfirmed',
        'coverage_scope': 'Full-roster immutable closeouts only; partial ongoing work is not counted as zero',
        'new_inference_draws': 0, 'new_job_submissions': 0,
    })
    print(json.dumps({'rows': rows, 'receipt_sha256': receipt['sha256'],
                      'table_file_sha256': receipt['table_file_sha256']}, indent=1))


if __name__ == '__main__':
    main()
