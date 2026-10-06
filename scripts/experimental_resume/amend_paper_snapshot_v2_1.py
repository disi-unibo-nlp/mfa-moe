"""Append the source-checked reader-token pricing caveat to paper snapshot v2."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


REPO = Path('/leonardo_work/IscrC_MIOSR/lmolfett/mfa-moe/repo')
DOC = REPO / 'report/experimental-resume-v1'
DRIVER = REPO / 'scripts/experimental_resume/rate_eligible_immediate_semantics_v1.py'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(f'broken saved seal: {path}')
    return value


def driver_rating_cap():
    tree = ast.parse(DRIVER.read_text())
    values = [node.value.value for node in tree.body
              if isinstance(node, ast.Assign) and
              any(isinstance(target, ast.Name) and target.id == 'MAX_TOKENS'
                  for target in node.targets) and
              isinstance(node.value, ast.Constant)]
    if values != [1024]:
        raise ValueError(f'eligible reader cap differs: {values}')
    return values[0]


def main():
    base_path = DOC / 'PAPER_SNAPSHOT_v2.json'
    price_path = DOC / 'CAUSAL_ELIGIBLE_DEACTIVATION_BLIND_RATING_PRICE_v3.json'
    audit_path = DOC / 'CAUSAL_READER_OUTPUT_LENGTH_AUDIT_v1.json'
    output = DOC / 'PAPER_SNAPSHOT_v2_1.json'
    if output.exists():
        raise FileExistsError(output)
    base, price, audit = sealed(base_path), sealed(price_path), sealed(audit_path)
    counts = audit['counts']
    if (base['schema'] != 'resume-paper-evidence-addendum-v2' or
        price['maximum_decode_tokens_per_rating'] != 256 or
        counts['valid_stopped_ratings'] != 97 or
        counts['valid_stopped_over_256_tokens'] != 44 or
        counts['all_assigned_ratings'] != 104 or
        counts['length_stops_at_1024'] != 3 or
        driver_rating_cap() != 1024):
        raise ValueError('reader-tail or pricing assumption changed')
    body = {
        'schema': 'resume-paper-operational-amendment-v2.1',
        'status': 'PRICING_MEASUREMENT_CAVEAT_NO_SCIENTIFIC_EFFECT',
        'base_snapshot_v2_sha256': base['sha256'],
        'base_claim_ledger_v2_file_sha256': base['claim_ledger_v2_file_sha256'],
        'source_file_sha256': {str(path): sha(path) for path in
                               (base_path, price_path, audit_path, DRIVER)},
        'generator_model': 'Qwen3.6',
        'independent_reader_model': 'Qwen3.8',
        'prior_reader_pilot': {
            'assigned_ratings': counts['all_assigned_ratings'],
            'valid_stopped_ratings': counts['valid_stopped_ratings'],
            'valid_stopped_over_256_output_tokens': counts['valid_stopped_over_256_tokens'],
            'length_stops_at_1024_output_tokens': counts['length_stops_at_1024'],
            'length_audit_sha256': audit['sha256'],
            'population_limit': 'prior micro-screen reader batches, not new eligible deactivation continuations'},
        'eligible_rating_resource_gate': {
            'preliminary_v3_price_sha256': price['sha256'],
            'preliminary_v3_max_output_tokens_per_rating': 256,
            'corrected_driver_max_output_tokens_per_rating': 1024,
            'preliminary_v3_envelope_applicable_to_corrected_driver': False,
            'required_before_rating_submission': 'post-generation exact blind-frame price at 1024 output tokens per rating; bind driver, prompts, completed generation manifest, loads and retry overhead',
            'rating_execution': 'NO_ELIGIBLE_RATING_JOB_LAUNCHED_AT_THIS_AMENDMENT'},
        'scientific_interpretation': 'Output-tail counts and a corrected resource gate are measurement/cost evidence only. They are not a causal semantic-control estimate.'}
    output.write_text(json.dumps({**body, 'sha256': digest(body)}, indent=1) + '\n')
    print(json.dumps({'path': str(output), 'sha256': digest(body)}))


if __name__ == '__main__':
    main()
