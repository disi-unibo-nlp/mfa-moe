"""One bounded read-only Codex diagnostic, with private agent state in HOME.

This cannot authorize or execute submissions or repairs. The deterministic
controller handles approved work. No tool use or runtime changes are requested.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from moe_exp.verification_study.contracts import check_files, digest, load, require, save
from moe_exp.verification_study.registry import REPO

SCHEMA = {'type': 'object', 'properties': {
    'status': {'type': 'string', 'enum': ['DIAGNOSTIC', 'BLOCKED', 'SMOKE_PASS']},
    'summary': {'type': 'string'}, 'recommended_action': {'type': 'string'}},
    'required': ['status', 'summary', 'recommended_action'], 'additionalProperties': False}


def review(campaign_path, frontier=None, *, smoke=False):
    campaign = load(campaign_path); check_files(campaign['code_files'])
    require(Path.home() == Path('/leonardo/home/userexternal/lmolfett'), 'agent state must stay in authorized private HOME')
    identity = {'campaign_sha256': campaign['sha256'], 'frontier': frontier or {}, 'smoke': smoke}
    directory = Path.home() / '.codex/cv3-readonly-reviews' / digest(identity)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = directory / 'RESULT.json'
    if result_path.exists():
        return load(result_path)
    if (directory / 'INTENT.json').exists():
        return {'status': 'REVIEW_INTENT_REQUIRES_RECONCILIATION', 'private_path': str(directory)}
    schema_path = directory / 'schema.json'; schema_path.write_text(json.dumps(SCHEMA))
    prompt = ('Use no tools and make no changes. Review only the supplied material. '
              'No submissions, file edits, shell actions, credentials, or runtime changes. ')
    if smoke:
        prompt += 'Return status SMOKE_PASS, summary bounded read-only launcher works, recommended_action none.'
    else:
        prompt += ('Diagnose this explicit stopped frontier for the frozen all-seven verification v3 study. '
                   'Do not propose narrowing models, changing targets/gates, unpriced final runs, unbounded retries or rewriting frozen code. '
                   'Distinguish execution/price/coverage errors from scientific limits. Give the smallest concrete operational repair recommendation. '
                   + json.dumps(frontier, indent=2))
        for name in ('DECISIONS.md', 'MODEL_COVERAGE.md'):
            prompt += '\nReference ' + name + ':\n' + (REPO / 'paper/wiki' / name).read_text()[:20000]
    (directory / 'prompt.md').write_text(prompt)
    intent = save(directory / 'INTENT.json', {'schema': 'verification-codex-readonly-intent-v3',
        **identity, 'model': 'gpt-6.1-sol', 'reasoning': 'xhigh', 'started_epoch': time.time(),
        'timeout_seconds': 180, 'mode': 'read-only diagnostic with no tools',
        'authorization': 'user: bounded scripted Codex call when progress needs a diagnostic'})
    answer = directory / 'answer.json'
    argv = ['/leonardo/home/userexternal/lmolfett/.local/bin/codex', 'exec', '--ephemeral', '--sandbox', 'read-only',
            '--model', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="xhigh"', '--cd', str(REPO),
            '--output-schema', str(schema_path), '--output-last-message', str(answer), '--json', '-']
    started = time.monotonic()
    try:
        with (directory / 'events.jsonl').open('w') as events, (directory / 'stderr.log').open('w') as errors:
            process = subprocess.run(argv, input=prompt, text=True, stdout=events, stderr=errors, timeout=180)
        require(process.returncode == 0 and answer.exists(), 'read-only Codex call did not produce a successful structured answer; inspect private logs')
        value = json.loads(answer.read_text())
        require(set(value) == set(SCHEMA['required']) and value['status'] in SCHEMA['properties']['status']['enum'], 'invalid structured diagnostic')
        check_files(campaign['code_files'])
        result = {'schema': 'verification-codex-readonly-result-v3', 'intent_sha256': intent['sha256'],
                  'campaign_sha256': campaign['sha256'], 'elapsed_seconds': time.monotonic() - started, **value}
    except Exception as error:
        result = {'schema': 'verification-codex-readonly-result-v3', 'intent_sha256': intent['sha256'],
                  'campaign_sha256': campaign['sha256'], 'status': 'CODEX_DIAGNOSTIC_FAILED', 'error': str(error)[:1000],
                  'elapsed_seconds': time.monotonic() - started, 'private_path': str(directory)}
    return save(result_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=REPO / 'report/verification-study-v3/CAMPAIGN.json')
    parser.add_argument('--frontier', type=Path); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); require(args.smoke or args.frontier, 'explicit diagnostic frontier or smoke required')
    result = review(args.campaign.resolve(), load(args.frontier) if args.frontier else None, smoke=args.smoke)
    print(json.dumps({k: v for k, v in result.items() if k in {'status', 'elapsed_seconds', 'private_path'}}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
