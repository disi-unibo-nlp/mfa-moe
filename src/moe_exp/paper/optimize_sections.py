"""Run resumable ELM section revisions with independent, counterbalanced judges.

This orchestrates the existing deep-research batch client; it never edits the paper.
Every proposal, request, response and selection remains in the private run directory.
The author-approved protocol and first generator requests must already exist there.

Usage: python -m moe_exp.paper.optimize_sections --run-dir RUN --client CLIENT
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read_jsonl(path: Path) -> list[dict]:
    """Read complete request or response records.

    Args:
        path: Existing UTF-8 JSONL artifact.
    Returns:
        Parsed records in recorded order.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numbers(text: str) -> Counter:
    """Extract numeric literals after excluding link addresses and comments."""
    text = re.sub(r'\\(?:href|url)\{[^{}]*\}', '', text)
    text = re.sub(r'(?m)^%.*$', '', text)
    return Counter(re.findall(r'(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?![A-Za-z])', text))


def _complete_response(record: dict) -> bool:
    """Reject truncated JSON even when the upstream client reports HTTP success."""
    if record.get('error'):
        return False
    try:
        parsed = json.loads(record['output_text'])
    except (json.JSONDecodeError, TypeError):
        return False
    required = {'scores_A', 'scores_B', 'choice', 'scientific_regressions_A',
                'scientific_regressions_B'} if '-judge-' in record['custom_id'] else {'section_id', 'candidate_latex'}
    if not isinstance(parsed, dict) or not required.issubset(parsed):
        return False
    # A schema-shaped refusal is not a scientific review. This occurred when a
    # historical model-availability note was mistakenly retained in the rubric.
    if '-judge-' in record['custom_id']:
        values = list(parsed['scores_A'].values()) + list(parsed['scores_B'].values())
        if not any(values) or re.search(r'placeholders?|review (?:not (?:conducted|performed)|blocked)',
                                       parsed.get('reason', ''), re.I):
            return False
    return True


def protected_changes(baseline: str, candidate: str) -> dict:
    """Detect changes to scientific and structural source bindings.

    Args:
        baseline: Fixed source span before optimization.
        candidate: Complete proposed replacement span.
    Returns:
        Differences that veto automatic selection and numeric additions for review.
    """
    patterns = {
        'intervals': r'\\cid\{[^{}]*\}\{[^{}]*\}\{[^{}]*\}',
        'labels': r'\\label\{[^{}]+\}',
        'references': r'\\(?:[cC]ref|[cC]refrange|ref|eqref)\*?\{[^{}]+\}',
        'inputs': r'\\(?:input|include)\{[^{}]+\}',
        'equations': r'\\begin\{(?:equation|align)\*?\}.*?\\end\{(?:equation|align)\*?\}',
    }
    changes = {}
    for name, pattern in patterns.items():
        original = Counter(re.findall(pattern, baseline, re.S))
        proposed = Counter(re.findall(pattern, candidate, re.S))
        if original != proposed:
            changes[name] = {'removed': list((original - proposed).elements()),
                             'added': list((proposed - original).elements())}
    citation_pattern = r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^{}]+)\}'
    original_keys = {key.strip() for group in re.findall(citation_pattern, baseline)
                     for key in group.split(',')}
    proposed_keys = {key.strip() for group in re.findall(citation_pattern, candidate)
                     for key in group.split(',')}
    if original_keys != proposed_keys:
        changes['citations'] = {'removed': sorted(original_keys - proposed_keys),
                                'added': sorted(proposed_keys - original_keys)}
    # Names and URL versions are not numerical measurements. The latest author
    # amendment explicitly removes hyperlink wrappers and the AMC web footnote.
    original_numbers, proposed_numbers = _numbers(baseline), _numbers(candidate)
    removed_numbers = list((original_numbers - proposed_numbers).elements())
    if removed_numbers:
        changes['removed_numeric_literals'] = removed_numbers
    return {'vetoes': changes,
            'added_numeric_literals': list((proposed_numbers - original_numbers).elements())}


def select_candidate(reviews: list[dict], mapping: list[dict], guard: dict) -> dict:
    """Choose the higher-scored admissible draft; retain the incumbent on ties.

    Args:
        reviews: Two validated independent pairwise judgments.
        mapping: Corresponding blind A/B identity mappings.
        guard: Protected-source differences against the fixed baseline.
    Returns:
        Scores and a conservative selection decision, not scientific approval.
    """
    if len(reviews) != 2 or {row['candidate_label'] for row in mapping} != {'A', 'B'}:
        raise ValueError('Selection requires two independently counterbalanced reviews')
    totals = {'candidate': [], 'incumbent': []}
    regressions = []
    candidate_votes = 0
    for review, identity in zip(reviews, mapping):
        for role in totals:
            label = identity[role + '_label']
            values = list(review['scores_' + label].values())
            if len(values) != 5 or any(not 0 <= value <= 10 for value in values):
                raise ValueError('Judge scores must contain five dimensions in [0,10]')
            totals[role].append(sum(values) / len(values))
        candidate_label = identity['candidate_label']
        regressions.extend(review['scientific_regressions_' + candidate_label])
        candidate_votes += review['choice'] == candidate_label
    averages = {role: sum(values) / len(values) for role, values in totals.items()}
    admissible = not guard['vetoes'] and not regressions
    accepted = admissible and candidate_votes > 0 and averages['candidate'] > averages['incumbent']
    return {'selected': 'candidate' if accepted else 'incumbent',
            'mean_scores': averages, 'judge_scores': totals,
            'scientific_regressions': regressions, 'protected_source': guard,
            'admissible': admissible, 'author_approval': False}


class SectionOptimizer:
    """Keep the fixed evidence and complete round artifacts together."""

    def __init__(self, run_dir: Path, client: Path, workers: int) -> None:
        """Load the author-approved run without altering historical responses."""
        self.root, self.client, self.workers = run_dir, client, workers
        self.protocol = json.loads((run_dir / 'PROTOCOL.json').read_text())
        self.sections = self.protocol['main_sections']
        self.templates = read_jsonl(run_dir / 'round-01-generators.jsonl')
        self.judge_schema = json.loads((run_dir / 'JUDGE_SCHEMA.json').read_text())
        self.rubric = (run_dir / 'PAIRWISE_RUBRIC.md').read_text()
        self.notes = run_dir.parent.parent / 'notes'
        self.incumbents = {section: 'baseline.tex' for section in self.sections}
        self.feedback = {section: {} for section in self.sections}
        self.progress = {section: 0 for section in self.sections}
        if self.protocol['judges']['model_ids'] != ['gpt-6-luna', 'gpt-6-luna']:
            raise ValueError('This run requires the two author-approved gpt-6-luna judges')

    def _context(self, section: str) -> str:
        """Retain complete source examples and Humanizer guidance with live rules."""
        template = next(row for row in self.templates
                        if row['custom_id'].startswith(section + '-'))
        context = template['body']['input'][1]['content'].split('(E) CURRENT COMPLETE SECTION:')[0]
        complete_examples = self.root / 'sections' / section / 'examples-complete.md'
        if complete_examples.exists():
            historical_examples = (complete_examples.parent / 'examples.md').read_text()
            if historical_examples not in context:
                raise ValueError('Historical example packet not found in its recorded context')
            context = context.replace(historical_examples, complete_examples.read_text(), 1)
        if section == 'method':
            supplement = self.root.parent / 'review-revision/method-examples.md'
            context += '\n\nCOMPLETE SUPPLEMENTARY METHOD EXAMPLES\n' + supplement.read_text()
        # Current local rules supersede historical copies in the original packet.
        context += '\n\nCURRENT AUTHOR AMENDMENTS AND PAPER RULES (supersede earlier versions)\n'
        context += (self.notes / 'paper-rules.md').read_text()
        context += '\n\nCURRENT BRIEF\n' + (self.notes / 'writing_brief.md').read_text()
        return context

    def _batch(self, round_number: int, role: str, requests: list[dict]) -> dict:
        """Invoke the maintained ELM client and require complete successful results."""
        prefix = self.root / f'round-{round_number:02d}-{role}'
        request_path = prefix.with_suffix('.jsonl')
        result_path = self.root / f'round-{round_number:02d}-{role[:-1]}-results.jsonl'
        if not request_path.exists():
            request_path.write_text(''.join(json.dumps(row) + '\n' for row in requests))
        elif read_jsonl(request_path) != requests:
            raise ValueError(f'Request mismatch at {request_path}; preserve original run inputs')
        subprocess.run(['bash', str(self.client), 'batch', str(request_path), str(result_path),
                        '--workers', str(self.workers), '--retries', '1', '--timeout', '600'],
                       check=True)
        results = {row['custom_id']: row for row in read_jsonl(result_path) if _complete_response(row)}
        # High reasoning can exhaust its allowance before finishing the JSON. Keep
        # that charged response intact and retry only incomplete IDs with more room.
        incomplete = [row for row in requests if row['custom_id'] not in results]
        if incomplete:
            recovered = self._recover(number=round_number, role=role, requests=incomplete)
            results.update(recovered)
        missing = [row['custom_id'] for row in requests if row['custom_id'] not in results]
        if missing:
            raise ValueError('Incomplete ELM responses: ' + ', '.join(missing))
        return results

    def _recover(self, number: int, role: str, requests: list[dict]) -> dict:
        """Recover incomplete JSON once, preserving both actual charged attempts."""
        retry_path = self.root / f'round-{number:02d}-{role}-completion.jsonl'
        output_path = self.root / f'round-{number:02d}-{role}-completion-results.jsonl'
        retries = copy.deepcopy(requests)
        for request in retries:
            request['custom_id'] += '-completion'
            request['body']['max_output_tokens'] = 40000
        if retry_path.exists() and read_jsonl(retry_path) != retries:
            raise ValueError('Incomplete-response retry inputs changed; inspect saved evidence')
        retry_path.write_text(''.join(json.dumps(row) + '\n' for row in retries))
        subprocess.run(['bash', str(self.client), 'batch', str(retry_path), str(output_path),
                        '--workers', str(self.workers), '--retries', '1', '--timeout', '600'],
                       check=True)
        return {row['custom_id'].removesuffix('-completion'): row
                for row in read_jsonl(output_path) if _complete_response(row)}

    def _generate(self, number: int) -> None:
        """Generate from each current incumbent and the preceding actual feedback."""
        if number == 1:
            results = {row['custom_id']: row for row in
                       read_jsonl(self.root / 'round-01-generator-results.jsonl')}
            requests = self.templates
        else:
            requests = []
            for section in self.sections:
                template = next(row for row in self.templates
                                if row['custom_id'].startswith(section + '-'))
                body = copy.deepcopy(template['body'])
                body['max_output_tokens'] = 20000
                body['input'][0]['content'] = body['input'][0]['content'].replace(
                    'iteration 1', f'iteration {number}').replace('Three independent judges', 'Two independent judges')
                incumbent = (self.root / 'sections' / section / self.incumbents[section]).read_text()
                body['input'][1]['content'] = (self._context(section)
                    + f'\n\nCURRENT COMPLETE SECTION: {section}\n\n' + incumbent
                    + '\n\nPRECEDING ACTUAL REVIEW AND SELECTION\n'
                    + json.dumps(self.feedback[section], ensure_ascii=False)
                    + '\nAddress justified criticism while preserving all scientific evidence. '
                    + 'An identical candidate is acceptable if no justified improvement remains.')
                requests.append({'custom_id': f'{section}-iteration-{number:02d}-generator', 'body': body})
            saved_request = self.root / f'round-{number:02d}-generators.jsonl'
            if saved_request.exists():
                requests = read_jsonl(saved_request)
            results = self._batch(number, 'generators', requests)
        for request in requests:
            result = results[request['custom_id']]
            if result.get('error'):
                raise ValueError(result['error'])
            proposal = json.loads(result['output_text'])
            section = proposal['section_id']
            if section not in self.sections or not proposal['candidate_latex'].strip():
                raise ValueError('Generator returned an unknown or empty section')
            folder = self.root / 'sections' / section
            target = folder / f'candidate-{number:02d}.tex'
            if not target.exists():
                target.write_text(proposal['candidate_latex'].rstrip() + '\n')
            (folder / f'generator-{number:02d}.json').write_text(json.dumps(proposal, indent=2) + '\n')

    def _judge(self, number: int) -> tuple[dict, list[dict]]:
        """Submit two independent blind reviews with opposite draft orders."""
        mapping_path = self.root / f'round-{number:02d}-judge-mapping.json'
        request_path = self.root / f'round-{number:02d}-judges.jsonl'
        if request_path.exists():
            requests = read_jsonl(request_path)
            mapping = json.loads(mapping_path.read_text())
        else:
            requests, mapping = [], []
            for section in self.sections:
                folder = self.root / 'sections' / section
                incumbent = (folder / self.incumbents[section]).read_text()
                candidate = (folder / f'candidate-{number:02d}.tex').read_text()
                for slot in (1, 2):
                    candidate_label = 'A' if (number + slot) % 2 else 'B'
                    drafts = {'A': incumbent, 'B': candidate} if candidate_label == 'B' else {'A': candidate, 'B': incumbent}
                    custom_id = f'{section}-iteration-{number:02d}-judge-{slot}'
                    body = {
                        'model': 'gpt-6-luna',
                        'reasoning': {'effort': 'high'},
                        'input': [{'role': 'developer', 'content': self.rubric},
                                  {'role': 'user', 'content': self._context(section)
                                   + '\n\nDRAFT A\n\n' + drafts['A'] + '\n\nDRAFT B\n\n' + drafts['B']}],
                        'max_output_tokens': 30000,
                        'text': {'format': {'type': 'json_schema', 'name': 'section_pairwise_review',
                                            'strict': True, 'schema': self.judge_schema}},
                    }
                    requests.append({'custom_id': custom_id, 'body': body})
                    mapping.append({'custom_id': custom_id, 'section': section, 'round': number,
                                    'slot': slot, 'candidate_label': candidate_label,
                                    'incumbent_label': 'B' if candidate_label == 'A' else 'A'})
            mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
        return self._batch(number, 'judges', requests), mapping

    def _select(self, number: int, results: dict, mapping: list[dict]) -> None:
        """Persist every judgment and selection before advancing the next round."""
        for section in self.sections:
            folder = self.root / 'sections' / section
            identities = [row for row in mapping if row['section'] == section]
            reviews = [json.loads(results[row['custom_id']]['output_text']) for row in identities]
            baseline = (folder / 'baseline.tex').read_text()
            candidate = (folder / f'candidate-{number:02d}.tex').read_text()
            decision = select_candidate(reviews, identities, protected_changes(baseline, candidate))
            decision['previous_incumbent'] = self.incumbents[section]
            if decision['selected'] == 'candidate':
                self.incumbents[section] = f'candidate-{number:02d}.tex'
            decision['retained_incumbent'] = self.incumbents[section]
            decision['candidate_sha256'] = hashlib.sha256(candidate.encode()).hexdigest()
            decision['reviews'] = reviews
            (folder / f'selection-{number:02d}.json').write_text(json.dumps(decision, indent=2) + '\n')
            self.feedback[section], self.progress[section] = decision, number
            print(f'round {number:02d} {section}: {decision["selected"]}; '
                  f'candidate {decision["mean_scores"]["candidate"]:.2f}, '
                  f'incumbent {decision["mean_scores"]["incumbent"]:.2f}', flush=True)
        self._status()

    def _status(self) -> None:
        """Report only completed scored rounds and actual returned token usage."""
        usage = Counter()
        successes = Counter()
        for path in self.root.rglob('round-*-results.jsonl'):
            records = {row['custom_id']: row for row in read_jsonl(path) if not row.get('error')}
            for row in records.values():
                usage.update({key: value or 0 for key, value in row['usage'].items()})
                if 'rejected-attempts' in path.parts:
                    successes['superseded_attempts'] += 1
                elif _complete_response(row):
                    successes['judges' if '-judge-' in row['custom_id'] else 'generators'] += 1
                else:
                    successes['incomplete_responses'] += 1
        status = {
            'baseline': self.protocol['baseline_overleaf_commit'],
            'generator': self.protocol['generator'],
            'judges': self.protocol['judges'],
            'completed_scored_iterations_per_section': self.progress,
            'best_incumbent_per_section': self.incumbents,
            'successful_calls': dict(successes),
            'observed_token_usage': dict(usage),
            'status': 'scoring_complete_pending_independent_adoption' if min(self.progress.values()) == 10 else 'running',
            'paper_replacements': 0,
        }
        (self.root / 'RUN_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')

    def run(self) -> None:
        """Complete every requested round, resuming source-bound selections.

        Args:
            None; the constructor binds the private run and existing client.
        Returns:
            None. All results remain in the run directory; the paper is untouched.
        """
        for number in range(1, self.protocol['iterations_per_section'] + 1):
            saved = [self.root / 'sections' / section / f'selection-{number:02d}.json'
                     for section in self.sections]
            if all(path.exists() for path in saved):
                for section, path in zip(self.sections, saved):
                    self.feedback[section] = json.loads(path.read_text())
                    self.incumbents[section] = self.feedback[section]['retained_incumbent']
                    self.progress[section] = number
                continue
            self._generate(number)
            results, mapping = self._judge(number)
            self._select(number, results, mapping)
        self._status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--client', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    options = parser.parse_args()
    SectionOptimizer(options.run_dir.resolve(), options.client.resolve(), options.workers).run()
