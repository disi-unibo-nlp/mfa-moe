"""Recompute CPU-only comparison summaries for complete saved guiding pairs."""
import argparse
import hashlib
import json
from pathlib import Path

from .bootstrap import add_arguments
from .run import compare


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    add_arguments(parser)
    args = parser.parse_args()
    for path in sorted(args.results_root.glob('moe*guiding/**/guided/manifest.json')):
        root = path.parent.parent
        baseline, guided = root/'baseline', root/'guided'
        if not (baseline/'manifest.json').exists():
            continue
        if any(json.loads((p/'manifest.json').read_text())['status'] != 'complete' for p in (baseline,guided)):
            print(f'Skipping incomplete pair: {root}', flush=True)
            continue
        print(f'Comparing {root} with {args.bootstrap_workers} CPU workers', flush=True)
        result = compare(baseline, guided, bootstrap_replicates=args.bootstrap_replicates,
                         bootstrap_seed=args.bootstrap_seed, bootstrap_workers=args.bootstrap_workers)
        result['source_sha256'] = {str(p/name): sha256(p/name) for p in (baseline,guided)
                                   for name in ('manifest.json','generations.jsonl')}
        output = root/'comparison_bootstrap.json'
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(output)
        print(json.dumps(result['bootstrap']['overall']), flush=True)


if __name__ == '__main__':
    main()
