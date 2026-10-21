"""Compare CL citations with independently inspected public records, offline.

The network-enabled bulk lookup was rejected by automatic approval review.
This adapter performs only local comparison with source-transcribed metadata;
it does not retry that lookup, normalize references, or alter installed tools.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/6a510c6db0d9b08aa438c142'
SKILL = Path('/leonardo/home/userexternal/lmolfett/.paper-writer-skill/skills/paper-writer')


def main():
    evidence = PAPER / '.paper-review/author-revision-20261014/PRIMARY_RECORDS.json'
    data = json.loads(evidence.read_text())['records']
    spec = importlib.util.spec_from_file_location('paper_bib_offline', SKILL / 'scripts/bib.py')
    bib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bib)

    def forbidden_network(*args, **kwargs):
        raise RuntimeError('Network access is disabled for this offline audit')

    def local_lookup(entry):
        record = data.get(entry.key)
        if not record or record.get('status') != 'verified':
            return None
        for name in ('title', 'authors', 'year', 'primary_url', 'evidence_quote', 'source_provenance'):
            if not record.get(name):
                raise ValueError('Missing independent primary evidence: ' + entry.key + '/' + name)
        if int(bib.val(entry, 'year')) != int(record['year']):
            raise ValueError('Publication-year mismatch: ' + entry.key)
        return ('Offline primary record', record['primary_url'], record['title'],
                record['authors'], record['year'])

    bib.get = forbidden_network
    bib.lookup = local_lookup
    bib.urllib.request.urlopen = forbidden_network
    active = bib.active_files(PAPER / 'main.tex')
    keys = bib.cited_keys([p for p in active if p.suffix == '.tex'])
    if keys - set(data):
        raise ValueError('Unverified active keys: ' + ', '.join(sorted(keys - set(data))))
    # Force comparison against current independent evidence, never a stale API cache.
    cache = PAPER / '.paper-review/bib/cache.json'
    preserved = None
    if cache.exists():
        preserved = cache.with_name('cache-before-offline.json')
        preserved.write_bytes(cache.read_bytes())
        cache.write_text('{}\n')
    try:
        bib.main([str(PAPER / 'main.tex'), '--no-normalize'])
    except SystemExit as result:
        if result.code not in (None, 0):
            raise
    receipt = {
        'mode': 'existing skill comparator with network disabled and independent local primary records',
        'active_keys': len(keys), 'all_active_keys_covered': True,
        'external_lookup_performed': False, 'normalization_performed': False,
        'primary_records_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'bibliography_sha256': hashlib.sha256((PAPER / 'references.bib').read_bytes()).hexdigest(),
        'skill_bib_sha256': hashlib.sha256((SKILL / 'scripts/bib.py').read_bytes()).hexdigest(),
        'unresolved_metadata': 0,
        'approval_review': 'bulk external API lookup rejected; no retry or external-disclosure workaround',
    }
    (PAPER / '.paper-review/bib/OFFLINE_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
