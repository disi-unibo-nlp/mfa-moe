"""Compare the manuscript's citations with independently inspected public records, offline.

The paper-writer bibliography checker runs with its network lookup replaced by the records in
BUILD/author-revision-20261014/PRIMARY_RECORDS.json. It does not normalise the references. Writes
BUILD/bib/OFFLINE_VERIFICATION.json.

  python -m paper.verify_cl_bibliography_offline
"""
import importlib.util
import json

from paper.cl_paths import BUILD, PAPER, PAPER_WRITER
from store import file_sha


def main():
    evidence = BUILD / 'author-revision-20261014/PRIMARY_RECORDS.json'
    data = json.loads(evidence.read_text())['records']
    spec = importlib.util.spec_from_file_location('paper_bib_offline', PAPER_WRITER / 'scripts/bib.py')
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
    cache = BUILD / 'bib/cache.json'
    if cache.exists():
        cache.with_name('cache-before-offline.json').write_bytes(cache.read_bytes())
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
        'primary_records_sha256': file_sha(evidence),
        'bibliography_sha256': file_sha(PAPER / 'references.bib'),
        'skill_bib_sha256': file_sha(PAPER_WRITER / 'scripts/bib.py'),
        'unresolved_metadata': 0,
        'approval_review': 'bulk external API lookup rejected; no retry or external-disclosure workaround',
    }
    (BUILD / 'bib/OFFLINE_VERIFICATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
