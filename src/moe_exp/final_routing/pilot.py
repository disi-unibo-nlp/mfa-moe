"""Frozen unstarted reader inventory and complete whole-batch timing evidence."""
from pathlib import Path
from .common import MODELS, Units, digest, load, publish, require, verified


def select(binding):
    from transformers import AutoTokenizer
    from .renderer import Renderer
    from moe_exp.research_campaign_v3_r6.reader_contract import token_ids
    from .contexts import build
    root = Path(binding['root'])
    destination = root/'READER_PILOT_TASKS.json'
    if destination.exists():
        return verified(destination)
    parent = load(Path(binding['source_campaign'])/'CAMPAIGN.json')
    tokenizer = AutoTokenizer.from_pretrained(parent['annotator']['path'], local_files_only=True)
    renderer = Renderer(parent)
    selected, seen = [], set()
    units = Units(Path(binding['config']['bulk_root'])/'reader'/'annotations')
    try:
        for model in MODELS:
            assignments = verified(root/model/'PRICING_ASSIGNMENTS.json')['records']
            tasks = build(binding, model, assignments, namespace='pricing')['tasks']
            candidates = []
            import sqlite3
            index = Path(binding['source_campaign'])/model/'LABEL_INDEX.sqlite'
            db = sqlite3.connect('file:'+str(index)+'?mode=ro', uri=True) if index.exists() else None
            for task in tasks:
                if task['uid'] in seen or units.state(task['uid']) is not None:
                    continue
                # Pricing completions have fresh continuation identities. Check
                # the inherited annotation journal as well before choosing them.
                from moe_exp.research_campaign_v3_r6.contracts import Units as OldUnits
                if model != 'lfm25' and OldUnits(Path(binding['source_campaign'])/model/'annotations').state(task['uid']) is not None:
                    continue
                if db is not None and db.execute('select 1 from labels where context=?',(task['uid'],)).fetchone():
                    continue
                conversation = renderer.request('render', inputs=task['inputs'])['conversation']
                n = len(token_ids(tokenizer, conversation, parent['annotator']['config']))
                require(n+16384 <= parent['annotator']['native_context'], 'pilot reader context exceeds fixed allowance')
                candidates.append({**task, 'input_tokens': n})
            if db is not None:
                db.close()
            candidates.sort(key=lambda r: (r['input_tokens'], r['uid']))
            require(len(candidates) >= 64, model+' has fewer than 64 unstarted pricing contexts')
            # Eight contexts/model from each source-model input octile. Extremes
            # are retained explicitly to cover the observed inventory bounds.
            for b in range(8):
                block = candidates[len(candidates)*b//8:len(candidates)*(b+1)//8]
                picks = sorted(block, key=lambda r: digest([binding['config']['seed'], 'reader-pilot', model, b, r['uid']]))[:8]
                edge = block[0] if b == 0 else block[-1] if b == 7 else None
                if edge is not None and edge not in picks:
                    picks[-1] = edge
                for task in picks:
                    selected.append({**task, 'source_octile': b})
                    seen.add(task['uid'])
    finally:
        renderer.close()
    require(len(selected) == len(seen) == 512, 'reader pilot must contain 512 distinct unstarted contexts')
    selected.sort(key=lambda r: (r['input_tokens'], r['uid']))
    batches = [selected[i:i+64] for i in range(0, 512, 64)]
    return publish(destination, {'records': selected, 'batches': batches, 'source_models': list(MODELS),
        'selection': '64/model, eight/model-input-octile, retained extremes; global length-sorted batches',
        'reader_binding': digest(parent['annotator']), 'primary_outcomes_used': False})


def run(binding, deadline):
    from .reader import Reader
    from .budget import reader_price
    root, bulk = Path(binding['root']), Path(binding['config']['bulk_root'])
    tasks = verified(root/'READER_PILOT_TASKS.json')
    reader = Reader(binding, deadline)
    try:
        for batch in tasks['batches']:
            reader.run(batch, input_bin='covered')
        receipts = []
        for batch in tasks['batches']:
            path = bulk/'reader'/'batches'/(digest([r['uid'] for r in batch])+'.json')
            require(path.exists(), 'pilot batch timing missing; ambiguous work cannot be redrawn')
            receipts.append(verified(path))
        require(len(receipts) == 8 and all(r['batch_size'] == 64 for r in receipts), 'incomplete reader timing coverage')
        profile = reader_price(receipts)
        return publish(root/'READER_PRICE.json', {'status': 'complete', 'profiles': profile,
            'batch_receipts': receipts, 'pilot_tasks_sha256': tasks['sha256'],
            'contexts': 512, 'reader_cap': 16384, 'tail_policy': 'full measured batch cost'})
    finally:
        reader.close()
