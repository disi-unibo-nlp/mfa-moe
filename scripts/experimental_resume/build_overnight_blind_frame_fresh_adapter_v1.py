"""Execute the frozen v2 builder with explicitly checked fresh-schema contexts."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import build_overnight_blind_frame_v2 as frozen
import fresh_frame_recovery_v1 as recovery


def build(manifest, price, source, enrollment, selection, units, run_out, out, tokenizer, amendment, salt=None):
    recovery.validate_amendment(amendment, manifest)
    derived, contexts, provenance = recovery.checked_contexts(manifest, source, enrollment, selection, units)
    originals = {r['uid']: r for r in source['records']}
    natives = {r['uid']: r for r in manifest['rows']}
    normalized = {r['uid']: r for r in derived['records']}
    for uid, native in natives.items():
        frozen.reader_context(normalized[uid], native, source['schema'])
    fields = recovery.recovery_fields(amendment, provenance, __file__)
    previous_save = frozen.rating.save

    def context(original, native, schema):
        uid = native['uid']
        recovery.require(schema == source['schema'] and original == originals[uid] and native == natives[uid],
                         'context callback rebound to different frozen row')
        return contexts[uid]

    def save(path, body, **kwargs):
        return previous_save(path, {**body, 'recovery_provenance': fields}, **kwargs)

    out = Path(out)
    recovery.require(not (out / 'BLIND_FRAME.json').exists() or (out / 'ARM_MAP.json').exists(),
                     'orphan frame cannot be rebound with a fresh salt')
    with patch.object(frozen, 'reader_context', context), patch.object(frozen.rating, 'save', save):
        result = frozen.build(manifest, price, source, Path(run_out), out, tokenizer, salt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'generation-price', 'run-out', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    recovery.require(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID'),
                     'fresh decoding requires CPU Slurm step')
    m = recovery.base.sealed(args.manifest)
    a = recovery.validate_amendment(recovery.base.sealed(recovery.AMENDMENT), m)
    recovery.require(str(args.run_out.resolve()) == a['paths']['generation'] and
                     str(args.out.resolve()) == a['paths']['measurement'], 'recovery output paths differ')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(frozen.TOKENIZER, local_files_only=True)
    amap, frame = build(m, recovery.base.sealed(args.generation_price),
                        recovery.base.sealed(m['source_frame_path']),
                        recovery.base.sealed(m['source_enrollment_path']),
                        recovery.base.sealed(recovery.SELECTION), recovery.base.sealed(recovery.UNITS),
                        args.run_out, args.out, tokenizer, a)
    print(json.dumps({'map_sha256': amap['sha256'], 'frame_sha256': frame['sha256'],
                      'assigned': frame['assigned'], 'gradeable': len(frame['records']),
                      'recovery_provenance': frame['recovery_provenance']}), flush=True)


if __name__ == '__main__':
    main()
