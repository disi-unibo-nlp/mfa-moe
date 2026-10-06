"""Exactly priced FRESH dense dispatch with explicit metadata-recovery bindings."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import socket

import dispatch_generated_dense_v1 as original
import generated_dense_fresh_metadata_recovery_v1 as recovery

WRAPPERS = [recovery.SCRIPTS / 'label_generated_dense_fresh_metadata_recovery_v1.sbatch',
            recovery.SCRIPTS / 'analyze_generated_dense_fresh_metadata_recovery_v1.sbatch']


def prepared(manifest_path, generation_price_path, run_out, parent):
    recovery.install_adapter()
    manifest, generation_price = recovery.sealed(manifest_path), recovery.sealed(generation_price_path)
    amendment = recovery.validate_amendment(recovery.sealed(recovery.AMENDMENT), manifest)
    recovery.require(Path(manifest_path).resolve() == recovery.context.MANIFEST.resolve() and
                     Path(generation_price_path).resolve() == recovery.context.PRICE.resolve() and
                     Path(run_out).resolve() == Path(amendment['generation_out']).resolve(),
                     'dense recovery dispatcher source paths differ')
    directory = recovery.output_path(manifest, parent)
    frame, price = recovery.inputs(directory)
    arm_map = recovery.sealed(directory / 'ARM_MAP.json')
    recovery.require(generation_price['manifest_sha256'] == manifest['sha256'] and
                     generation_price['status'] == 'PASS_COMPLETE_STAGE_GENERATION_ONLY' and
                     generation_price['shards'] == manifest['shards'] and
                     frame['generation_run_out'] == str(Path(run_out).resolve()) and
                     recovery.sealed(Path(run_out) / 'STAGE_COMPLETION.json')['sha256'] ==
                     frame['generation_stage_sha256'], 'dense recovery prepared generation differs')
    original.validate_complete_price(frame, price)
    binding = {'schema': 'generated-dense-fresh-recovery-dispatch-binding-v1',
               'manifest_path': str(Path(manifest_path).resolve()),
               'generation_manifest_sha256': manifest['sha256'],
               'generation_price_sha256': generation_price['sha256'], 'dense_directory': str(directory),
               'frame_sha256': frame['sha256'], 'arm_map_sha256': arm_map['sha256'],
               'price_sha256': price['sha256'], 'measurement_plan_sha256': amendment['original_plan_sha256'],
               'recovery_amendment_sha256': amendment['sha256'],
               'recovery_provenance': frame['recovery_provenance'],
               'complete_stage_GPU_h': price['complete_stage_GPU_h'],
               'dispatch_driver_sha256': recovery.file_sha(__file__),
               'scientific_dispatch_driver_sha256': recovery.file_sha(original.__file__),
               'shared_dispatch_sha256': recovery.file_sha(original.shared.__file__),
               'wrapper_hashes': {str(p): recovery.file_sha(p) for p in WRAPPERS}}
    return directory, frame, price, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'generation_price', 'generation_out', 'dense_parent'):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    recovery.require(os.environ.get('SLURM_JOB_ID') and
                     os.environ.get('SLURM_JOB_PARTITION') == 'lrd_all_viz' and
                     getpass.getuser() == 'lmolfett' and not socket.gethostname().startswith('login'),
                     'dense recovery dispatcher requires its authorized one-shot viz CPU job')
    directory, frame, price, binding = prepared(args.manifest, args.generation_price,
                                               args.generation_out, args.dense_parent)
    original.WRAPPERS = WRAPPERS
    value = original.dispatch(directory, frame, price, binding)
    print(json.dumps({'chain': str(directory / 'dispatch-v1/CHAIN.json'), 'sha256': value['sha256'],
                      'label_array_job': value['label_array_job'], 'analysis_job': value['analysis_job']}), flush=True)


if __name__ == '__main__':
    main()
