"""CPU Slurm proof of exact HF token IDs and complete parallel reader pricing."""
from __future__ import annotations

from collections import Counter
import json
import os
import socket

import mechanism_extension_reader_contract_v2 as contract
import rate_mechanism_extension_start_readers_v2 as reader
import seal_mechanism_extension_reader_v2 as sealer
from freeze_mechanism_extension_220_v1 import digest, sealed


def main():
    contract.require(bool(os.environ.get('SLURM_JOB_ID')) and
                     bool(os.environ.get('SLURM_STEP_ID')) and
                     not socket.gethostname().startswith('login'),
                     'actual tokenizer preflight requires a CPU Slurm step')
    family, selection, frame, source, manifest = reader.expected_manifest()
    if contract.CPU_RESULT.exists():
        value = sealer.check_cpu_preflight(frame, source, manifest)
        print(json.dumps({'result': str(contract.CPU_RESULT), 'sha256': value['sha256']}))
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(contract.primary.MODEL,
                                              local_files_only=True)
    records, containers = [], Counter()
    lengths = {kind: [] for kind in contract.KINDS}
    for kind in contract.KINDS:
        for row in frame['records']:
            raw = tokenizer.apply_chat_template(
                contract.messages(row, kind), tokenize=True,
                add_generation_prompt=True, enable_thinking=True,
                reasoning_effort='low')
            containers[type(raw).__module__ + '.' + type(raw).__name__] += 1
            ids = contract.exact_token_ids(raw)
            contract.require(len(ids) + contract.MAX_TOKENS <=
                             contract.PROFILE['max_model_len'],
                             'full extension prompt exceeds model context')
            lengths[kind].append(len(ids))
            records.append({'uid': row['uid'], 'kind': kind,
                            'prompt_tokens': len(ids),
                            'prompt_ids_sha256': digest(ids)})
        print(json.dumps({'kind': kind, 'exact_prompts': len(lengths[kind])}), flush=True)
    contract.require(2 * sum(lengths['primary']) == source['primary_prompt_tokens_exact_twice'] and
                     2 * sum(lengths['veto']) == source['strict_veto_prompt_tokens_exact_twice'] and
                     min(lengths['primary'] + lengths['veto']) == source['prompt_tokens_min'] and
                     max(lengths['primary'] + lengths['veto']) == source['prompt_tokens_max'],
                     'actual exact prompt IDs disagree with source price')
    parallel = sealer.extension.price_reader(
        frame, selection, family, sealed(sealer.extension.PRIOR_PRICE),
        lengths['primary'], lengths['veto'], contract.PARALLEL_WALL_SECONDS)
    body = {
        'schema': 'mechanism-extension-reader-cpu-preflight-v2',
        'status': 'PASS_EXACT_IDS_AND_SOURCE_PRICE',
        'qualification_manifest_sha256': manifest['sha256'],
        'source_exact_price_sha256': source['sha256'],
        'driver_sha256': contract.file_sha(__file__),
        'contract_sha256': contract.file_sha(contract.__file__),
        'job_id': os.environ['SLURM_JOB_ID'],
        'rows': frame['rows'], 'unique_prompts': len(records),
        'tokenizer_container_counts': dict(containers),
        'exact_tokenizer_container_normalization_pass': True,
        'source_original_complete_stage_GPU_h': source['complete_stage_projected_GPU_h'],
        'parallel_price_preview': parallel,
        'records': records,
        'scope': 'CPU tokenizer and complete-stage price proof; no GPU qualification or eligibility result.',
    }
    value = contract.save(contract.CPU_RESULT, body)
    sealer.check_cpu_preflight(frame, source, manifest)
    print(json.dumps({'result': str(contract.CPU_RESULT), 'sha256': value['sha256'],
                      'rows': frame['rows'], 'parallel_shards': len(parallel['shards']),
                      'parallel_GPU_h': parallel['complete_stage_projected_GPU_h']}), flush=True)


if __name__ == '__main__':
    main()
