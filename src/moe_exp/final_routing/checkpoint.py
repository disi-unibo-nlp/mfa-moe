"""Pinned public LFM checkpoint staging through the CINECA data mover."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
from .common import file_sha, load, private, publish, require

ALLOWED = re.compile(r'(?:model-\d+-of-\d+\.safetensors|model\.safetensors\.index\.json|'
                     r'config\.json|generation_config\.json|tokenizer(?:_config)?\.json|chat_template\.jinja|LICENSE)')


def manifest(config):
    lfm = config['lfm25']
    require(lfm['repository'] == 'LiquidAI/LFM2.5-24B-A2B' and
            re.fullmatch('[0-9a-f]{40}', lfm['revision']), 'unexpected public checkpoint source')
    url = 'https://huggingface.co/api/models/'+lfm['repository']+'/revision/'+lfm['revision']+'?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as stream:
        data = json.load(stream)
    require(data['sha'] == lfm['revision'], 'checkpoint revision mismatch')
    files = [{'name': r['rfilename'], 'bytes': r['size'], 'git_blob': r.get('blobId'),
              'sha256_file': (r.get('lfs') or {}).get('sha256')} for r in data['siblings']
             if ALLOWED.fullmatch(r['rfilename'])]
    require(sum(r['name'].endswith('.safetensors') for r in files) == 10, 'checkpoint shard inventory changed')
    return {'repository': lfm['repository'], 'revision': lfm['revision'], 'files': files,
            'bytes': sum(r['bytes'] for r in files), 'destination': str(private(lfm['local_path']))}


def stage(config):
    root = Path(config['root'])
    path = root/'LFM_TRANSFER_MANIFEST.json'
    value = load(path) if path.exists() else publish(path, manifest(config))
    folder = private(value['destination'])
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Only allow the already approved private FAST root and a bounded 48-GB checkpoint.
    require(value['bytes'] < 50*1024**3, 'checkpoint exceeds approved staging size')
    logs = root/'transfer_logs'
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    def transfer(row):
        target = folder/row['name']
        if target.exists() and target.stat().st_size == row['bytes']:
            return {'name': row['name'], 'status': 'present_pending_checksum'}
        url = 'https://huggingface.co/'+value['repository']+'/resolve/'+value['revision']+'/'+row['name']
        if config.get('checkpoint_transfer') == 'huggingface_https':
            # Explicit user override: "just download the model from huggingface".
            # TLS stays verified. Commit only a complete, checksum-verified file.
            temporary = target.with_name('.'+target.name+'.download')
            request = urllib.request.Request(url+'?download=true', headers={'User-Agent': 'mfa-moe-final-routing/1'})
            h = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
                for block in iter(lambda: response.read(4*1024**2), b''):
                    output.write(block)
                    h.update(block)
            require(temporary.stat().st_size == row['bytes'], 'Hugging Face download size differs')
            if row['sha256_file']:
                require(h.hexdigest() == row['sha256_file'], 'Hugging Face checksum differs')
            elif row['git_blob']:
                data = temporary.read_bytes()
                require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == row['git_blob'],
                        'Hugging Face metadata checksum differs')
            temporary.replace(target)
            return {'name': row['name'], 'status': 'downloaded_checksum_verified', 'sha256_file': h.hexdigest()}
        args = ['ssh', '-xT', '-o', 'StrictHostKeyChecking=yes', '-o', 'BatchMode=yes',
                'lmolfett@data.leonardo.cineca.it', 'wget', '--quiet', '--continue',
                '--timeout=30', '--tries=2', url, '-P', str(folder)]
        with (logs/(row['name']+'.log')).open('ab') as stream:
            subprocess.run(args, check=True, stdout=stream, stderr=stream, timeout=1800)
        require(target.stat().st_size == row['bytes'], 'downloaded file size differs: '+row['name'])
        return {'name': row['name'], 'status': 'transferred_pending_checksum'}
    # Metadata first so the frozen model profile includes the real config/tokenizer.
    for row in value['files']:
        if not row['name'].endswith('.safetensors'):
            transfer(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(transfer, [r for r in value['files'] if r['name'].endswith('.safetensors')]))
    return publish(root/'LFM_TRANSFER_COMPLETE.json', {'status': 'transferred',
        'manifest_sha256': value['sha256'], 'weights_verified': False, 'files': results,
        'checksum_stage': 'allocated CPU validation before model loading'})


def verify(config):
    value = load(Path(config['root'])/'LFM_TRANSFER_MANIFEST.json')
    folder = Path(value['destination'])
    files = {}
    for row in value['files']:
        path = folder/row['name']
        require(path.is_file() and path.stat().st_size == row['bytes'], 'checkpoint file absent/truncated')
        if row['sha256_file']:
            require(file_sha(path) == row['sha256_file'], 'checkpoint LFS checksum differs: '+row['name'])
        elif row['git_blob']:
            data = path.read_bytes()
            require(hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == row['git_blob'],
                    'checkpoint Git blob checksum differs: '+row['name'])
        files[str(path)] = row['sha256_file'] or file_sha(path)
    config_data = load(folder/'config.json')
    require(config_data['num_hidden_layers'] == 40 and config_data['num_experts'] == 64 and
            config_data['num_experts_per_tok'] == 4 and config_data['num_dense_layers'] == 2,
            'pinned LFM geometry differs')
    return publish(Path(config['root'])/'LFM_CHECKPOINT_VERIFIED.json',
                   {'status': 'complete', 'manifest_sha256': value['sha256'], 'files': files,
                    'routed_layers': list(range(2, 40)), 'experts': 64, 'native_k': 4})
