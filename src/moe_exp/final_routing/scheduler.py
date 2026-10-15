"""One locked submission journal, exact readback and afterany dependencies."""
import os
from pathlib import Path
import re
import subprocess
import time
from .common import REPO, digest, load, locked, publish, require, verified


def command(args, *, timeout=45):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SRUN_', 'SBATCH_'))} if args[0] == 'sbatch' else None
    return subprocess.run(list(map(str, args)), text=True, capture_output=True,
                          check=True, timeout=timeout, env=env).stdout.strip()


def job_fields(raw):
    return dict(re.findall(r'(\w+)=([^\s]+)', raw))


class Slurm:
    def __init__(self, config, run=command):
        self.config, self.root, self.run = config, Path(config['root']), run

    def readback(self, job_id, card):
        raw = self.run(['scontrol', 'show', 'job', '-o', job_id])
        fields = job_fields(raw)
        require(fields.get('JobId') == job_id, 'Slurm returned another job')
        require(fields.get('Account') == self.config['slurm']['account'], 'job account differs')
        require(fields.get('UserId','').startswith('lmolfett('),'job user differs')
        require(fields.get('JobName') == card['job_name'], 'job name differs')
        require(fields.get('Partition') == self.config['slurm']['partition'], 'job partition differs')
        require(fields.get('NumCPUs') == str(card['cpus']), 'CPU allocation differs')
        for key, expected in [('QOS', self.config['slurm']['qos']), ('NumTasks', '1')]:
            require(fields.get(key) == expected, key+' allocation differs')
        require(fields.get('NumNodes') in ('1','1-1'),'node allocation differs')
        require(fields.get('MinMemoryNode') == str(card['memory_gib'])+'G', 'memory allocation differs')
        require(duration_minutes(fields.get('TimeLimit', '')) == card['minutes'], 'time limit differs')
        billing=re.search(r'(?:^|,)billing=([\d.]+)',fields.get('ReqTRES',''))
        require(billing and float(billing[1])==card['billing_units'],'allocation billing differs')
        if card['gpus']:
            require(re.search(r'(?:^|,)gres/gpu='+str(card['gpus'])+r'(?:,|$)', fields.get('ReqTRES', '')), 'GPU request differs')
        else:
            require('gres/gpu=' not in fields.get('ReqTRES', ''), 'CPU-only job requests GPUs')
        return {'job_id': job_id, 'fields': fields, 'raw': raw}

    def reconcile(self, folder, card):
        if (folder/'ACCEPTED.json').exists():
            old = verified(folder/'ACCEPTED.json')
            require(old['card_sha256'] == digest(card), 'accepted card changed')
            return old
        if not (folder/'INTENT.json').exists():
            return None
        # Never resubmit an unresolved intent. Reconcile by unique immutable name.
        accepted = folder/'SBATCH.json'
        if accepted.exists():
            ids = [verified(accepted)['job_id']]
        else:
            rows = self.run(['squeue', '-h', '-u', 'lmolfett', '-n', card['job_name'], '-o', '%i']).splitlines()
            ids = sorted(set(v.strip() for v in rows if v.strip().isdigit()))
        if not ids:
            history = self.run(['sacct', '-n', '-X', '-S', load(folder/'INTENT.json')['date'],
                                '--name', card['job_name'], '--format=JobIDRaw', '--parsable2'])
            ids = sorted(set(v.strip().strip('|') for v in history.splitlines()
                             if v.strip().strip('|').isdigit()))
        require(len(ids) == 1, f'unresolved submission intent: {folder}; found {ids}')
        evidence = self.readback(ids[0], card)
        return publish(folder/'ACCEPTED.json', {'card_sha256': digest(card), **evidence})

    def submit(self, card, dependencies=(), *, hold=False, test_only=False):
        cfg = self.config['slurm']
        folder = self.root/'submissions'/digest(card)
        logs = self.root/'logs'
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        args = ['sbatch', '--test-only' if test_only else '--parsable',
                '--account='+cfg['account'], '--partition='+cfg['partition'], '--qos='+cfg['qos'],
                '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(card['cpus']),
                '--mem='+str(card['memory_gib'])+'G', '--time='+str(card['minutes']),
                '--job-name='+card['job_name'], '--chdir='+str(REPO),
                '--output='+str(logs/'%x-%j.out'), '--error='+str(logs/'%x-%j.err')]
        if card['gpus']:
            args += ['--gres=gpu:'+str(card['gpus'])]
        if dependencies:
            require(all(re.fullmatch(r'\d+(?:_\d+)?',str(j)) for j in dependencies), 'invalid dependency job ID')
            args += ['--dependency=afterany:'+':'.join(map(str, dependencies))]
        if hold:
            args.append('--hold')
        card_path = self.root/'cards'/(digest(card)+'.json')
        publish(card_path, card)
        args += [str(REPO/'scripts/slurm/final_routing.sbatch'), self.config['environment'],
                 str(self.root/'BINDING.json'), str(card_path)]
        if test_only:
            return {'test_only': True, 'output': self.run(args)}
        with locked(self.root, 'submission'):
            if (folder/'INTENT.json').exists():
                require(verified(folder/'INTENT.json')['argv'] == args, 'submission dependencies or hold state changed')
            old = self.reconcile(folder, card)
            if old:
                return old
            publish(folder/'INTENT.json', {'card_sha256': digest(card), 'argv': args,
                    'date': time.strftime('%Y-%m-%d'), 'epoch': time.time()})
            raw = self.run(args)
            require(re.fullmatch(r'\d+(;[\w.-]+)?', raw) is not None, 'invalid sbatch acceptance')
            job_id = raw.split(';')[0]
            publish(folder/'SBATCH.json', {'job_id': job_id, 'raw': raw})
            evidence = self.readback(job_id, card)
            return publish(folder/'ACCEPTED.json', {'card_sha256': digest(card), **evidence})


def duration_minutes(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    pieces = clock.split(':')
    require(len(pieces) in (2, 3), 'invalid Slurm time limit')
    if len(pieces) == 2:
        pieces = ['0'] + pieces
    h, m, s = map(int, pieces)
    return int(days)*1440+h*60+m+s/60


def card(config, stage, model='all', *, attempt=0, shard=0, dependencies=(), required_cards=(), **extra):
    gpu = stage in ('qualify', 'reader_pilot', 'generate', 'annotate', 'lfm_native', 'lfm_capture')
    shape = config['slurm']['gpu' if gpu else 'cpu']
    minutes = (config['slurm']['qualification_minutes'] if stage == 'qualify' else
               config['slurm']['reader_pilot_minutes'] if stage == 'reader_pilot' else
               config['slurm']['validation_minutes'] if stage in ('validate', 'prepare', 'advance', 'admit', 'finalize') else
               config['slurm']['production_minutes'])
    value = {'stage': stage, 'model': model, 'attempt': attempt, 'shard': shard,
             'minutes': minutes, **shape, 'dependencies': list(dependencies),
             'config_sha256': config['config_file_sha256'], 'required_cards': list(required_cards), **extra}
    value['job_name'] = 'fr-'+stage[:9]+'-'+model+'-'+digest(value)[:10]
    return value
