"""One locked submission journal, exact readback and afterany dependencies."""
import os
from pathlib import Path
import re
import subprocess
import time
from .common import REPO, digest, load, locked, publish, require, verified
from moe_exp.final_routing.scheduler import command, job_fields, duration_minutes






class Slurm:
    def __init__(self, config, run=command):
        self.config, self.root, self.run = config, Path(config.get('control_root', config['root'])), run

    def read_current(self, job_id, card):
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

    def readback(self, job_id, card):
        try:
            return self.read_current(job_id, card)
        except subprocess.CalledProcessError as error:
            if 'Invalid job id' not in error.stderr:
                raise
        columns = 'JobIDRaw,JobName%100,Account,Partition,QOS,ReqCPUS,NTasks,NNodes,ReqTRES%160,TimelimitRaw,State,ElapsedRaw,Start,AllocTRES%160'
        raw = self.run(['sacct', '-nP', '-X', '-j', job_id, '--format=' + columns])
        rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == job_id]
        require(len(rows) == 1, 'job vanished without unique native accounting: ' + job_id)
        jid, name, account, partition, qos, cpus, tasks, nodes, tres, minutes, state, elapsed, start, allocated = rows[0][:14]
        require(name == card['job_name'] and account == self.config['slurm']['account'], 'historical job identity differs')
        require(partition == self.config['slurm']['partition'] and qos == self.config['slurm']['qos'], 'historical scheduling differs')
        require(int(cpus) == card['cpus'] and int(nodes) == 1 and tasks in ('', '1'), 'historical CPU/node shape differs')
        require(int(minutes) == card['minutes'], 'historical time limit differs')
        gpu = re.search(r'(?:^|,)gres/gpu=(\d+)', tres)
        require((int(gpu[1]) if gpu else 0) == card['gpus'], 'historical GPU shape differs')
        memory = re.search(r'(?:^|,)mem=([\d.]+)([GM])', tres)
        require(memory is not None, 'historical memory request unavailable')
        gib = float(memory[1]) / (1024 if memory[2] == 'M' else 1)
        require(gib == card['memory_gib'], 'historical memory differs')
        fields = {'JobId': jid, 'JobName': name, 'JobState': state.split()[0], 'Reason': 'Terminal',
                  'Account': account, 'Partition': partition, 'QOS': qos, 'StartTime': start,
                  'RunTime': elapsed, 'AllocTRES': allocated, 'ReqTRES': tres}
        require(fields['JobState'] not in ('PENDING', 'RUNNING', 'COMPLETING'), 'missing live job cannot be treated as terminal')
        return {'job_id': jid, 'fields': fields, 'raw': raw, 'readback_source': 'sacct'}

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
        if not test_only:
            reservation = verified(self.root / 'reservations' / (digest(card) + '.json'))
            require(reservation['card_sha256'] == digest(card) and reservation['phase'] == card['budget_phase'],
                    'submission lacks its exact allocation reservation')
        cfg = self.config['slurm']
        folder = self.root/'submissions'/digest(card)
        logs = self.root/'logs'
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        args = ['sbatch', '--test-only' if test_only else '--parsable',
                '--account='+cfg['account'], '--partition='+cfg['partition'], '--qos='+cfg['qos'],
                '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(card['cpus']),
                '--mem='+str(card['memory_gib'])+'G', '--time='+str(card['minutes']),
                '--job-name='+card['job_name'], '--chdir='+self.config.get('code_root', str(REPO)),
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
        args += [str(Path(self.config.get('code_root', str(REPO)))/'scripts/study/job.sh'), self.config['environment'],
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




def card(config, stage, model='all', *, attempt=0, shard=0, dependencies=(), required_cards=(), **extra):
    gpu = stage in ('engine_check', 'qualify', 'reader_pilot', 'generate', 'annotate', 'lfm_native', 'lfm_capture')
    shape = config['slurm']['gpu' if gpu else 'cpu']
    minutes = (config['slurm']['qualification_minutes'] if stage == 'qualify' else
               config['slurm']['reader_pilot_minutes'] if stage == 'reader_pilot' else
               config['slurm']['validation_minutes'] if stage in ('validate', 'prepare', 'advance', 'admit', 'finalize') else
               config['slurm']['production_minutes'])
    value = {'stage': stage, 'model': model, 'attempt': attempt, 'shard': shard,
             'minutes': minutes, **shape, 'dependencies': list(dependencies),
             'config_sha256': config['config_file_sha256'], 'source_commit': config.get('source_commit'), 'required_cards': list(required_cards), **extra}
    value['job_name'] = 'study-'+stage[:9]+'-'+model+'-'+digest(value)[:10]
    return value
