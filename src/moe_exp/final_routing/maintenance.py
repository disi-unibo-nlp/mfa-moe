"""Remove only this campaign's disposable, terminal allocation caches."""
from pathlib import Path
import os
import shutil
import sys
from .common import publish,require,verified


def cleanup(binding,job_id):
    require(job_id.isdigit() and os.environ.get('SLURM_JOB_ID')==job_id,'cache cleanup needs its own allocation')
    root=Path(binding['root']);target=Path(binding['config']['bulk_root'])/'runtime'/('fr-'+job_id)
    receipt=root/'cache_cleanup'/(job_id+'.json')
    if receipt.exists():return verified(receipt)
    require(not target.is_symlink() and target.parent.resolve()==Path(binding['config']['bulk_root']).resolve()/'runtime',
            'cache target escaped the private campaign runtime directory')
    total=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()) if target.exists() else 0
    if target.exists():shutil.rmtree(target)
    require(not target.exists(),'disposable cache removal did not finish')
    return publish(receipt,{'status':'complete','job_id':job_id,'path':str(target),'bytes':total,
        'scientific_outputs_removed':False,'after_srun_exit':True})


if __name__=='__main__':
    cleanup(verified(sys.argv[1]),os.environ['SLURM_JOB_ID'])
