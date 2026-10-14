"""Keep memory cells independent when saved-input or calibration cells are limited."""
from pathlib import Path
import numpy as np
from .contracts import CONFIGURATIONS, load, publish


def finalize(campaign,model):
    from .tasks import freeze_masks,task_table
    root=Path(campaign['root']);contract=campaign['models'][model]
    if (root/model/'TASKS.json').exists():
        return {'status':'complete','tasks_sha256':load(root/model/'TASKS.json')['sha256'],
            'independent_memory_preserved':True,'existing_frozen_table_reused':True}
    if not (root/model/'MASKS.json').exists():
        freeze_masks(campaign,model,np.full((5,len(contract['layers']),contract['experts']),np.nan))
    nominations=load(root/model/'NOMINATIONS.json')['records'];rows=[];missing=[]
    for fold in range(5):
        path=root/model/f'CONFIGURATIONS-fold-{fold}.json'
        if path.exists():rows.extend(load(path)['records']);continue
        missing.append(fold)
        for nomination in nominations:
            if nomination['fold']!=fold:continue
            for sign in (-1,1):
                for index,name in enumerate(CONFIGURATIONS):
                    for arm in ('target','random'):
                        rows.append({'model':model,'fold':fold,'class':nomination['class'],
                            'depth_third':nomination['depth_third'],'layer':nomination['layer'],
                            'sign':sign,'threshold_index':index,'configuration':name,'arm':arm,
                            'windows':nomination['windows'],'status':'upstream_unavailable',
                            'outcomes_used':False,'reason':'required saved-input/calibration component exhausted its finite recovery'})
    config=publish(root/model/'CONFIGURATIONS.json',{'records':rows,'missing_calibration_folds':missing,
        'numerical_values_frozen_before_outcomes':True,'dependent_cells_only':True,
        'mask_sha256':load(root/model/'MASKS.json')['sha256']})
    tasks=task_table(campaign,model,rows)
    return {'status':'complete','tasks_sha256':tasks['sha256'],'configurations_sha256':config['sha256'],
        'independent_memory_preserved':True,'missing_calibration_folds':missing}
