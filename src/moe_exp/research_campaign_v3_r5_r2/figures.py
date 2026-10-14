"""Source-bound plotting tables and vector figures with explicit causal strata."""
from pathlib import Path
import numpy as np
from .contracts import MODELS,METHODS,REMOVALS,CLASSES,CONFIGURATIONS,load,publish,digest
from .artifacts import atomic_file,frame


def render(directory,rows,campaign):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'svg.fonttype':'none','svg.hashsalt':campaign['campaign_id']})
    panels=load(directory/'UNCERTAINTY.json')['panels'];figures=[];plotrows=[]
    byid={tuple(p['identity']):p['results'] for p in panels}
    def save(fig,name):
        fig.tight_layout();folder=directory/'figures';folder.mkdir(parents=True,exist_ok=True)
        for kind in ('svg','pdf'):
            path=folder/(name+'.'+kind)
            metadata={'Date':None} if kind=='svg' else {'CreationDate':None,'ModDate':None}
            with atomic_file(path) as stream:fig.savefig(stream,format=kind,metadata=metadata)
            figures.append(str(path))
        plt.close(fig)
    def value(results,name):return results.get(name,{}).get('estimate')
    for model in MODELS:
        if any(r.get('model')==model and r.get('study')=='memory' for r in rows):
            results=byid.get(('memory',model),{})
            fig,axes=plt.subplots(1,2,figsize=(10,3.4))
            for allocation,axis in zip(('equal','unequal'),axes):
                native_memory=value(results,'native:shared:0|resident_bytes')
                native_accuracy=value(results,'native:shared:0|strict_correct')
                plotrows.append({'figure':model+'-memory','model':model,'allocation':allocation,'method':'native',
                    'removal':0.,'resident_bytes':native_memory,'accuracy':native_accuracy,
                    'status':'measured' if native_memory is not None and native_accuracy is not None else 'unavailable'})
                if native_memory is not None and native_accuracy is not None:
                    axis.scatter([native_memory/1024**3],[native_accuracy],marker='*',color='black',label='native')
                for method in METHODS:
                    x=[];y=[]
                    for removal in REMOVALS:
                        arm=f'{method}:{allocation}:{removal}'
                        memory,accuracy=value(results,arm+'|resident_bytes'),value(results,arm+'|strict_correct')
                        plotrows.append({'figure':model+'-memory','model':model,'allocation':allocation,'method':method,
                            'removal':removal,'resident_bytes':memory,'accuracy':accuracy,
                            'accuracy_interval':results.get(arm+'|strict_correct',{}).get('interval'),
                            'accuracy_family_support':results.get(arm+'|strict_correct',{}).get('family_support',0),
                            'unknown_lower':value(results,arm+'|unknown_lower'),'unknown_upper':value(results,arm+'|unknown_upper'),
                            'status':'measured' if memory is not None and accuracy is not None else 'unavailable'})
                        if memory is not None and accuracy is not None:
                            x.append(memory/1024**3);y.append(accuracy)
                            interval=results.get(arm+'|strict_correct',{}).get('interval')
                            if interval:axis.vlines(memory/1024**3,*interval,alpha=.45)
                            lower=value(results,arm+'|unknown_lower');upper=value(results,arm+'|unknown_upper')
                            if lower is not None and upper is not None and lower!=upper:
                                axis.vlines(memory/1024**3,lower,upper,color='grey',linewidth=5,alpha=.25)
                    if x:axis.plot(x,y,'o-',label=method)
                axis.set(title=model+' / '+allocation,xlabel='Qualified process residence (GiB)',ylabel='Strict accuracy',ylim=(0,1))
                if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=6)
                else:axis.text(.5,.5,'Resident-memory comparison unavailable',ha='center',transform=axis.transAxes)
            save(fig,model+'-memory')
            fig,axes=plt.subplots(1,2,figsize=(10,3.4))
            for allocation,axis in zip(('equal','unequal'),axes):
                for method in METHODS:
                    x=[];y=[]
                    for removal in REMOVALS:
                        key=f'{method}:{allocation}:{removal}-minus-native|strict_correct';r=results.get(key,{})
                        if r.get('estimate') is not None:
                            x.append(removal);y.append(r['estimate'])
                            if r.get('interval'):axis.vlines(removal,*r['interval'],alpha=.4)
                        plotrows.append({'figure':model+'-accuracy-differences','model':model,'allocation':allocation,'method':method,'removal':removal,**r})
                    if x:axis.plot(x,y,'o-',label=method)
                axis.axhline(0,color='grey',lw=.7)
                axis.set(title=model+' / '+allocation,xlabel='Removal fraction',ylabel='Pruned minus native strict accuracy')
                if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=6)
            save(fig,model+'-accuracy-differences')
        if not any(r.get('model')==model and r.get('study')=='causal' for r in rows):continue
        primary=next((p for p in panels if p['identity'][0]=='steering-primary'),None)
        if primary:
            fig,axes=plt.subplots(2,2,figsize=(11,8))
            for ei,endpoint in enumerate(('first','recovery_w2')):
                for ci,control in enumerate(('native','random')):
                    axis=axes[ei,ci]
                    for ti,target in enumerate(CLASSES):
                        for depth in range(3):
                            for si,sign in enumerate((-1,1)):
                                key=f'{model}|{target}|{depth}|{sign}|target-minus-{control}|{endpoint}'
                                value_=primary['identified_effects'][key];position=ti*6+depth*2+si
                                lo,hi=value_['sample_bounds'];interval=value_['simultaneous_range']
                                if lo is not None and hi is not None:axis.vlines(position,lo,hi,color='tab:blue',linewidth=4,alpha=.5)
                                if interval:axis.vlines(position,*interval,color='black',linewidth=.8)
                                plotrows.append({'figure':model+'-central-primary','model':model,'target':target,
                                    'depth_third':depth,'sign':sign,'control':control,'endpoint':endpoint,**value_})
                    axis.axhline(0,color='grey',linewidth=.5)
                    axis.set_xticks([i*6+2.5 for i in range(7)],CLASSES,rotation=35,ha='right')
                    axis.set(title=endpoint+' / target minus '+control,ylabel='Target response difference',ylim=(-1.1,1.1))
            fig.suptitle(model+' / central primary: identification bounds (thick); simultaneous range (thin)')
            save(fig,model+'-central-primary')
        configurations=[0]+sorted({r.get('threshold_index') for r in rows if r.get('model')==model and r.get('study')=='causal' and
                                  r.get('status') in ('complete','reused') and r.get('threshold_index',0)>0})
        for config in configurations:
            fig,axes=plt.subplots(3,2,figsize=(9,9))
            names=[f'{arm}:w{i}' for arm in ('target','random') for i in range(3)]
            for depth in range(3):
                for sj,sign in enumerate((-1,1)):
                    results=byid.get(('causal',model,depth,sign,config),{});matrix=np.full((7,6),np.nan)
                    for i,target in enumerate(CLASSES):
                        for j,name in enumerate(names):
                            arm,window=name.split(':');r=results.get(f'{target}:{arm}|coverage:{window}',{});v=r.get('estimate')
                            if v is not None:matrix[i,j]=v
                            plotrows.append({'figure':model+'-coverage','model':model,'configuration':CONFIGURATIONS[config],
                                'depth_third':depth,'sign':sign,'target':target,'arm':arm,'window':window,**r})
                    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('lightgrey')
                    axes[depth,sj].imshow(matrix,vmin=0,vmax=1,cmap=cmap)
                    axes[depth,sj].set_xticks(range(6),names,rotation=45,ha='right');axes[depth,sj].set_yticks(range(7),CLASSES)
                    axes[depth,sj].set_title(f'Depth {depth+1} / sign {sign:+d}')
            fig.suptitle(f'{model} / {CONFIGURATIONS[config]} / Labelled token coverage; grey = unavailable',y=1.01)
            save(fig,f'{model}-{CONFIGURATIONS[config]}-coverage')
            for metric,label in [('first:','First complete sentence'),('recovery:w2:','Subsequent recovery dynamics')]:
                for contrast in ('target-minus-random','target-minus-native'):
                    fig,axes=plt.subplots(3,2,figsize=(11,10))
                    for depth in range(3):
                        for sj,sign in enumerate((-1,1)):
                            axis=axes[depth,sj];results=byid.get(('causal',model,depth,sign,config),{})
                            matrix=np.full((7,7),np.nan)
                            for i,target in enumerate(CLASSES):
                                for j,dest in enumerate(CLASSES):
                                    key=f'{target}:{contrast}|{metric}{dest}' if contrast=='target-minus-random' else f'{target}:target|native_difference:{metric}{dest}'
                                    r=results.get(key,{});v=r.get('estimate')
                                    if v is not None:matrix[i,j]=v
                                    plotrows.append({'figure':model+'-response','model':model,'configuration':CONFIGURATIONS[config],
                                        'depth_third':depth,'sign':sign,'contrast':contrast,'target':target,'destination':dest,'metric':metric,**r})
                            cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('lightgrey')
                            im=axis.imshow(matrix,cmap=cmap,vmin=-1,vmax=1)
                            axis.set_xticks(range(7),CLASSES,rotation=45,ha='right');axis.set_yticks(range(7),CLASSES)
                            axis.set(title=f'Depth {depth+1} / sign {sign:+d}',xlabel='Observed class',ylabel='Target class')
                    fig.suptitle(f'{model} / {CONFIGURATIONS[config]} / {label} / {contrast}; grey = unavailable',y=1.01)
                    save(fig,f'{model}-{CONFIGURATIONS[config]}-{metric.split(":")[0]}-{contrast}')
    table=publish(directory/'PLOTTING_TABLES.json',{'records':plotrows,'uncertainty_sha256':load(directory/'UNCERTAINTY.json')['sha256'],
        'campaign_sha256':campaign['sha256'],'analysis_binding':campaign.get('analysis_binding'),'rows_sha256':digest(rows),
        'signs_depths_configurations_pooled':False})
    frame(directory/'plotting_tables.parquet',[{k:str(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in plotrows])
    publish(directory/'FIGURES.json',{'files':{p:__import__('hashlib').sha256(Path(p).read_bytes()).hexdigest() for p in figures},
        'plotting_tables_sha256':table['sha256'],'inference_reruns':0})
    return figures
