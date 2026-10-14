"""Original reader observations and outcome-blind corrected input coverage."""
from pathlib import Path
import time
from .contracts import MODELS, digest, load, publish, require
from .reader_contract import ReaderUnits, token_ids
from .lifecycle import deadline_before


def price(campaign,deadline):
    from .annotation import Renderer,prepare_contexts,annotate
    from transformers import AutoTokenizer
    root=Path(campaign['root']);a=campaign['annotator']
    if (root/'ANNOTATOR_PRICE.json').exists():return load(root/'ANNOTATOR_PRICE.json')
    original_path=root/'ANNOTATOR_PRICE_TASKS.json'
    coverage_path=root/'ANNOTATOR_PRICE_COVERAGE.json'
    if coverage_path.exists():
        coverage=load(coverage_path);original=load(original_path)
        supplemental=load(root/'ANNOTATOR_PRICE_COVERAGE_TASKS.json')
    else:
        candidates=[]
        for model in MODELS:
            tables=sorted((root/model).glob('ANNOTATION_TASKS-price-*.json'))
            if not tables:
                table=prepare_contexts(campaign,model,phase='price')
                records=table['records']
            else:
                records=list({r['uid']:r for p in tables for r in load(p)['records']}.values())
            units=ReaderUnits(root/model/'annotations')
            candidates.extend((model,r) for r in records if units.state(r['uid']) is None)
        renderer=Renderer(campaign);tokenizer=AutoTokenizer.from_pretrained(a['path'],local_files_only=True)
        sizes={};size_rows=[]
        try:
            for model,row in candidates:
                deadline_before(deadline,120,'correct original reader input sizing')
                conversation=renderer.request('render',inputs=row['inputs'])['conversation']
                n=len(token_ids(tokenizer,conversation,a['config']));sizes[model,row['uid']]=n
                size_rows.append({'model':model,'uid':row['uid'],'input_tokens':n,'eligible':n<a['native_context']})
        finally:renderer.close()
        eligible=sorted([(m,r) for m,r in candidates if sizes[m,r['uid']]<a['native_context']],
                        key=lambda v:(-sizes[v[0],v[1]['uid']],v[1]['uid']))
        if original_path.exists():original=load(original_path)
        elif eligible:
            model=eligible[0][0];chosen=[r for m,r in eligible if m==model][:a['config']['batch_size']]
            original=publish(original_path,{'records':chosen,'original_batch_size':a['config']['batch_size'],'outcomes_used':False})
        else:
            return publish(root/'ANNOTATOR_PRICE.json',{'status':'upstream_unavailable','reason':'no untouched original reader context fits the unchanged context'})
        original_ids={(r['model'],r['uid']) for r in original['records']}
        maximum=max((v['input_tokens'] for v in size_rows if v['eligible']),default=0)
        original_max=max((sizes.get((r['model'],r['uid']),0) for r in original['records']
                          if sizes.get((r['model'],r['uid']),a['native_context'])<a['native_context']),default=0)
        additional=[]
        if eligible and original_max<maximum:
            model=eligible[0][0]
            additional=[r for m,r in eligible if m==model and (m,r['uid']) not in original_ids][:a['config']['batch_size']]
        supplemental=publish(root/'ANNOTATOR_PRICE_COVERAGE_TASKS.json',{'records':additional,
            'original_assignments_not_replaced':True,'outcomes_used':False,'selection':'largest eligible actual token input, original UID tie-breaker'})
        coverage=publish(coverage_path,{'input_sizes':size_rows,'maximum_required_input_tokens':maximum,
            'original_maximum_input_tokens':original_max,'original_table_sha256':original['sha256'],
            'supplemental_table_sha256':supplemental['sha256'],'outcomes_used':False,
            'maximum_input_covered':original_max>=maximum or bool(additional)})
    for table,name in ((original,'original'),(supplemental,'coverage')):
        rows=table['records']
        for model in dict.fromkeys(r['model'] for r in rows):
            annotate(campaign,model,[r for r in rows if r['model']==model],deadline)
        # Commit one batch's disposition before releasing the next batch.
        publish(root/('ANNOTATOR_BATCH-'+name+'.json'),{'table_sha256':table['sha256'],
            'units':len(rows),'original_observations_retained':True,'unpriced_shared_batches_outstanding':0})
    return finish(campaign,original['records']+supplemental['records'],coverage)


def finish(campaign,rows,coverage):
    root=Path(campaign['root']);measured=[];unavailable=[]
    for row in rows:
        value=ReaderUnits(root/row['model']/'annotations').state(row['uid'])
        require(value,'original reader assignment has no disposition')
        if value.get('engine_admitted') is False and value['status']=='upstream_unavailable':
            unavailable.append({'model':row['model'],'uid':row['uid'],'reason':value['reason']});continue
        require('seconds' in value,'original annotator execution price unavailable')
        measured.append({'model':row['model'],'uid':row['uid'],'seconds':value['seconds'],
            'generated_tokens':value['generated_tokens'],'input_tokens':value['input_tokens'],
            'status':value['status'],'bytes':len(str(value).encode()),'load_seconds':value['load_seconds']})
    require(measured and coverage['maximum_input_covered'] and
            max(v['input_tokens'] for v in measured)>=coverage['maximum_required_input_tokens'],
            'reader price lacks eligible maximum-input coverage')
    upper=max(v['seconds']/max(1,v['generated_tokens'])*campaign['annotator']['config']['max_tokens'] for v in measured)
    return publish(root/'ANNOTATOR_PRICE.json',{'status':'complete','records':measured,'unavailable_original_contexts':unavailable,
        'seconds_per_context_upper':upper,'load_seconds':max(v['load_seconds'] for v in measured),
        'bytes_per_context_upper':max(v['bytes'] for v in measured)*4,
        'original_assignments_retained':True,'table_sha256':coverage['original_table_sha256'],
        'coverage_sha256':coverage['sha256'],'maximum_input_covered':True})
