"""Source-bound pre-production decisions and observed paper exhibits."""
from pathlib import Path
import os
from .contracts import REPO,MODELS,load,publish,file_sha,private_path


def preproduction(campaign,admission,*,hours=None,bytes_=None,missing=()):
    root=Path(campaign['root']);path=root/'PRE_PRODUCTION_DECISION.json'
    if path.exists():return load(path)
    evidence={};support={};limitations={};seconds=0.
    for model in MODELS:
        evidence[model]={}
        for name in ('DERIVED','NATIVE_ACCURACY','SUPPLEMENTARY','ANALYSIS_PRICE','COORDINATION_SHARDS',
                     'REAP_QUALIFICATION','QUALIFICATION','PRICES','PREFIXES'):
            p=root/model/(name+'.json')
            if p.exists():evidence[model][name]={'path':str(p),'sha256_file':file_sha(p)}
        p=root/model/'PREFIXES.json'
        if p.exists():
            rows=load(p)['records'];support[model]={'frozen_prefixes':len(rows),
                'available_prefixes':sum(r.get('prefix') is not None for r in rows),
                'independent_families':len({r['family'] for r in rows}),
                'unavailable_prefixes_retained':True}
        p=root/model/'REAP_QUALIFICATION.json'
        if p.exists():limitations[model]=load(p)
        p=root/model/'ANALYSIS_PRICE.json'
        if p.exists():seconds+=load(p)['seconds']
    return publish(path,{'campaign_sha256':campaign['sha256'],
        'status':'qualified_and_admitted' if admission and admission.get('admitted') else 'incomplete_or_infeasible',
        'admission':admission,'observational_evidence':evidence,'realized_support':support,
        'method_qualification_and_limitations':limitations,'residual_local_hours_before_margin':hours,
        'peak_additional_output_bytes':bytes_,'required_missing_prices':list(missing),
        'measured_analysis_elapsed_seconds':seconds,'runtime_projection':{
            'residual_local_accounting_hours':hours,'execution_prices':'retained original role/shape measurements',
            'queue_uncertainty':'unmeasured; no calendar completion promise'},
        'original_ceiling_local_hours':42050,'residual_includes_analysis_and_operational_owners':True,
        'fixed_samples_may_be_imprecise':True,'undefined_draws_discarded':0,
        'negative_and_imprecise_outcomes_retained':True})


def replace_text(path,text):
    path=private_path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.pending')
    with temporary.open('w') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def paper_exhibits(campaign,directory,report):
    """Update only v3 generated exhibits; historical paper findings stay intact."""
    root=Path(campaign['root']);directory=Path(directory)
    if root.parent!=REPO/'report' or not root.name.startswith('research-campaign-v3-'):return
    base=REPO/'paper/6a510c6db0d9b08aa438c142/generated/research_campaign_v3'
    target=base/directory.relative_to(root)
    files={}
    for source in directory.rglob('*'):
        if source.is_file() and source.suffix in ('.pdf','.svg','.csv','.parquet','.tex','.json'):
            dest=target/source.relative_to(directory);dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            if not dest.exists():os.link(source,dest)
            files[str(dest)]=file_sha(dest)
    publish(target/'EXHIBITS_BINDING.json',{'report_sha256':report['sha256'],'campaign_sha256':campaign['sha256'],
        'files':files,'source_output':str(directory),'original_findings_preserved':True})
    complete='complete' if report['SCIENCE_COMPLETE'] else 'incomplete'
    snapshot='October 15 snapshot' if report['snapshot'] else 'terminal report'
    text=('\\paragraph{Campaign v3 observed closeout.} The seven-model '+snapshot+
        ' retains '+str(report['cells'])+' configured cells. Scientific coverage is '+complete+
        '. Failed and unavailable cells remain explicit. Paired accuracy, token-cost and causal contrasts, '
        'unknown bounds, support, and source-bound PDF/SVG figures are saved with the campaign. '
        'Memory values require matching full-context physical qualification; an empty panel establishes no measured savings. '
        'Intervals crossing zero do not establish equivalence.\\par\n')
    replace_text(base/'status.tex',text)
    replace_text(REPO/'paper/wiki/CAMPAIGN_V3_STATUS.md',
        '# Campaign v3 observed evidence\n\n'+snapshot.capitalize()+': report complete; science '+complete+'.\n\n'+
        'Campaign: `'+root.name+'`. [Source-bound report](../../'+str(directory.relative_to(REPO))+'/REPORT.json). '+
        '[Decision package](../../'+str((root/'PRE_PRODUCTION_DECISION.json').relative_to(REPO))+').\n\n'+
        'All seven models, fixed enrollment, failed cells and unavailable outcomes are retained. '+
        'See the report for observed qualification, paired effects, uncertainty, unknown bounds and accounting. '+
        'Historical valid findings and failed receipts remain preserved.\n')
