"""Source-bound pre-production decisions and observed paper exhibits."""
from pathlib import Path
import os
import errno
import shutil
from .contracts import REPO,MODELS,load,publish,file_sha,private_path,require,seal


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


def checked_copy(source, dest):
    source, dest = Path(source), private_path(dest)
    expected = file_sha(source)
    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not dest.exists():
        try:
            os.link(source, dest)
        except OSError as exc:
            if exc.errno != errno.EXDEV: raise
            from .artifacts import atomic_file
            with source.open('rb') as reader, atomic_file(dest) as writer:
                shutil.copyfileobj(reader, writer, 1024**2)
    require(file_sha(dest) == expected, 'immutable copied exhibit differs')
    return expected


def paper_exhibits(campaign,directory,report):
    """Update only v3 generated exhibits; historical paper findings stay intact."""
    root=Path(campaign['root']);directory=Path(directory)
    if root.parent!=REPO/'report' or not root.name.startswith('research-campaign-v3-'):return
    base=REPO/'paper/6a510c6db0d9b08aa438c142/generated/research_campaign_v3'
    target=base/campaign['campaign_id']/report['sha256']
    files={};references={};copied=0
    names={'REPORT.json','FINAL_SCIENTIFIC_DECISION.json','FIGURES.json','paper_tables.csv',
           'paper_tables.tex','plotting_tables.parquet'}
    for source in sorted(directory.rglob('*')):
        if not source.is_file() or source.name.startswith('PAPER_DELIVERY'):continue
        selected=source.name in names or (source.parent.name=='figures' and source.suffix in ('.pdf','.svg'))
        if selected and source.stat().st_size <= 16*1024**2:
            copied+=source.stat().st_size
            require(copied<=256*1024**2,'compact publication exceeds declared WORK budget')
            dest=target/source.relative_to(directory)
            files[str(dest)]=checked_copy(source,dest)
        elif source.suffix in ('.json','.parquet','.csv','.tex'):
            references[str(source)]={'sha256':file_sha(source),'bytes':source.stat().st_size}
    binding=publish(target/'EXHIBITS_BINDING.json',{'report_sha256':report['sha256'],'campaign_sha256':campaign['sha256'],
        'files':files,'scientific_outputs_by_reference':references,'source_output':str(directory),'original_findings_preserved':True})
    complete='complete' if report['SCIENCE_COMPLETE'] else 'incomplete'
    phase=report.get('phase')
    snapshot=(phase+' results' if phase in ('memory','central','sensitivity') else
              'October 15 snapshot' if report['snapshot'] else 'terminal report')
    text=('\\paragraph{Campaign v3 observed closeout.} The seven-model '+snapshot+
        ' retains '+str(report['cells'])+' configured cells. Scientific coverage is '+complete+
        '. Failed and unavailable cells remain explicit. Paired accuracy, token-cost and causal contrasts, '
        'unknown bounds, support, and source-bound PDF/SVG figures are saved with the campaign. '
        'Memory values require matching full-context physical qualification; an empty panel establishes no measured savings. '
        'Intervals crossing zero do not establish equivalence.\\par\n')
    replace_text(base/'status.tex',text)
    pointer=seal({'campaign_id':campaign['campaign_id'],'report_sha256':report['sha256'],
        'binding':str(target/'EXHIBITS_BINDING.json'),'binding_sha256':binding['sha256'],'phase':phase})
    if phase in ('memory','central','sensitivity'):
        replace_text(base/('CURRENT-'+phase+'.json'),__import__('json').dumps(pointer,sort_keys=True)+'\n')
    # The main-paper result slots use only copied, verified vector exhibits.
    # Preserve the memory delivery when central/diagnostic results arrive.
    sources=[binding]
    previous_phases=('memory','central') if phase=='sensitivity' else ('memory',) if phase=='central' else ()
    for previous_phase in previous_phases:
        pointer_path=base/('CURRENT-'+previous_phase+'.json')
        if pointer_path.exists():
            previous=load(pointer_path)
            if previous['campaign_id']==campaign['campaign_id']:sources.append(load(previous['binding']))
    memory=[];central=[]
    for source in sources:
        from .contracts import check_files
        check_files(source['files'])
        memory.extend(p for p in source['files'] if p.endswith('-memory.pdf'))
        central.extend(p for p in source['files'] if p.endswith('-central-primary.pdf'))
    body=('\\section{Observed continuation results}\n'
        'These results use the frozen seven-model study. Missing cells remain unavailable; '
        'qualification and scientific completion are reported separately. '
        'Accuracy uses the original strict scoring contract, with unknown-answer bounds. '
        'Intervals use 10,000 paired family resamples and equal benchmark/family weights. '
        'One replicate does not estimate run-to-run variability.\\par\n')
    if memory:
        body+='\\subsection{Accuracy and resident memory}\nThe fixed evaluation has 420 questions in 400 families. '
        body+='Five pruning methods and both layer allocations are compared at 12.5, 25 and 50 percent removal, alongside native. '
        body+='Resident bytes come from qualified full-context fixtures with fixed KV capacity. '
        body+='Same-budget contrasts, paired uncertainty and unknown bounds are retained in the source-bound tables.\\par\n'
    for filename in sorted(set(memory)):
        relative=Path(filename).relative_to(REPO/'paper/6a510c6db0d9b08aa438c142')
        body+='\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{'+str(relative)+'}\\caption{Accuracy versus measured resident memory; unavailable comparisons are explicit.}\\end{figure}\n'
    if central:
        body+='\\subsection{Central steering}\nThe 129-family cohort retains all seven classes, three depths, both signs and native/random controls. '
        body+='First-complete-sentence response and post-pulse target occupancy share one simultaneous family over 1,176 contrasts and 2,352 bound components. '
        body+='Dwell, transitions, re-entry, recovery and target exits have separate supporting families; accuracy and tokens are reported jointly. '
        body+='Small or imprecise effects remain unresolved.\\par\n'
    for filename in sorted(set(central)):
        relative=Path(filename).relative_to(REPO/'paper/6a510c6db0d9b08aa438c142')
        body+='\\begin{figure}[htbp]\\centering\\includegraphics[width=\\linewidth]{'+str(relative)+'}\\caption{Central primary identification bounds and study-wide simultaneous uncertainty.}\\end{figure}\n'
    replace_text(base/'main_results.tex',body)
    # Generated observation has a stable section; the review/history is retained.
    page=REPO/'paper/wiki/CAMPAIGN_V3_STATUS.md'
    history=page.read_text() if page.exists() else ''
    marker='<!-- generated-current-campaign -->'
    if marker in history:history=history.split(marker,1)[1].lstrip()
    replace_text(page,
        '# Campaign v3 observed evidence\n\n'+snapshot.capitalize()+': report complete; science '+complete+'.\n\n'+
        'Campaign: `'+root.name+'`. [Source-bound report](../../'+str(directory.relative_to(REPO))+'/REPORT.json). '+
        '[Decision package](../../'+str((root/'PRE_PRODUCTION_DECISION.json').relative_to(REPO))+').\n\n'+
        'All seven models, fixed enrollment, failed cells and unavailable outcomes are retained. '+
        'See the report for observed qualification, paired effects, uncertainty, unknown bounds and accounting. '+
        'Historical valid findings and failed receipts remain preserved.\n\n'+marker+'\n\n'+history)
    replace_text(base/'CURRENT.json',__import__('json').dumps(pointer,sort_keys=True)+'\n')
    return binding
