"""Bounded source-snapshot PDF build and read-back of authorized destinations."""
from pathlib import Path
import json
import subprocess
import sys
import re
import shutil
import hashlib
import os
from .contracts import REPO, STAGING, check_files, file_sha, load, locked, publish, require

PAPER=REPO/'paper/6a510c6db0d9b08aa438c142'
INSPECTOR=STAGING/'verification-paper-tectonic-0.17.0/pdf-inspection'


def command(args,cwd,env=None):
    try:
        result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=90,env=env)
    except subprocess.TimeoutExpired as exc:
        exc.publication_diagnostic={"command":args[:2],"timeout_seconds":90,
            "stderr":str(exc.stderr or "")[-3000:]}
        raise
    if result.returncode:
        error=RuntimeError('publication command failed: '+args[0]+' '+args[1])
        error.publication_diagnostic={'exit_code':result.returncode,
            'stderr':re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1[redacted]@',result.stderr[-6000:]),
            'command':args[:2]}
        raise error
    return result.stdout.strip()


def push(directory,branch,paths,message):
    require(command(['git','branch','--show-current'],directory)==branch,'publication branch differs')
    command(['git','add','--',*paths],directory)
    changed=command(['git','diff','--cached','--name-only','--',*paths],directory)
    if changed:command(['git','commit','--only','-m',message,'--',*paths],directory)
    local=command(['git','rev-parse','HEAD'],directory)
    command(['git','push','origin','HEAD:refs/heads/'+branch],directory)
    remote=command(['git','ls-remote','--exit-code','origin','refs/heads/'+branch],directory).split()[0]
    require(local==remote,'published commit readback differs')
    return {'commit':local,'branch':branch,'remote_readback':remote}


def build_local(campaign,report,binding):
    """Commit a source-bound PDF and immutable destination snapshots locally."""
    root=Path(campaign['root']);path=root/'publication'/report['sha256']/'LOCAL_PDF.json'
    if path.exists():return load(path)
    with locked(root,'publication'):
        if path.exists():return load(path)
        check_files(binding['files'])
        build=STAGING/('research-r6-paper-'+report['sha256'][:16])
        if not (build/'BUILD.json').exists():
            # An interrupted build gets one fresh source snapshot; preserve the
            # interrupted directory and its diagnostics.
            if build.exists():build=build.with_name(build.name+'-export-recovery')
            require(not build.exists(),'both bounded PDF build attempts were exhausted')
            command([sys.executable,'-B',str(REPO/'scripts/paper/build_campaign_paper_r3.py'),
                     '--only-cached','--output',str(build)],REPO)
        receipt=json.loads((build/'BUILD.json').read_text())
        require(receipt['state']=='COMPILED_FULL_CLASS' and not receipt['changed_sources'],
                'source-bound PDF compilation failed')
        pdf=Path(receipt['pdf_path']);require(file_sha(pdf)==receipt['pdf_sha256'],'compiled PDF changed')
        check_files(receipt['canonical_source_files'])
        sys.path.insert(0,str(INSPECTOR))
        import fitz
        doc=fitz.open(pdf)
        require(len(doc)>0 and any(page.get_text().strip() for page in doc),'PDF contains no text')
        pages=[i for i,page in enumerate(doc) if 'Campaign v3 observed' in page.get_text()
               or 'Accuracy and resident memory' in page.get_text() or 'Central steering' in page.get_text()]
        previews=[]
        for i in sorted({0,*pages}):
            preview=build/('page-'+str(i+1)+'.png');doc[i].get_pixmap(matrix=fitz.Matrix(1,1)).save(preview)
            previews.append({'page':i+1,'path':str(preview),'sha256':file_sha(preview)})
        inspection={'pages':len(doc),'text_pages_checked':len(doc),'rendered_pages':previews,
                    'visual_review':'rendered previews retained; no human visual review inferred'}
        doc.close()
        snapshot=build/'publication-source'; snapshot.mkdir(exist_ok=True)
        names=[Path(p) for p in receipt['canonical_source_files']]
        names += [REPO/'paper/wiki/CAMPAIGN_V3_STATUS.md']
        files={}
        for source in names:
            if not source.is_file():continue
            allowed=source.suffix in ('.tex','.bib','.cls','.sty','.png','.svg','.pdf','.md') or source.name in (
                'CURRENT.json','CURRENT-memory.json','CURRENT-central.json','CURRENT-sensitivity.json',
                'SOURCES.json','FIGURES.json','REPORT.json','FINAL_SCIENTIFIC_DECISION.json','EXHIBITS_BINDING.json')
            if not allowed:continue
            relative=source.relative_to(REPO);dest=snapshot/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            files[str(relative)]=file_sha(dest)
        # These small provenance manifests are outside TeX's dependency list.
        for source in (PAPER/'generated/research_campaign_v3').rglob('*.json'):
            if source.name not in ('CURRENT.json','CURRENT-memory.json','CURRENT-central.json','CURRENT-sensitivity.json',
                                   'EXHIBITS_BINDING.json','FIGURES.json','REPORT.json','FINAL_SCIENTIFIC_DECISION.json'):continue
            if source.stat().st_size>16*1024**2:continue
            relative=source.relative_to(REPO);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest);files[str(relative)]=file_sha(dest)
        pdf_relative=Path('report')/campaign['campaign_id']/('MANUSCRIPT-'+report['sha256'][:16]+'.pdf')
        dest=snapshot/pdf_relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(pdf,dest)
        files[str(pdf_relative)]=file_sha(dest)
        return publish(path,{'report_sha256':report['sha256'],'build_receipt':str(build/'BUILD.json'),
            'build_sha256':file_sha(build/'BUILD.json'),'pdf_path':str(pdf),'pdf_sha256':file_sha(pdf),
            'inspection':inspection,'snapshot':str(snapshot),'files':files,
            'phase':report.get('phase','final'),'local_manuscript_complete':True,'remote_delivery_complete':False})


def remote_files(directory,commit,files):
    result={}
    for relative,expected in files.items():
        blob=command(['git','rev-parse',commit+':'+relative],directory)
        raw=subprocess.run(['git','cat-file','blob',blob],cwd=directory,capture_output=True,timeout=90,check=True).stdout
        observed=hashlib.sha256(raw).hexdigest()
        if observed!=expected:return None
        result[relative]=observed
    return result


def push_snapshot(campaign, local, destination):
    """Overlay the bound allowlist with a private index, never the working index."""
    source=REPO if destination=='github' else PAPER
    expected='https://github.com/disi-unibo-nlp/mfa-moe.git' if destination=='github' else 'https://git.overleaf.com/6a510c6db0d9b08aa438c142'
    remote=command(['git','remote','get-url','origin'],source)
    require(remote in (expected,expected.replace('https://git.overleaf','https://git@git.overleaf')),'publication destination differs')
    branch='analysis/routing-dynamics' if destination=='github' else 'main'
    prefix=Path('paper/6a510c6db0d9b08aa438c142');files={};originals={}
    for relative,sha in local['files'].items():
        rel=Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts,'publication path escapes checkout')
        if destination=='overleaf':
            if not rel.is_relative_to(prefix):continue
            target=rel.relative_to(prefix)
        else:target=rel
        original=Path(local['snapshot'])/rel
        require(file_sha(original)==sha,'frozen publication source changed')
        files[str(target)]=sha;originals[str(target)]=original
    command(['git','fetch','--no-tags','origin',branch],source)
    base=command(['git','rev-parse','FETCH_HEAD'],source)
    try:observed=remote_files(source,base,files)
    except (RuntimeError,subprocess.CalledProcessError):observed=None
    # A push interrupted after acceptance is recovered by exact file readback.
    if observed is not None:
        return {'commit':base,'branch':branch,'remote_readback':base,'files':observed,
                'reconciled_existing_delivery':True,'source_snapshot':local['snapshot']}
    directory=Path(campaign['root'])/'publication'/local['report_sha256']/destination
    attempt=max([load(p)['attempt'] for p in directory.glob('INTENT-*.json')],default=0)
    staging=STAGING/('r6-publish-'+destination+'-'+local['report_sha256'][:16]);staging.mkdir(exist_ok=True)
    index=staging/('index-'+str(attempt))
    env={**os.environ,'GIT_INDEX_FILE':str(index)}
    command(['git','read-tree',base],source,env)
    for target,original in originals.items():
        blob=command(['git','hash-object','-w',str(original)],source,env)
        command(['git','update-index','--add','--cacheinfo','100644',blob,target],source,env)
    tree=command(['git','write-tree'],source,env)
    commit=command(['git','commit-tree',tree,'-p',base,'-m','Publish research r6 '+local['phase']+' observed results'],source,env)
    publish(directory/('COMMIT-'+str(attempt)+'.json'),{'commit':commit,'parent':base,'files':files,
        'isolated_index':str(index),'report_sha256':local['report_sha256'],'destination':destination})
    command(['git','push','origin',commit+':refs/heads/'+branch],source)
    head=command(['git','ls-remote','--exit-code','origin','refs/heads/'+branch],source).split()[0]
    require(head==commit,'publication commit readback differs')
    observed=remote_files(source,head,files)
    require(observed is not None,'publication file readback differs')
    return {'commit':commit,'branch':branch,'remote_readback':head,'files':observed,'source_snapshot':local['snapshot']}


def deliver(campaign, report, binding=None):
    root=Path(campaign['root']);directory=root/'publication'/report['sha256']
    local=load(directory/'LOCAL_PDF.json')
    outcomes={}
    with locked(root,'publication'):
        for destination in ('overleaf','github'):
            final=directory/destination/'DESTINATION.json'
            if final.exists():outcomes[destination]=load(final);continue
            for attempt in range(2):
                receipt=directory/destination/f'ATTEMPT-{attempt}.json'
                if receipt.exists():continue
                # INTENT precedes the external write. On interruption inspect the
                # exact bound checkout/remote commit before this single recovery.
                intent=directory/destination/f'INTENT-{attempt}.json'
                if not intent.exists():publish(intent,{'destination':destination,'report_sha256':report['sha256'],'attempt':attempt})
                try:
                    value={'status':'complete',**push_snapshot(campaign,local,destination)}
                except Exception as exc:
                    value={'status':'publication_failed','error_type':type(exc).__name__,
                        'error':str(exc)[:1000],'diagnostic':getattr(exc,'publication_diagnostic',None)}
                publish(receipt,{'destination':destination,'attempt':attempt,**value})
                if value['status']=='complete':break
            attempts=[load(p) for p in sorted((directory/destination).glob('ATTEMPT-*.json'))]
            outcomes[destination]=publish(final,{'destination':destination,'status':'complete' if any(a['status']=='complete' for a in attempts) else 'publication_failed',
                'attempts':attempts,'bounded_retry_exhausted':not any(a['status']=='complete' for a in attempts)})
        final=publish(directory/'PUBLICATION.json',{'status':'complete' if all(v['status']=='complete' for v in outcomes.values()) else 'publication_failed',
            'destinations':outcomes,'report_sha256':report['sha256'],'scientific_progression_affected':False})
        if report.get('phase')=='final' and not report.get('snapshot'):
            publish(root/'PAPER_REMOTE_DELIVERY.json',{'complete':final['status']=='complete',
                'report_sha256':report['sha256'],'publication_sha256':final['sha256'],
                'github':outcomes['github']['status'],'overleaf':outcomes['overleaf']['status']})
            if final['status']=='complete':publish(root/'PAPER_DELIVERY.json',{'complete':True,
                'report_sha256':report['sha256'],'remote_receipt_sha256':final['sha256']})
        return final
