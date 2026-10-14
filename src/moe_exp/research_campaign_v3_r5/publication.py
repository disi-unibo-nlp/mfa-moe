"""Bounded source-snapshot PDF build and read-back of authorized destinations."""
from pathlib import Path
import json
import subprocess
import sys
import re
import shutil
from .contracts import REPO, STAGING, check_files, file_sha, load, locked, publish, require

PAPER=REPO/'paper/6a510c6db0d9b08aa438c142'
INSPECTOR=STAGING/'verification-paper-tectonic-0.17.0/pdf-inspection'


def command(args,cwd):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=90)
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
        build=STAGING/('research-r5-paper-'+report['sha256'][:16])
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
        names=[PAPER/'main.tex',*(PAPER/'generated/research_campaign_v3').rglob('*'),REPO/'paper/wiki/CAMPAIGN_V3_STATUS.md']
        files={}
        for source in names:
            if not source.is_file():continue
            relative=source.relative_to(REPO);dest=snapshot/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            files[str(relative)]=file_sha(dest)
        return publish(path,{'report_sha256':report['sha256'],'build_receipt':str(build/'BUILD.json'),
            'build_sha256':file_sha(build/'BUILD.json'),'pdf_path':str(pdf),'pdf_sha256':file_sha(pdf),
            'inspection':inspection,'snapshot':str(snapshot),'files':files,
            'phase':report.get('phase','final'),'local_manuscript_complete':True,'remote_delivery_complete':False})


def push_snapshot(campaign, local, destination):
    """Use a private checkout of the current remote tip; publish exact bound files."""
    source=REPO if destination=='github' else PAPER
    expected='https://github.com/disi-unibo-nlp/mfa-moe.git' if destination=='github' else 'https://git.overleaf.com/6a510c6db0d9b08aa438c142'
    remote=command(['git','remote','get-url','origin'],source)
    require(remote in (expected,expected.replace('https://git.overleaf','https://git@git.overleaf')),'publication destination differs')
    branch='analysis/routing-dynamics' if destination=='github' else command(['git','branch','--show-current'],source)
    command(['git','fetch','origin',branch],source)
    worktree=STAGING/('r5-publish-'+destination+'-'+local['report_sha256'][:16])
    if not worktree.exists():command(['git','worktree','add','--detach',str(worktree),'FETCH_HEAD'],source)
    prefix=Path('paper/6a510c6db0d9b08aa438c142')
    paths=[]
    for relative,sha in local['files'].items():
        rel=Path(relative)
        if destination=='overleaf':
            if not rel.is_relative_to(prefix):continue
            target=rel.relative_to(prefix)
        else:target=rel
        original=Path(local['snapshot'])/rel;require(file_sha(original)==sha,'frozen publication source changed')
        dest=worktree/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
        require(file_sha(dest)==sha,'publication checkout differs');paths.append(str(target))
    command(['git','add','--',*paths],worktree)
    if command(['git','diff','--cached','--name-only'],worktree):
        command(['git','commit','-m','Publish research r5 '+local['phase']+' observed results'],worktree)
    head=command(['git','rev-parse','HEAD'],worktree)
    command(['git','push','origin','HEAD:refs/heads/'+branch],worktree)
    readback=command(['git','ls-remote','--exit-code','origin','refs/heads/'+branch],worktree).split()[0]
    require(head==readback,'publication remote readback differs')
    return {'commit':head,'branch':branch,'remote_readback':readback,'files':local['files'],'source_snapshot':local['snapshot']}


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
        return publish(directory/'PUBLICATION.json',{'status':'complete' if all(v['status']=='complete' for v in outcomes.values()) else 'publication_failed',
            'destinations':outcomes,'report_sha256':report['sha256'],'scientific_progression_affected':False})
