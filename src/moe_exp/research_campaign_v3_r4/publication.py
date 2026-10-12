"""Bounded source-snapshot PDF build and read-back of authorized destinations."""
from pathlib import Path
import json
import subprocess
import sys
from .contracts import REPO, STAGING, check_files, file_sha, load, locked, publish, require

PAPER=REPO/'paper/6a510c6db0d9b08aa438c142'
INSPECTOR=STAGING/'verification-paper-tectonic-0.17.0/pdf-inspection'


def command(args,cwd):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,timeout=90)
    require(result.returncode==0,'publication command failed: '+args[0]+' '+args[1])
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


def deliver(campaign,report,binding):
    """A failed build/publication is export-only; no scientific task is retried."""
    root=Path(campaign['root']);path=root/'publication'/(report['sha256']+'.json')
    if path.exists():return load(path)
    with locked(root,'publication'):
        if path.exists():return load(path)
        check_files(binding['files'])
        build=STAGING/('research-r4-paper-'+report['sha256'][:16])
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
        phase=report.get('phase','final')
        message='Publish research r4 '+phase+' observed results'
        require(command(['git','remote','get-url','origin'],REPO)=='https://github.com/disi-unibo-nlp/mfa-moe.git',
                'GitHub destination differs from standing authorization')
        require(command(['git','remote','get-url','origin'],PAPER) in (
            'https://git.overleaf.com/6a510c6db0d9b08aa438c142',
            'https://git@git.overleaf.com/6a510c6db0d9b08aa438c142'),
                'Overleaf destination differs from standing authorization')
        overleaf_branch=command(['git','branch','--show-current'],PAPER)
        overleaf=push(PAPER,overleaf_branch,
            ['main.tex','generated/research_campaign_v3'],message)
        # Keep bulk FAST files and original captures out of Git. The compact
        # checked paper exhibits and campaign metadata are independently durable.
        paths=['paper/6a510c6db0d9b08aa438c142/main.tex',
               'paper/6a510c6db0d9b08aa438c142/generated/research_campaign_v3',
               'paper/wiki/CAMPAIGN_V3_STATUS.md']
        github=push(REPO,'analysis/routing-dynamics',paths,message)
        return publish(path,{'report_sha256':report['sha256'],'build_receipt':str(build/'BUILD.json'),
            'build_sha256':file_sha(build/'BUILD.json'),'pdf_path':str(pdf),'pdf_sha256':file_sha(pdf),
            'inspection':inspection,'github':github,'overleaf':overleaf,
            'scientific_results_recomputed':False,'all_remote_writes_read_back':True})
