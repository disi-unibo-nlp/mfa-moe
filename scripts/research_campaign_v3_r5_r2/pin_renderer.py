"""Copy the existing original reader's CPU dependency closure into private WORK."""
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    source=Path("/leonardo_scratch/large/userexternal/lmolfett/mfa-moe/qwen-fp-fa-smoke/envs/qwen-fp-fa-dspy-88189c22f214ed65/lib/python3.11/site-packages")
    repo=Path(__file__).resolve().parents[2]
    destination=repo/".research-runtime/original-renderer"
    distributions={canonicalize_name(d.metadata["Name"]):d for d in metadata.distributions(path=[str(source)])}
    pending=[("dspy",set())];selected={}
    while pending:
        name,extras=pending.pop()
        name=canonicalize_name(name)
        if name in selected:
            continue
        d=distributions[name]
        selected[name]=d
        for text in d.requires or []:
            req=Requirement(text)
            if req.marker is None or any(req.marker.evaluate({"extra":extra}) for extra in ["",*extras]):
                pending.append((req.name,set(req.extras)))
    files={}
    for d in selected.values():
        for f in d.files or []:
            p=Path(d.locate_file(f)).resolve()
            if not p.is_relative_to(source) or "__pycache__" in p.parts or not p.is_file():
                continue
            files[p.relative_to(source)]=p
    size=sum(p.stat().st_size for p in files.values())
    if size>512*1024**2:
        raise RuntimeError("original renderer closure exceeds bounded private-copy size")
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    import hashlib
    hashes={}
    for relative,sourcefile in sorted(files.items()):
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        expected=hashlib.sha256(sourcefile.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise RuntimeError("existing private renderer copy differs")
        else:
            shutil.copyfile(sourcefile,target)
            target.chmod(0o600)
        hashes[str(target)]=expected
    document={"source_environment":str(source),"path":str(destination),"bytes":size,
        "versions":{n:d.version for n,d in sorted(selected.items())},"files":hashes,
        "external_pointer_used_at_execution":False}
    manifest=destination.parent/"RENDERER.json"
    text=json.dumps(document,sort_keys=True)+"\n"
    if manifest.exists() and manifest.read_text()!=text:
        raise RuntimeError("immutable original renderer manifest differs")
    manifest.write_text(text);manifest.chmod(0o600)
    print(json.dumps({"packages":len(selected),"files":len(files),"bytes":size,"dspy":selected["dspy"].version,"manifest":str(manifest)}))


if __name__=="__main__":
    main()
