"""Store each frozen expert mask once so small controllers read bounded tables."""
from pathlib import Path
from functools import lru_cache
from .contracts import check_files, digest, load, publish, require


def reference(campaign, model, mask):
    if mask is None or "reference" in mask:
        return mask
    key=tuple((str(l),tuple(mask[str(l)])) for l in campaign['models'][model]['layers'])
    return dict(_reference(campaign['root'],model,key))


@lru_cache(maxsize=1024)
def _reference(root,model,key):
    mask={l:list(ids) for l,ids in key}
    mask_sha=digest(mask)
    path=Path(root)/model/"mask_objects"/(mask_sha+".json")
    value=publish(path,{"mask":mask})
    return {"reference":str(path),"sha256":value["sha256"],"mask_sha256":mask_sha,
            "counts":[len(ids) for _,ids in key]}


def resolve(mask):
    if mask is None or "reference" not in mask:
        return mask
    p=Path(mask['reference']);s=p.stat()
    return _resolve(mask['reference'],mask['sha256'],mask['mask_sha256'],tuple(mask['counts']),
        (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))


@lru_cache(maxsize=1024)
def _resolve(path,sha,mask_sha,counts,fingerprint):
    value=load(path)
    require(value["sha256"]==sha and digest(value["mask"])==mask_sha,
            "frozen mask reference differs")
    actual=[len(value['mask'][str(l)]) for l in sorted(map(int,value['mask']))]
    require(actual==list(counts),'frozen mask count vector differs')
    return value["mask"]
