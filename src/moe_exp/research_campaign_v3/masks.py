"""Store each frozen expert mask once so small controllers read bounded tables."""
from pathlib import Path
from .contracts import check_files, digest, load, publish, require


def reference(campaign, model, mask):
    if mask is None or "reference" in mask:
        return mask
    path=Path(campaign["root"])/model/"mask_objects"/(digest(mask)+".json")
    value=publish(path,{"mask":mask})
    return {"reference":str(path),"sha256":value["sha256"],"mask_sha256":digest(mask),
            "counts":[len(mask[str(l)]) for l in campaign["models"][model]["layers"]]}


def resolve(mask):
    if mask is None or "reference" not in mask:
        return mask
    value=load(mask["reference"])
    require(value["sha256"]==mask["sha256"] and digest(value["mask"])==mask["mask_sha256"],
            "frozen mask reference differs")
    actual=[len(value['mask'][str(l)]) for l in sorted(map(int,value['mask']))]
    require(actual==mask['counts'],'frozen mask count vector differs')
    return value["mask"]
