"""Bounded reuse of original all-layer captures; no new native generations."""
import gzip
from pathlib import Path
import numpy as np
from .contracts import require


def archive(source):
    import torch
    import hashlib
    import io
    import json
    path=Path(source["tensor_path"])
    with gzip.open(source["packet_path"],"rt") as stream:
        packet=json.load(stream)
    artifact=packet["tensor_artifact"]
    require(Path(artifact["path"])==path and artifact["bytes"]==path.stat().st_size,"native archive receipt differs")
    compressed=path.read_bytes()
    require(hashlib.sha256(compressed).hexdigest()==artifact["sha256"],"original native archive checksum differs")
    # A bounded allocated-worker memory buffer avoids repeated gzip rewinds by
    # the checkpoint ZIP reader. No archive is copied to another filesystem.
    with gzip.GzipFile(fileobj=io.BytesIO(compressed),mode="rb") as stream:
        unpacked=stream.read(8*1024**3+1)
    require(len(unpacked)<=8*1024**3,"native archive exceeds allocated per-attempt memory bound")
    result=torch.load(io.BytesIO(unpacked),map_location="cpu",weights_only=True)
    return result


def native_capture(campaign,model,row,source):
    raw=archive(source)
    ids=np.asarray(raw["ids"])
    weights=np.asarray(raw["weights"].float())
    inputs=np.asarray(raw["reasoning_tokens"],dtype=int)
    positions=inputs+1
    layers=campaign["models"][model]["layers"]
    require(ids.shape==weights.shape and ids.shape[0]==len(layers),"native archive geometry differs")
    start=row["source"]["prefix_tokens"]
    keep=(positions>=start)&(positions<start+row["window"])
    capture={}
    for layer in row["capture_layers"]:
        i=layers.index(layer)
        capture[str(layer)]={"positions":positions[keep].tolist(),"native_ids":ids[i,inputs[keep]].tolist(),
            "executed_ids":ids[i,inputs[keep]].tolist(),
            "native_weights":weights[i,inputs[keep]].tolist(),"executed_weights":weights[i,inputs[keep]].tolist(),
            "active":[False]*int(keep.sum()),
            "coverage":"original saved reasoning-input positions; missing positions remain missing"}
    return {"capture":capture,"native_source":source["tensor_path"],
        "position_convention":"next-token prediction; original input completion position + 1",
        "all_layer_capture_reused":True}
