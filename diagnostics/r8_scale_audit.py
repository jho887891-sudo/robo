#!/usr/bin/env python3
"""Read-only CPU audit: can R8 state-bounds normalization erase signed V/Acc?

No models are trained; no input data is written or modified.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np


def original_transform(x, lo, hi):
    """Independent reproduction of r8_contract.normalize_block's arithmetic."""
    return np.clip(2.0 * (np.asarray(x,np.float32)-lo) /
                   (hi-lo+1e-8)-1.0,-1.0,1.0).astype(np.float32)


def train_only_scale(train_x):
    x=np.asarray(train_x,np.float32)
    if x.ndim != 2 or x.shape[1] != 14 or len(x) < 100 or not np.isfinite(x).all():
        raise ValueError("require 100+ finite 14D TRAIN observations")
    return np.maximum(np.quantile(np.abs(x),0.99,axis=0).astype(np.float32),1e-6)


def signed_transform(x, scale):
    return np.clip(np.asarray(x,np.float32)/np.asarray(scale,np.float32),-1,1).astype(np.float32)


def history_arrays(episodes):
    vs, accs = [], []
    for ep in episodes:
        s=np.asarray(ep["state"],np.float32)
        if len(s) < 4: continue
        t=np.arange(2,len(s)-1)
        v=s[t]-s[t-1]
        a=v-(s[t-1]-s[t-2])
        vs.append(v);accs.append(a)
    if not vs: raise ValueError("no valid history frames")
    return np.concatenate(vs),np.concatenate(accs)


def summarize(signal, old, new):
    x=np.asarray(signal,np.float32)
    result={"rows":len(x)}
    for name, norm in (("original",old),("candidate",new)):
        result[name+"_floor_fraction"]=(norm<=-1.0+1e-7).mean(axis=0).tolist()
        result[name+"_ceiling_fraction"]=(norm>=1.0-1e-7).mean(axis=0).tolist()
        result[name+"_unique_per_dim"]=[int(len(np.unique(norm[:,i]))) for i in range(14)]
    result["negative_velocity_original_floor"]=[
        float((old[x[:,i]<-1e-8,i] <= -1+1e-7).mean()) if np.any(x[:,i]<-1e-8) else None
        for i in range(14)]
    result["positive_velocity_original_floor"]=[
        float((old[x[:,i]>1e-8,i] <= -1+1e-7).mean()) if np.any(x[:,i]>1e-8) else None
        for i in range(14)]
    return result


def run(bounds_path, output_path):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"diagnostics"))
    sys.path.insert(0,str(root/"code/xpolicylab/policy/OpenVLA_Pure"))
    import r6_data as D
    import r8_contract as C
    b=np.load(bounds_path)
    lo=np.asarray(b["pmin"],np.float32)
    hi=np.asarray(b["pmax"],np.float32)
    if lo.shape!=(14,) or hi.shape!=(14,) or not (hi>lo).all():
        raise ValueError("invalid positional bounds")
    splits=D.load_split()
    tr=D.load_episodes(splits["train"])
    va=D.load_episodes(splits["val"])
    vtr,atr=history_arrays(tr)
    vva,ava=history_arrays(va)
    vscale=train_only_scale(vtr)
    ascale=train_only_scale(atr)
    out={"source_contract":"r8_contract.normalize_block",
         "zero_velocity_as_position":original_transform(np.zeros((1,14),np.float32),lo,hi)[0].tolist(),
         "bounds":{"pmin":lo.tolist(),"pmax":hi.tolist()},
         "candidate_train_only_scale":{"V":vscale.tolist(),"Acc":ascale.tolist()},
         "splits":{}}
    for group,v,a in (("train",vtr,atr),("val",vva,ava)):
        out["splits"][group]={}
        for label,x,scale in (("V",v,vscale),("Acc",a,ascale)):
            old=C.normalize_block(x,lo,hi)
            calc=original_transform(x,lo,hi)
            if not np.allclose(old,calc,atol=1e-7,rtol=0):
                raise AssertionError("independent reproduction mismatch")
            new=signed_transform(x,scale)
            out["splits"][group][label]=summarize(x,old,new)
    out["scope"]="Information audit only. Candidate requires dev-only selection; no claim of better L1."
    with open(output_path,"w") as f: json.dump(out,f,indent=2)
    print("zero physical velocity ->",np.round(out["zero_velocity_as_position"],3).tolist())
    for group in ("train","val"):
        for label in ("V","Acc"):
            s=out["splits"][group][label]
            print(group,label,"old floor",np.round(s["original_floor_fraction"],3).tolist())
            print(group,label,"new floor",np.round(s["candidate_floor_fraction"],3).tolist())
    print("wrote",output_path)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--bounds",default="/home/dgut/pusht_runtime/push_T_bounds.npz")
    ap.add_argument("--out",required=True)
    opt=ap.parse_args()
    run(opt.bounds,opt.out)
