# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Fixed-query experiment contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run the 133-output RN50 linear development pilot on the fixed A1 subset of DEV."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


SEED = 93120261115
ARMS = ("DIR1","DIR2","PLC")
WIDTH = 133


def gradient(logits,labels,features):
    known = labels>=0
    residual = ((logits.sigmoid()-labels)*known) / known.sum((0,2)).clamp(min=1)[None,:,None]
    residual = residual.flatten(1)
    return features.T @ residual, residual.sum(0)


def smoke():
    rng = torch.Generator().manual_seed(SEED)
    x = torch.randn(9,10,generator=rng,dtype=torch.float64)
    w = torch.randn(10,3*WIDTH,generator=rng,dtype=torch.float64,requires_grad=True)
    b = torch.zeros(3*WIDTH,dtype=torch.float64,requires_grad=True)
    y = torch.randint(-1,2,(9,3,WIDTH),generator=rng)
    v = (x@w+b).reshape_as(y)
    l = F.binary_cross_entropy_with_logits(v,y.clamp(min=0).to(v),reduction="none")
    ref = torch.autograd.grad(((l*(y>=0)).sum((0,2))/(y>=0).sum((0,2))).sum(),(w,b))
    for a,expected in zip(gradient(v,y,x),ref,strict=True):
        torch.testing.assert_close(a,expected,rtol=0,atol=1e-14)


def features(path):
    with np.load(path,allow_pickle=False) as d:
        ids = d["ids"]
        vectors = torch.from_numpy(d["features"])
    assert vectors.shape == (len(ids),1024) and torch.isfinite(vectors).all() and (vectors.norm(dim=1)>0).all()
    return ids, F.normalize(vectors)


def fit_weights(x,y,seeds,draw,epochs):
    w=torch.cat([torch.randn(1024,WIDTH,generator=torch.Generator().manual_seed(seed))*.02
                 for seed in seeds for _ in ARMS],1)
    b=torch.zeros(len(seeds)*len(ARMS)*WIDTH)
    mw,mb=torch.zeros_like(w),torch.zeros_like(b)
    started=perf_counter()
    for epoch in range(epochs):
        lr=.1*(1+np.cos(np.pi*epoch/49))/2
        indexes=np.random.default_rng(np.random.SeedSequence([91001,draw,epoch,0])).permutation(len(x))
        for begin in range(0,len(indexes),32):
            batch=indexes[begin:begin+32]
            data,labels=x[batch],y[batch]
            logits=(data@w+b).reshape_as(labels)
            dw,db=gradient(logits,labels,data)
            mw.mul_(.9).add_(dw)
            mb.mul_(.9).add_(db)
            w.add_(mw,alpha=-lr)
            b.add_(mb,alpha=-lr)
        assert torch.isfinite(w).all() and torch.isfinite(b).all()
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            print(draw,epoch+1,perf_counter()-started,flush=True)
    return w,b


def calculate_scores(vectors,w,b,nseed):
    return (vectors@w+b).sigmoid().reshape(len(vectors),len(ARMS)*nseed,WIDTH).transpose(0,1).numpy()


def main(work,fit_cache,dev_cache,dev_ids,limit,nseed,bridge_cache=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    smoke()
    plan = json.loads((work/"PANOPTIC_MASK_SCREEN.json").read_text(encoding="utf-8"))
    masks = [r for r in plan["dev_seeds"] if (work/f"PANOPTIC_MASK_{r}.npz").is_file()]
    fit_ids,x = features(fit_cache)
    dev_sample,q = features(dev_cache)
    expected = np.load(dev_ids,allow_pickle=False)["ids"]
    assert len(dev_sample)==5000 and np.isin(dev_sample,expected).all() and not np.isin(dev_sample,fit_ids).any()
    epochs = limit or 50
    if bridge_cache is not None:
        source_ids,source_features=features(bridge_cache)
        with np.load(work/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as bridge_labels:
            np.testing.assert_array_equal(source_ids,bridge_labels["ids"])
        assert len(source_ids)==5000 and not set(source_ids.tolist())&set(np.r_[fit_ids,dev_sample].tolist())
    prefix="LINEAR_BRIDGE" if bridge_cache is not None else "LINEAR_DEV"
    start = perf_counter()
    rows = []
    for draw in masks[:1] if limit else masks:
        target = work/f"{prefix}_{draw}_{epochs}E_{nseed}R.npz"
        if target.is_file():
            with np.load(target,allow_pickle=False) as f:
                expected_ids=source_ids if bridge_cache is not None else dev_sample
                assert f["scores"].shape==(3*nseed,5000,WIDTH) and np.array_equal(f["ids"],expected_ids)
            rows.append(dict(draw=draw,status="REUSED"))
            continue
        with np.load(work/f"PANOPTIC_MASK_{draw}.npz",allow_pickle=False) as d:
            if not np.array_equal(d["ids"],fit_ids):
                row = {i:j for j,i in enumerate(fit_ids.tolist())}
                order = [row[i] for i in d["ids"]]
                fit_ids,x = fit_ids[order],x[order]
            np.testing.assert_array_equal(fit_ids,d["ids"])
            cats = d["categories"]
            y = torch.from_numpy(np.tile(np.stack([d[a] for a in ARMS],axis=1),(1,nseed,1))).float()
        seeds = [int(np.random.SeedSequence([SEED,11,j+1]).generate_state(1,np.uint64)[0]) for j in range(nseed)]
        started = perf_counter()
        w,b=fit_weights(x,y,seeds,draw,epochs)
        values=calculate_scores(q,w,b,nseed)
        export_ids=dev_sample
        if bridge_cache is not None:
            with np.load(work/f"LINEAR_DEV_{draw}_{epochs}E_{nseed}R.npz",allow_pickle=False) as source:
                np.testing.assert_array_equal(dev_sample,source["ids"])
                np.testing.assert_array_equal(values,source["scores"])
            values=calculate_scores(source_features,w,b,nseed)
            export_ids=source_ids
        assert np.isfinite(values).all()
        temp = target.with_suffix(".partial")
        with temp.open("wb") as f:
            np.savez(f,ids=export_ids,categories=cats,scores=values,arms=np.tile(ARMS,nseed),
                     seeds=np.repeat(np.asarray(seeds,dtype=np.uint64),3))
        temp.replace(target)
        rows.append(dict(draw=draw,status="COMPLETE",epochs=epochs,updates=epochs*126,seconds=perf_counter()-started))
    receipt = dict(status="COMPLETE",wall_seconds=perf_counter()-start, development_images=len(dev_sample),
                   epochs=epochs, endpoints=len(rows)*3*nseed, gradient_qa="PASS", classes=WIDTH, runs=rows,
                   formal_scores_read=0, role="DEV-BRIDGE" if bridge_cache is not None else "DEV-A1",
                   original_A1_scores_exact=bridge_cache is not None)
    (work/f"{prefix}_QA_{epochs}E_{nseed}R.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(receipt,flush=True)


if __name__=="__main__":
    parser=ArgumentParser()
    for name in ("work","fit_cache","dev_cache","dev_ids"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--nseed",type=int,default=6)
    parser.add_argument("--bridge-cache",type=Path)
    options=parser.parse_args()
    if options.bridge_cache is None:
        main(**vars(options))
    else:
        import fcntl
        epochs=options.limit or 50
        done=options.work/f"LINEAR_BRIDGE_QA_{epochs}E_{options.nseed}R.json"
        with (options.work/"LINEAR_BRIDGE_WORKER.lock").open("a") as lease:
            fcntl.flock(lease.fileno(),fcntl.LOCK_EX)
            if done.exists() and json.loads(done.read_text(encoding="utf-8"))["status"]=="COMPLETE":
                print("Existing linear bridge receipt retained",flush=True)
            else:
                main(**vars(options))
