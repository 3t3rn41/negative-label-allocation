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

"""Fit the locked T3 linear heads on sealed, frozen RN50 global features."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V9=HERE.parent
T3=HERE/"T3"


def masked_gradient(logits,labels,features):
    known=labels!=-1
    denom=known.sum(dim=(0,2)).clamp(min=1)
    residual=((logits.sigmoid()-labels)*known)/denom[None,:,None]
    flat=residual.flatten(1)
    return features.T@flat,flat.sum(dim=0)


def check_gradient():
    rng=torch.Generator().manual_seed(72410)
    x=torch.randn(7,5,generator=rng,dtype=torch.float64)
    w=torch.randn(5,4*3,generator=rng,dtype=torch.float64,requires_grad=True)
    b=torch.randn(4*3,generator=rng,dtype=torch.float64,requires_grad=True)
    y=torch.randint(-1,2,(7,4,3),generator=rng,dtype=torch.float64)
    y[:,0]=-1
    logits=(x@w+b).reshape_as(y)
    known=y!=-1
    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y.clamp(min=0),reduction='none')*known)
    independent=torch.autograd.grad((loss.sum(dim=(0,2))/known.sum(dim=(0,2)).clamp(min=1)).sum(),(w,b))
    for got,ref in zip(masked_gradient(logits,y,x),independent,strict=True):
        torch.testing.assert_close(got,ref,rtol=0,atol=1e-14)


def features(path,expected):
    with np.load(path) as z:
        assert z["ids"].tolist()==expected and z["features"].shape==(len(expected),1024)
        value=torch.from_numpy(z["features"])
        classes=z["classes"]
    assert value.dtype==torch.float32 and torch.isfinite(value).all()
    assert (value.norm(dim=1)>0).all()
    return torch.nn.functional.normalize(value),classes


def main(smoke):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    check_gradient()
    if smoke:
        print("masked gradient smoke passed",flush=True)
        return
    cfg=json.loads((T3/"RUN_PROTOCOL.json").read_text(encoding="utf8"))
    receipt=json.loads((T3/"T3_FEATURE_RECEIPT.json").read_text(encoding="utf8"))
    assert receipt["status"]=="FEATURES_SEALED" and not receipt["truth_loaded"]
    assert cfg["status"]=="LOCKED"
    arms,seeds,draws=cfg["arms"],cfg["seeds"],cfg["draws"]
    output=[]
    keys=[]
    receipts=[]
    score,classes=features(T3/"T3_a1_features.npz",cfg["a1_ids"])
    start=time.time()
    for draw in draws:
        with np.load(V9/"coco_design"/f"MASK_{draw}.npz") as f:
            ids=f["ids"].tolist()
            np.testing.assert_array_equal(f["classes"],classes)
            targets=np.stack([f[arm] for arm in arms],axis=1)
        x,check=features(T3/"T3_fit_features.npz",ids)
        assert not set(ids)&set(cfg["a1_ids"])
        np.testing.assert_array_equal(check,classes)
        truth=torch.from_numpy(np.tile(targets,(1,len(seeds),1))).float()
        init=[]
        for seed in seeds:
            g=torch.Generator().manual_seed(seed)
            init.extend([torch.randn(1024,20,generator=g)*.02]*len(arms))
        w=torch.cat(init,dim=1)
        b=torch.zeros(w.shape[1])
        mw,mb=torch.zeros_like(w),torch.zeros_like(b)
        assert truth.shape==(4008,len(seeds)*len(arms),20)
        for epoch in range(50):
            lr=.1*(1+np.cos(np.pi*epoch/49))/2
            # Same deterministic index order as the v9 image pipeline.
            order=np.random.default_rng(np.random.SeedSequence([91001,draw,epoch,0])).permutation(len(x))
            for index in np.array_split(order,np.arange(32,len(x),32)):
                sample,labels=x[index],truth[index]
                logits=(sample@w+b).reshape_as(labels)
                dw,db=masked_gradient(logits,labels,sample)
                mw.mul_(.9).add_(dw)
                mb.mul_(.9).add_(db)
                w.add_(mw,alpha=-lr)
                b.add_(mb,alpha=-lr)
            assert torch.isfinite(w).all() and torch.isfinite(b).all()
            if (epoch+1)%10==0:
                print(draw,"epoch",epoch+1,flush=True)
        vals=(score@w+b).sigmoid().reshape(len(score),len(seeds)*len(arms),20).transpose(0,1).numpy()
        output.append(vals)
        keys.extend((draw,seed,arm) for seed in seeds for arm in arms)
        receipts.append(dict(draw=draw,epochs=50,steps_per_head=6300))
    d,s,a=zip(*keys,strict=True)
    output=np.concatenate(output)
    assert output.shape==(72,5000,20) and np.isfinite(output).all()
    path=T3/"SCORES.npz"
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f,scores=output,ids=np.array(cfg["a1_ids"]),classes=classes,draws=d,seeds=s,arms=a)
    tmp.replace(path)
    (T3/"SCORE_RECEIPT.json").write_text(json.dumps(dict(status="SCORES_SEALED",truth_loaded=False,
        sealed_unix=time.time(),fit_seconds=time.time()-start,endpoints=len(keys),shape=list(output.shape),
        exact_gradient_reference=True,draws=receipts),indent=2),encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)

