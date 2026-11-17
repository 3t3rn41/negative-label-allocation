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

"""Verify M3 prediction and direction calculations on disjoint development folds."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from formal_metrics import pair_columns, read, seal
from joint_replica import Ranking


def cells(truth,target,partner):
    a,b=truth[:,target],truth[:,partner]
    return [(a==v)&(b==w) for v,w in ((1,0),(1,1),(0,0),(0,1))]


def proportion(groups):
    n=np.array([v.sum() for v in groups])
    assert (n>0).all()
    return np.array([n[1]/(n[0]+n[1]),n[3]/(n[2]+n[3])])


def blend(p,r):
    return np.stack(((1-p)*(1-r),(1-p)*r,p*(1-r),p*r),axis=-1)


def four(score,truth,pairs):
    fold_cells=[cells(truth,t,p) for t,p in pairs]
    p=np.stack([proportion(g) for g in fold_cells])
    weights=np.ones((1,len(truth)))
    contrasts=np.empty((*score.shape[:2],len(pairs),4))
    for m,row in enumerate(score):
        for r,starts in enumerate(row):
            for j,((target,_),g) in enumerate(zip(pairs,fold_cells,strict=True)):
                signed=[]
                for a in g[:2]:
                    for b in g[2:]:
                        native=Ranking(starts[j%2,:,target],a,b).calculate(weights)[0]
                        baseline=Ranking(starts[2,:,target],a,b).calculate(weights)[0]
                        signed.append(native-baseline)
                contrasts[m,r,j]=signed
    return contrasts,p,min(int(v.sum()) for g in fold_cells for v in g)


def replay(source,held_out,truth,pairs,scores,rng):
    a,p_source,support_a=four(scores[:,:,:,source],truth[source],pairs)
    b,p_target,support_b=four(scores[:,:,:,held_out],truth[held_out],pairs)
    allocations=np.stack([blend(p_target[:,0],p_target[:,1]),blend(p_source[:,0],.05),blend(p_source[:,0],.5)],axis=1)
    predictions=np.einsum("mrpc,psc->mrps",a,allocations)
    target=np.einsum("mrpc,psc->mrps",b,allocations).mean((0,1))
    p=predictions.mean((0,1))
    naive=np.einsum("mrpc,pc->mrp",a,blend(p_source[:,0],p_source[:,1])).mean((0,1))
    m,r=scores.shape[:2]
    mult_m=rng.multinomial(m,np.full(m,1/m),size=5000)
    mult_r=rng.multinomial(r,np.full(r,1/r),size=5000)
    product=(mult_m[:,:,None]*mult_r[:,None,:])/(m*r)
    replicates=np.einsum("bmr,mrps->bps",product,predictions)
    lo,hi=p-np.quantile(replicates-p,[.975,.025],axis=0)
    predicts_sign=(np.abs(p)>.001)&((lo>0)|(hi<0))
    count=int(predicts_sign.sum())
    matches=int((np.sign(p[predicts_sign])==np.sign(target[predicts_sign])).sum())
    return dict(source_images=len(source),held_out_images=len(held_out),minimum_cell_sources=support_a,
         minimum_cell_held_out=support_b,mae_pp=float(np.abs(p-target).mean()*100),zero_mae_pp=float(np.abs(target).mean()*100),
         natural_baseline_mae_pp=float(np.abs(naive[:,None]-target).mean()*100),
         directional_coverage=count/p.size,directional_hits=matches,directional_outputs=count)


def run(work):
    plan=read(work/"PANOPTIC_MASK_SCREEN.json")
    prompt=read(work/"PROMPT_PILOT_INDEPENDENT_QA.json")
    assert prompt["status"]=="PASS" and prompt["confirmation_scores_read"]==0 and prompt["endpoints"]==96
    with np.load(work/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as bridge:
        ids=bridge["ids"]
        overall_truth=bridge["truth"]
        categories=bridge["categories"].tolist()
    full=pair_columns(dict(categories=categories,targets=plan["chosen"]))
    used=sorted({column for position in full for column in position})
    pairs=[(used.index(a),used.index(b)) for a,b in full]
    truth=overall_truth[:,used]
    values=[]
    for draw in plan["dev_seeds"]:
        with np.load(work/f"PROMPT_DEV_{draw}.npz",allow_pickle=False) as score:
            assert score["categories"].tolist()==categories
            np.testing.assert_array_equal(score["ids"][5000:],ids)
            values.append(score["scores"][:,5000:,used].reshape(4,3,len(ids),len(used)))
    scores=np.asarray(values)
    h=np.logical_or.reduce([(truth[:,a]==1)&(truth[:,b]==1) for a,b in pairs])
    report=[]
    for index in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([93120261115,30,1410,index]))
        left,right=[],[]
        for stratum in (False,True):
            order=rng.permutation(np.flatnonzero(h==stratum))
            left.extend(order[::2])
            right.extend(order[1::2])
        split=[np.asarray(left),np.asarray(right)]
        for first in (0,1):
            result=replay(split[first],split[1-first],truth,pairs,scores,rng)
            report.append(dict(split_seed_index=index,source_side=first,**result))
    seal(work/"M3_DEV_PREFLIGHT_QA.json",dict(status="PASS",scope="Held-out DEV-BRIDGE folds under COCONut labels",
         training_endpoints=96,replays=len(report),confirmation_scores_read=0,rows=report))


if __name__=="__main__":
    cli=ArgumentParser()
    cli.add_argument("work",type=Path)
    args=cli.parse_args()
    run(args.work)
