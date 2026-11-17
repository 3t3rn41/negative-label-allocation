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

"""Replay background-negative AUC and split natural AUC in sealed B and VOC."""
import argparse
from time import perf_counter

import numpy as np

from replay import (AB, HERE, N_BOOT, ROLE, VOC, Metric, bootstrap_for, design_for, group_codes,
                    read, replay_auc, save, set_num_threads, summarize, table)

METRICS = ("background_auc", "background_fixed_pstar_auc", "partner_negative_auc",
           "background_contribution", "partner_negative_contribution", "natural_auc", "conditional_auc")


def load(stage):
    if stage == "B":
        return AB/"B_SCORES.npz", AB/"B_GOLD.npz", "S0_PLC"
    return VOC/"SCORES.npz", ROLE/"TEST_GOLD.npz", "PLC"


def count_weights(group, weights):
    counts = np.array([weights[:, group == i].sum(axis=1, dtype=np.int32) for i in range(4)])
    assert (counts>0).all()
    n00,n01,n10,n11 = counts
    p = n11/(n10+n11)
    r = n01/(n00+n01)
    return p,r,counts[:,0].tolist()


def metrics(cells, p, r, pstar):
    a10_00, a10_01, a11_00, a11_01, natural = cells.T
    background = (1-p)*a10_00 + p*a11_00
    exposed = (1-p)*a10_01 + p*a11_01
    fixed = (1-pstar)*a10_00 + pstar*a11_00
    bg_contribution = (1-r)*background
    exposed_contribution = r*exposed
    np.testing.assert_allclose(bg_contribution+exposed_contribution,natural,atol=1e-12,rtol=0)
    return np.stack((background,fixed,exposed,bg_contribution,exposed_contribution,natural,a10_01),axis=1)


def direct_check(score, group, weights, values):
    pos = group>=2
    max_error = 0.
    for label, index in [(0,0),(1,2)]:
        signs = np.where(pos,1,np.where(group==label,0,-1))
        ref = Metric(score,signs).compute_auc(weights)
        np.testing.assert_allclose(values[:,index],ref,atol=1e-12,rtol=0)
        max_error = max(max_error,float(np.max(np.abs(values[:,index]-ref))))
    return max_error


def replay(stage, limit):
    start = perf_counter()
    score_path, gold_path, control = load(stage)
    with np.load(score_path) as s, np.load(gold_path) as g:
        ids, classes, scores = s["ids"],s["classes"].tolist(),s["scores"]
        np.testing.assert_array_equal(ids,g["ids"])
        np.testing.assert_array_equal(s["classes"],g["classes"])
        gold = g["truth"]
        keys = list(zip(s["seeds"].tolist(),s["draws"].tolist(),s["arms"].tolist(),strict=True))
    assert len(keys)==len(set(keys))==72 and len(ids)==len(set(ids))
    assert scores.shape==(72,len(ids),len(classes)) and np.isfinite(scores).all()
    weights, streams = bootstrap_for(stage,len(ids),limit)
    indices = streams[0]
    subset = np.unique(np.linspace(0,limit,min(13,limit+1),dtype=int))
    stage_ref = read(HERE/f"{stage}_REPLAY.json")
    with np.load(HERE/f"{stage}_REPLAY.npz") as reference:
        np.testing.assert_array_equal(reference["indices"][:limit],indices)
        expected = reference["effects"][:limit+1]
    seeds, draws, arms = (stage_ref[k] for k in ("seeds","masks","arms"))
    values = np.empty((limit+1,12,2,len(arms),2,3,len(METRICS)))
    paired_summary, per_seed, supports, endpoint_rows = [],[],[],[]
    errors = {"direct_background_and_partner_auc":0.,"prior_natural_fixed_bg_conditional_auc":0.}
    for ei,(evaluation, pairs) in enumerate(design_for(stage,classes).items()):
        for pi,(target,partner,pstar) in enumerate(pairs):
            group = group_codes(gold,target,partner,stage=="VOC")
            p,r,counts = count_weights(group,weights)
            n00,n01,n10,n11 = counts
            supports.append(dict(stage=stage,evaluation=evaluation,target=classes[target],partner=classes[partner],
                                 n00=n00,n01=n01,n10=n10,n11=n11,n_excluded=int((group<0).sum()),
                                 p_natural=p[0],r_natural=r[0],pstar=pstar))
            for (seed,draw,arm),matrix in zip(keys,scores,strict=True):
                si,di,ai = seeds.index(seed),draws.index(draw),arms.index(arm)
                score = matrix[:,target]
                values[:,si,di,ai,ei,pi] = vals = metrics(replay_auc(score,group,weights),p,r,pstar)
                direct_error = direct_check(score,group,weights[subset],vals[subset])
                errors["direct_background_and_partner_auc"] = max(errors["direct_background_and_partner_auc"],direct_error)
                old = expected[:,si,di,ai,ei,pi][:,[2,1,0]]
                check = vals[:,[1,5,6]]
                np.testing.assert_allclose(check,old,atol=1e-12,rtol=0)
                error = float(np.max(np.abs(check-old)))
                errors["prior_natural_fixed_bg_conditional_auc"] = max(errors["prior_natural_fixed_bg_conditional_auc"],error)
                endpoint_rows.append(dict(stage=stage,evaluation=evaluation,target=classes[target],partner=classes[partner],
                                          seed=seed,mask=draw,arm=arm,**dict(zip(METRICS,vals[0],strict=True))))
        for train in arms:
            if train==control:
                continue
            true = values[:,:,:,arms.index(train),ei].mean(axis=2)
            plc = values[:,:,:,arms.index(control),ei].mean(axis=2)
            targets = [(classes[t],classes[q]) for t,q,_ in pairs]
            levels = [("aggregate","",true.mean(axis=2),plc.mean(axis=2)),
                      *((t,q,true[:,:,j],plc[:,:,j]) for j,(t,q) in enumerate(targets))]
            for target,partner,first,second in levels:
                identity = dict(stage=stage,train=train,control=control,evaluation=evaluation,target=target,partner=partner)
                diff = 100*(first-second)
                np.testing.assert_allclose(diff[...,3]+diff[...,4],diff[...,5],atol=1e-12,rtol=0)
                for j,name in enumerate(METRICS):
                    paired_summary.append(dict(**identity,metric=name,**summarize(diff[...,j],indices),
                                               true_pp=float(100*first[0,:,j].mean()),plc_pp=float(100*second[0,:,j].mean())))
                    for seed,point in zip(seeds,diff[0,:,j],strict=True):
                        per_seed.append(dict(**identity,metric=name,seed=seed,delta_pp=point))
        print(stage,evaluation,"verified",round(perf_counter()-start,2),"seconds",flush=True)
    qa = dict(stage=stage,endpoints=len(keys),images=len(ids),bootstrap=limit,weight_and_init_match=True,
              full_decomposition_exact=True,**errors,seconds=perf_counter()-start)
    return paired_summary, per_seed, supports, endpoint_rows, qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates",type=int,choices=(8,N_BOOT),default=N_BOOT)
    parser.add_argument("--threads",type=int,default=6)
    args = parser.parse_args()
    set_num_threads(args.threads)
    summary, seeds, support, endpoints, qa = [],[],[],[],[]
    for stage in ("B","VOC"):
        a,b,c,d,e = replay(stage,args.replicates)
        summary.extend(a)
        seeds.extend(b)
        support.extend(c)
        endpoints.extend(d)
        qa.append(e)
    print(qa,flush=True)
    if args.replicates==N_BOOT:
        assert len(summary)==224 and all(r["valid"]==N_BOOT for r in summary)
        table("background_auc_B_VOC.csv",summary)
        table("background_auc_initializations.csv",seeds)
        table("background_auc_support.csv",support)
        table("background_auc_endpoints.csv",endpoints)
        save("background_auc_QA.json",qa)


if __name__ == "__main__":
    main()

