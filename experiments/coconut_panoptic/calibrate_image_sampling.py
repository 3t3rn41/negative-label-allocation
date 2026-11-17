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

"""Check weighted conditional AUC on a visible finite population with tied scores."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm


HERE=Path(__file__).resolve().parent
BINS=12
SEED=93120261115


def auc_by_counts(counts,quota,N):
    hist=counts.reshape(-1,2,2,BINS).astype(float)
    weighted=np.einsum("h,bhpk->bpk",N/quota,hist)
    p,n=weighted[:,0,:],weighted[:,1,:]
    a,b=p.sum(1),n.sum(1)
    assert (a>0).all() and (b>0).all()
    left=np.cumsum(n,axis=1)-.5*n
    right=np.cumsum(p[:,::-1],axis=1)[:,::-1]-.5*p
    theta=np.sum(p*left,axis=1)/(a*b)
    gradients=np.stack(((left/b[:,None]-theta[:,None])/a[:,None],
                        (right/a[:,None]-theta[:,None])/b[:,None]),1)
    first=np.einsum("bhpk,bpk->bh",hist,gradients)
    second=np.einsum("bhpk,bpk->bh",hist,gradients**2)
    var=np.sum(N**2*(1-quota/N)/quota * (second-first**2/quota)/(quota-1),1)
    return theta,np.sqrt(np.maximum(var,0))


def main(reps):
    start=perf_counter()
    plan=json.loads((HERE/"TARGET_REPAIR_STATUS.json").read_text(encoding="utf-8"))
    sampling=json.loads((HERE/"PANOPTIC_SAMPLE_QA.json").read_text(encoding="utf-8"))
    with np.load(HERE/"PANOPTIC_POPULATION133.npz",allow_pickle=False) as f:
        cats=f["categories"].tolist()
        y=f["truth"].astype(bool)
    pairs=[(cats.index(row["target_id"]),cats.index(p)) for row in plan["selected"] for p in row["partners"]]
    h=np.column_stack([y[:,t]&y[:,p] for t,p in pairs]).any(1).astype(np.int8)
    N=np.bincount(h,minlength=2)
    z=norm.ppf(.975)
    results=[]
    for j,(t,p) in enumerate(pairs):
        positive=y[:,t]&~y[:,p]
        negative=~y[:,t]&y[:,p]
        rng=np.random.default_rng(np.random.SeedSequence([SEED,30,j+1]))
        # Discrete synthetic scores yield ties while preserving the observed class support.
        v=(rng.normal(size=len(y))+.48*y[:,t]+.15*h+.04*j*y[:,p])
        bins=np.clip(np.floor(2*v+5),0,BINS-1).astype(int)
        cell=np.where(positive|negative,bins+BINS*negative,2*BINS)
        f=np.stack([np.bincount(cell[h==i],minlength=2*BINS+1) for i in range(2)])
        oracle=auc_by_counts(f[None,:,:-1],N,N)[0][0]
        for l,design in enumerate(sampling["designs"]):
            if design["total"] not in (10000,16000,30000) or design["fraction"] not in (.25,.60):
                continue
            quota=np.array(design["sample_counts"])
            draws=np.stack([rng.multivariate_hypergeometric(row,n,size=reps,method="marginals")
                            for row,n in zip(f,quota,strict=True)],axis=1)
            estimate,se=auc_by_counts(draws[:,:,:-1],quota,N)
            results.append(dict(target=cats[t],partner=cats[p],design=l,images=int(quota.sum()),quota=quota.tolist(),
                                replicates=reps,bias_pp=float(100*(estimate.mean()-oracle)),
                                coverage=float((abs(estimate-oracle)<=z*se).mean()),true_auc=float(oracle),
                                approximate_mean_halfwidth_pp=float(z*se.mean()*100)))
    result=dict(status="IMAGE_ONLY_LINEARIZATION_CALIBRATION",pairs=len(pairs),cases=len(results),
                min_coverage=min(x["coverage"] for x in results),max_abs_bias_pp=max(abs(x["bias_pp"]) for x in results),
                cases_meeting_coverage_and_bias=sum(x["coverage"]>=.93 and abs(x["bias_pp"])<=.05 for x in results),
                confidence_interval="Nominal 95% intervals using stratified finite-population linearization",
                scope="Synthetic image scores only. Crossed training, model-specific power and full bootstrap parity are pending.",
                confirmation_scores_read=0,wall_seconds=perf_counter()-start,rows=results)
    (HERE/"STATISTICS_IMAGE_CALIBRATION.json").write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k!="rows"},flush=True)


if __name__=="__main__":
    a=ArgumentParser()
    a.add_argument("--replicates",type=int,default=2500)
    main(a.parse_args().replicates)
