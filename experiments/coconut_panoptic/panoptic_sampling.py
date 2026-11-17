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

"""Run the frozen pure-label design simulations on the validated population."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import beta, hypergeom


HERE = Path(__file__).resolve().parent
SEED = 93120261115
REPLICATES = 2000
ORDER = np.array([2,1,3,0])


def groups(y, pairs):
    return np.column_stack([y[:,t]*2+y[:,p] for t,p in pairs])


def ess_gate(counts, weights, source_p):
    raw = counts.sum(0)
    weighted = np.einsum("h,hbre->bre",weights,counts)
    squared = np.einsum("h,hbre->bre",weights**2,counts)
    with np.errstate(divide="ignore",invalid="ignore"):
        natural_pos = weighted[:,:,0]+weighted[:,:,2]
        natural_neg = weighted[:,:,1]+weighted[:,:,3]
        natural_pos = natural_pos**2/(squared[:,:,0]+squared[:,:,2])
        natural_neg = natural_neg**2/(squared[:,:,1]+squared[:,:,3])
        terms = squared/weighted**2
        fixed_pos = 1/((1-source_p)**2*terms[:,:,0]+source_p**2*terms[:,:,2])
        fixed_neg = np.minimum(*[1/(r*r*terms[:,:,1]+(1-r)**2*terms[:,:,3]) for r in (.05,.5)])
    return ((raw>=np.array([50,50,30,200])).all((1,2)) &
            (np.minimum(natural_pos,fixed_pos)>=80).all(1) &
            (np.minimum(natural_neg,fixed_neg)>=250).all(1))


def main(data, designs):
    started = perf_counter()
    plan = json.loads(designs.read_text(encoding="utf-8"))
    with np.load(data,allow_pickle=False) as pool:
        cats,y = pool["categories"].tolist(),pool["truth"].astype(np.uint8)
    pair_cols = [(cats.index(row["target_id"]),cats.index(p)) for row in plan["chosen"] for p in row["partners"]]
    assert len(pair_cols) >= 8
    with np.load(HERE/"PANOPTIC_BRIDGE_LABELS.npz",allow_pickle=False) as bridge:
        assert bridge["categories"].tolist()==cats
        source = bridge["truth"]
    source_p = np.array([np.sum((source[:,t]==1)&(source[:,p]==1))/np.sum((source[:,t]==1)&(source[:,p]>=0))
                         for t,p in pair_cols])
    p = groups(y,pair_cols)
    strata = (p==3).any(1)
    N = np.bincount(strata,minlength=2)
    matrices, frequencies = [], []
    for h in range(2):
        cat_matrix = p[strata==h]
        cell, n = np.unique(cat_matrix,axis=0,return_counts=True)
        encode = np.eye(4,dtype=np.int64)[cell][:,:,ORDER]
        matrices.append(encode.reshape(len(cell),-1))
        frequencies.append(n)
    pos = np.stack([y[strata==h].sum(0) for h in range(2)]).astype(np.int64)
    result = []
    for j,n in enumerate((10000,12000,16000,20000,30000)):
        for k,q in enumerate((.25,.40,.50,.60)):
            n1 = min(int(N[1]), max(n-int(N[0]),round(q*n)))
            quota = np.array([n-n1,n1])
            assert (quota>0).all() and (quota<=N).all()
            rng = np.random.default_rng(np.random.SeedSequence([SEED,2,j*4+k]))
            mc = []
            for f, mat, nh in zip(frequencies,matrices,quota,strict=True):
                projected = rng.multivariate_hypergeometric(f,nh,size=REPLICATES,method="marginals") @ mat
                mc.append(projected.reshape(REPLICATES,len(pair_cols),4))
            passed = int(ess_gate(np.stack(mc),N/quota,source_p).sum())
            # Probability of losing AP support is bounded jointly across classes by exact hypergeometric marginals.
            ap_loss_bound = 0.
            for sign in (pos,N[:,None]-pos):
                loss = np.ones(len(cats))
                for h in range(2):
                    loss *= hypergeom.pmf(0,N[h],sign[h],quota[h])
                ap_loss_bound += float(loss.sum())
            lower = 0. if not passed else float(beta.ppf(.05,passed,REPLICATES-passed+1))
            qa = dict(total=n,fraction=q,strata_N=N.tolist(),sample_counts=quota.tolist(),reps=REPLICATES,
                      pair_support_pass=passed,ap_failure_probability_upper_bound=ap_loss_bound,
                      joint_pass_point_lower_bound=max(0.,passed/REPLICATES-ap_loss_bound),
                      joint_pass_95pct_lower_bound=max(0.,lower-ap_loss_bound),
                      passed=passed/REPLICATES-ap_loss_bound>=.97 and lower-ap_loss_bound>=.95)
            result.append(qa)
            print(n,q,passed,qa["passed"],flush=True)
    report = dict(status="PASS_LABEL_PLANNING" if any(r["passed"] for r in result) else "HOLD_SAMPLE_SUPPORT",
                  panel=plan["chosen"], class_count=len(cats), designs=result, real_sample_drawn=False,
                  evaluation_score_reads=0, method="Exact multivariate hypergeometric draws on distinct joint pair signatures. AP support enters the all-gates probability through a conservative exact union bound.",
                  wall_seconds=perf_counter()-started)
    (HERE/"PANOPTIC_SAMPLE_QA.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--data",type=Path,default=HERE/"PANOPTIC_POPULATION133.npz")
    parser.add_argument("--designs",type=Path,default=HERE/"PANOPTIC_MASK_SCREEN.json")
    main(**vars(parser.parse_args()))
